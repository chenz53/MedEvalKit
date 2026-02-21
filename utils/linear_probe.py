"""Feature extraction and linear-probe training for 3D vision encoders.

Provides:
- Encoder loading (SMB-Vision)
- Lazy ``VolumeDataset`` that loads NIfTI volumes on access via DataLoader workers
- Mean-pooled feature extraction with disk caching
- Scan-level aggregation (averaging reconstructions of the same scan)
- ``LinearClassifier`` compatible with HF Trainer for multi-label probing
- Multi-label classification metrics
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

# ── Encoder ──────────────────────────────────────────────────────────


def load_encoder(model_path, device="cuda"):
    """Load a 3-D vision encoder from HuggingFace.

    Tries Flash-Attention 2 → SDPA → eager, loads in bf16 on CUDA.
    Returns the ``SMBVisionEncoder`` ready for feature extraction.
    """
    from transformers import AutoModel

    dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32
    for attn in ("flash_attention_2", "sdpa", "eager"):
        try:
            full = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                attn_implementation=attn,
            )
            break
        except Exception:
            if attn == "eager":
                raise
    encoder = full.encoder.to(device).eval()
    del full
    return encoder


# ── Lazy-loading volume dataset ──────────────────────────────────────


class VolumeDataset(TorchDataset):
    """Lazy dataset: loads and patchifies NIfTI volumes on access.

    Each ``__getitem__`` returns ``(name, patches, grid_thw, labels)``.
    Failed samples return ``None`` and are filtered by :func:`collate_volumes`.
    """

    def __init__(self, records, preprocess_kwargs):
        self.records = records
        self.pp = preprocess_kwargs

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        name, path, labels = self.records[idx]
        try:
            from smb_utils.imaging_process import preprocess_image

            patches, grid = preprocess_image(path, modality="CT", **self.pp)
            return name, patches, grid, torch.from_numpy(labels)
        except Exception:
            return None


def collate_volumes(batch):
    """Collate patchified volumes, dropping failed samples."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    names = [b[0] for b in batch]
    return (
        names,
        torch.cat([b[1] for b in batch], dim=0),
        torch.stack([b[2] for b in batch], dim=0),
        torch.stack([b[3] for b in batch], dim=0),
    )


# ── Feature extraction ───────────────────────────────────────────────


def extract_features(encoder, records, preprocess_kwargs, device, batch_size=1, num_workers=4):
    """Extract mean-pooled features from NIfTI volumes using lazy loading.

    Uses batch_size=1 with high prefetch_factor so workers stay ahead
    of GPU processing, hiding I/O latency.

    Returns ``(features, labels, names)`` as numpy arrays.
    """
    dataset = VolumeDataset(records, preprocess_kwargs)
    use_cuda = "cuda" in str(device)
    prefetch = max(4, num_workers * 4)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=collate_volumes,
        pin_memory=use_cuda,
        shuffle=False,
    )

    dtype = next(encoder.parameters()).dtype
    all_names, all_feats, all_labs = [], [], []

    for batch in tqdm(loader, desc="Extracting features"):
        if batch is None:
            continue
        names, patches, grids, labels = batch
        patches = patches.to(device=device, dtype=dtype)
        grids = grids.to(device=device)

        with torch.no_grad():
            encoded, _ = encoder(patches, grid_thw=grids)

        idx = 0
        for i, g in enumerate(grids):
            n = g[0].item() * g[1].item() * g[2].item()
            all_feats.append(encoded[idx : idx + n].float().mean(0).cpu())
            all_names.append(names[i])
            idx += n
        all_labs.append(labels)

    features = torch.stack(all_feats).numpy()
    labels = torch.cat(all_labs).numpy()
    names_arr = np.array(all_names)
    skipped = len(records) - len(features)
    if skipped:
        print(f"  Skipped {skipped}/{len(records)} volumes due to errors")
    return features, labels, names_arr


# ── Scan-level aggregation ───────────────────────────────────────────


def scan_key(volume_name: str) -> str:
    """Derive scan key by stripping the reconstruction suffix.

    ``train_53_a_1`` → ``train_53_a``  (split_patient_scan)
    """
    parts = volume_name.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else volume_name


def aggregate_by_scan(features, labels, names):
    """Average reconstruction features per scan; take first label (identical).

    Returns ``(scan_features, scan_labels, scan_keys)`` with one row per scan.
    """
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for i, n in enumerate(names):
        groups.setdefault(scan_key(n), []).append(i)

    agg_f, agg_l, agg_k = [], [], []
    for key, idxs in groups.items():
        agg_f.append(features[idxs].mean(axis=0))
        agg_l.append(labels[idxs[0]])
        agg_k.append(key)

    n_recons = len(names)
    n_scans = len(agg_k)
    print(f"  Aggregated {n_recons} reconstructions → {n_scans} scans")
    return np.stack(agg_f), np.stack(agg_l), np.array(agg_k)


# ── Trainer-compatible linear classifier ─────────────────────────────


class LinearClassifier(nn.Module):
    """Single ``nn.Linear`` for multi-label classification.

    Compatible with HF ``Trainer``: ``forward`` returns
    ``{"loss": ..., "logits": ...}`` when labels are provided.
    """

    def __init__(self, hidden_size: int, num_labels: int, pos_weight=None):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)
        if pos_weight is not None:
            self.register_buffer(
                "pos_weight", torch.tensor(pos_weight, dtype=torch.float32)
            )
        else:
            self.pos_weight = None

    def forward(self, inputs_embeds, labels=None):
        logits = self.classifier(inputs_embeds)
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=self.pos_weight
            )
        return {"loss": loss, "logits": logits}


# ── HF Dataset / metrics helpers ─────────────────────────────────────


def build_feature_dataset(features, labels):
    """Create an HF ``Dataset`` from numpy feature and label arrays."""
    return Dataset.from_dict(
        {"inputs_embeds": features.tolist(), "labels": labels.tolist()}
    ).with_format("torch")


def build_compute_metrics():
    """Return a ``compute_metrics`` callable for HF Trainer (multi-label)."""
    from sklearn.metrics import f1_score, roc_auc_score

    def _safe(fn, *a, **kw):
        try:
            return float(fn(*a, **kw))
        except ValueError:
            return None

    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        return {
            "macro_f1": float(
                f1_score(label_ids, preds, average="macro", zero_division=0)
            ),
            "micro_f1": float(
                f1_score(label_ids, preds, average="micro", zero_division=0)
            ),
            "micro_auc_roc": _safe(roc_auc_score, label_ids, probs, average="micro"),
        }

    return compute_metrics


# ── Detailed evaluation ──────────────────────────────────────────────


def evaluate_linear_head(model, val_features, val_labels, label_names=None, device="cuda"):
    """Compute multi-label classification metrics.

    Returns ``{"aggregate": {...}, "per_label": {...}}``.
    """
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model = model.to(device).eval()
    with torch.no_grad():
        output = model(torch.from_numpy(val_features).float().to(device))
        logits = output["logits"] if isinstance(output, dict) else output
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    n = val_labels.shape[1]
    names = label_names or [f"label_{i}" for i in range(n)]

    def _safe(fn, *a, **kw):
        try:
            return float(fn(*a, **kw))
        except ValueError:
            return None

    per_label = {}
    for i, name in enumerate(names):
        yt, pp, pd = val_labels[:, i], probs[:, i], preds[:, i]
        per_label[name] = {
            "auc_roc": _safe(roc_auc_score, yt, pp),
            "avg_precision": _safe(average_precision_score, yt, pp),
            "f1": float(f1_score(yt, pd, zero_division=0)),
            "precision": float(precision_score(yt, pd, zero_division=0)),
            "recall": float(recall_score(yt, pd, zero_division=0)),
            "accuracy": float(accuracy_score(yt, pd)),
            "positive_count": int(yt.sum()),
            "total_count": len(yt),
        }

    va = [v["auc_roc"] for v in per_label.values() if v["auc_roc"] is not None]
    vp = [v["avg_precision"] for v in per_label.values() if v["avg_precision"] is not None]

    aggregate = {
        "macro_auc_roc": float(np.mean(va)) if va else None,
        "mean_avg_precision": float(np.mean(vp)) if vp else None,
        "macro_f1": float(f1_score(val_labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(val_labels, preds, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(val_labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(val_labels, preds, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(val_labels, preds, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(val_labels, preds, average="micro", zero_division=0)),
        "micro_auc_roc": _safe(roc_auc_score, val_labels, probs, average="micro"),
        "num_samples": len(val_labels),
        "num_labels": n,
    }

    return {"aggregate": aggregate, "per_label": per_label}
