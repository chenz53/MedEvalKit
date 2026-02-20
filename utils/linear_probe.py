"""Feature extraction and linear-probe training for 3D vision encoders.

Provides:
- Encoder loading (SMB-Vision)
- Lazy ``VolumeDataset`` that loads NIfTI volumes on access via DataLoader workers
- Mean-pooled feature extraction with disk caching
- ``nn.Linear`` probe trained via HF Accelerate
- Multi-label classification metrics
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
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


class VolumeDataset(Dataset):
    """Lazy dataset: loads and patchifies NIfTI volumes on access.

    Each ``__getitem__`` call runs ``smb_utils.imaging_process.preprocess_image``
    in a DataLoader worker, returning ``(patches, grid_thw, labels)``.
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
            return patches, grid, torch.from_numpy(labels)
        except Exception:
            return None


def collate_volumes(batch):
    """Collate patchified volumes, dropping failed samples."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return (
        torch.cat([b[0] for b in batch], dim=0),
        torch.stack([b[1] for b in batch], dim=0),
        torch.stack([b[2] for b in batch], dim=0),
    )


# ── Feature extraction ───────────────────────────────────────────────


def extract_features(encoder, records, preprocess_kwargs, device, batch_size=2, num_workers=4):
    """Extract mean-pooled features from NIfTI volumes using lazy loading.

    Volumes are loaded and patchified in DataLoader workers (parallel I/O),
    then encoded on GPU.  Only the per-volume mean-pooled feature vectors
    are retained.

    Returns ``(features, labels)`` as numpy arrays ``[N, D]`` and ``[N, L]``.
    """
    dataset = VolumeDataset(records, preprocess_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_volumes,
        pin_memory=("cuda" in str(device)),
        shuffle=False,
    )

    dtype = next(encoder.parameters()).dtype
    all_feats, all_labs = [], []

    for batch in tqdm(loader, desc="Extracting features"):
        if batch is None:
            continue
        patches, grids, labels = batch
        patches = patches.to(device=device, dtype=dtype)
        grids = grids.to(device=device)

        with torch.no_grad():
            encoded, _ = encoder(patches, grid_thw=grids)

        idx = 0
        for g in grids:
            n = g[0].item() * g[1].item() * g[2].item()
            all_feats.append(encoded[idx : idx + n].float().mean(0).cpu())
            idx += n
        all_labs.append(labels)

    features = torch.stack(all_feats).numpy()
    labels = torch.cat(all_labs).numpy()
    skipped = len(records) - len(features)
    if skipped:
        print(f"  Skipped {skipped}/{len(records)} volumes due to errors")
    return features, labels


# ── Linear classification head ───────────────────────────────────────


class LinearProbeHead(nn.Module):
    """Single ``nn.Linear`` layer for multi-label classification."""

    def __init__(self, in_features: int, num_labels: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ── Training (Accelerate-native) ────────────────────────────────────


def train_linear_head(
    train_features,
    train_labels,
    val_features,
    val_labels,
    accelerator,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    seed=42,
):
    """Train a linear classification head on cached features.

    Uses HF :class:`~accelerate.Accelerator` for device placement,
    distributed training, and mixed-precision support.
    """
    torch.manual_seed(seed)

    head = LinearProbeHead(train_features.shape[1], train_labels.shape[1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_features).float(),
            torch.from_numpy(train_labels).float(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    head, optimizer, loader = accelerator.prepare(head, optimizer, loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pos_counts = train_labels.sum(axis=0).clip(min=1)
    pw = torch.tensor(
        (len(train_labels) - pos_counts) / pos_counts,
        dtype=torch.float32,
        device=accelerator.device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    log_every = max(1, epochs // 10)
    for epoch in range(epochs):
        head.train()
        total, count = 0.0, 0
        for xb, yb in loader:
            loss = criterion(head(xb), yb)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        scheduler.step()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            msg = f"  Epoch {epoch + 1:4d}/{epochs}  loss={total / count:.4f}"
            if val_features is not None:
                vl = _val_loss(
                    accelerator.unwrap_model(head),
                    val_features,
                    val_labels,
                    criterion,
                    accelerator.device,
                )
                msg += f"  val_loss={vl:.4f}"
            accelerator.print(msg)

    return accelerator.unwrap_model(head).eval()


def _val_loss(head, vf, vl, criterion, device):
    head.eval()
    with torch.no_grad():
        return criterion(
            head(torch.from_numpy(vf).float().to(device)),
            torch.from_numpy(vl).float().to(device),
        ).item()


# ── Evaluation ───────────────────────────────────────────────────────


def evaluate_linear_head(head, val_features, val_labels, label_names=None, device="cuda"):
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

    head = head.to(device).eval()
    with torch.no_grad():
        probs = (
            torch.sigmoid(head(torch.from_numpy(val_features).float().to(device)))
            .cpu()
            .numpy()
        )
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
