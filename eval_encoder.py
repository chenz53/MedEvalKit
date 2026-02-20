"""Evaluate a 3D vision encoder via linear probing.

Default: ``smb-vision-v1`` on CT-RATE.  Volumes are lazy-loaded via
DataLoader workers.  Pooled features are cached to disk so re-runs
only retrain the linear head.

    accelerate launch eval_encoder.py --dataset_path ./datas/CT_RATE
"""

import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

DEFAULT_MODEL = "standardmodelbio/smb-vision-v1"


def parse_args():
    p = ArgumentParser(description="3D vision encoder linear-probe evaluation")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--dataset", default="CT_RATE")
    p.add_argument("--dataset_path", default="./datas/CT_RATE")
    p.add_argument("--output_path", default=None)
    # extraction
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--force_extract", action="store_true")
    # training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--head_batch_size", type=int, default=256)
    # general
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    # preprocessing (smb-utils CT defaults)
    p.add_argument("--spatial_size", type=int, nargs=3, default=[416, 416, 192])
    p.add_argument("--pixdim", type=float, nargs=3, default=[1.0, 1.0, 2.0])
    p.add_argument("--a_min", type=float, default=-1000.0)
    p.add_argument("--a_max", type=float, default=1000.0)
    return p.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    set_seed(args.seed)

    output = args.output_path or os.path.join(
        "eval_results", "encoder", os.path.basename(args.model_path), args.dataset
    )
    cache_dir = os.path.join(output, "feature_cache")

    # ── Dataset ──
    dataset = _load_dataset(args)

    # ── Features (extract on main process → cache → all processes load) ──
    if accelerator.is_main_process:
        _ensure_features(args, dataset, cache_dir, str(accelerator.device))
    accelerator.wait_for_everyone()

    train_f = np.load(os.path.join(cache_dir, "train_features.npy"))
    train_l = np.load(os.path.join(cache_dir, "train_labels.npy"))
    val_f = np.load(os.path.join(cache_dir, "valid_features.npy"))
    val_l = np.load(os.path.join(cache_dir, "valid_labels.npy"))
    accelerator.print(
        f"Features: train {train_f.shape}, val {val_f.shape}, "
        f"labels {len(dataset.label_names)}"
    )

    # ── Train ──
    from utils.linear_probe import train_linear_head

    accelerator.print("Training linear head ...")
    head = train_linear_head(
        train_f, train_l, val_f, val_l, accelerator,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.head_batch_size, seed=args.seed,
    )

    # ── Evaluate & save ──
    if accelerator.is_main_process:
        from utils.linear_probe import evaluate_linear_head

        device = next(head.parameters()).device
        metrics = evaluate_linear_head(head, val_f, val_l, dataset.label_names, device)
        metrics["config"] = {
            k: v for k, v in vars(args).items() if not k.startswith("_")
        }

        os.makedirs(output, exist_ok=True)
        torch.save(head.state_dict(), os.path.join(output, "linear_head.pt"))
        with open(os.path.join(output, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        _print_summary(metrics, output)


# ── Helpers ──────────────────────────────────────────────────────────


def _load_dataset(args):
    from utils.CT_RATE.CT_RATE_LinearProbe import CT_RATE_LinearProbe

    ds = CT_RATE_LinearProbe(
        dataset_path=args.dataset_path,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    ds.load_data()
    return ds


def _ensure_features(args, dataset, cache_dir, device):
    """Extract and cache pooled features for any splits not already on disk."""
    os.makedirs(cache_dir, exist_ok=True)

    def _cached(split):
        return (
            not args.force_extract
            and os.path.exists(os.path.join(cache_dir, f"{split}_features.npy"))
            and os.path.exists(os.path.join(cache_dir, f"{split}_labels.npy"))
        )

    splits = [
        (s, r)
        for s, r in [("train", dataset.train_records), ("valid", dataset.val_records)]
        if not _cached(s)
    ]
    if not splits:
        return

    from utils.linear_probe import extract_features, load_encoder

    print(f"Loading encoder: {args.model_path}")
    encoder = load_encoder(args.model_path, device)

    pp = dict(
        spatial_size=tuple(args.spatial_size),
        pixdim=tuple(args.pixdim),
        a_min=args.a_min,
        a_max=args.a_max,
    )

    for split, records in splits:
        feats, labs = extract_features(
            encoder, records, pp, device, args.batch_size, args.num_workers
        )
        np.save(os.path.join(cache_dir, f"{split}_features.npy"), feats)
        np.save(os.path.join(cache_dir, f"{split}_labels.npy"), labs)
        print(f"Cached {split} features ({feats.shape}) → {cache_dir}")

    del encoder
    torch.cuda.empty_cache()


def _print_summary(metrics, output):
    agg = metrics["aggregate"]
    print(f"\n{'=' * 60}\nRESULTS  ({output})\n{'=' * 60}")
    for k, v in agg.items():
        if v is None:
            continue
        print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")
    per = metrics.get("per_label", {})
    if per:
        print(f"\n{'PER-LABEL':─<60}")
        for name, m in per.items():
            if m.get("auc_roc") is not None:
                print(
                    f"  {name:40s}  AUC={m['auc_roc']:.4f}  "
                    f"F1={m['f1']:.4f}  n+={m['positive_count']}"
                )
    print("=" * 60)


if __name__ == "__main__":
    main()
