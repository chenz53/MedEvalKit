"""Evaluate a 3D vision encoder via linear probing.

Default: ``smb-vision-v1`` on CT-RATE.  Volumes are lazy-loaded via
DataLoader workers.  Pooled features are cached to disk so re-runs
only retrain the linear head.

Uses HF ``Trainer`` for the linear-probe training step.

    accelerate launch eval_encoder.py --dataset_path ./datas/CT_RATE
"""

import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
from accelerate.utils import set_seed

DEFAULT_MODEL = "standardmodelbio/smb-vision-v1"


def parse_args():
    p = ArgumentParser(description="3D vision encoder linear-probe evaluation")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--dataset", default="CT_RATE")
    p.add_argument("--dataset_path", default="./datas/CT_RATE")
    p.add_argument("--train_label_csv", default=None, help="Path to train labels CSV")
    p.add_argument("--val_label_csv", default=None, help="Path to val labels CSV")
    p.add_argument("--output_path", default=None)
    # extraction
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
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
    set_seed(args.seed)

    output = args.output_path or os.path.join(
        "eval_results", "encoder", os.path.basename(args.model_path), args.dataset
    )
    cache_dir = os.path.join(output, "feature_cache")

    # ── Dataset ──
    dataset = _load_dataset(args)

    # ── Features (extract → aggregate by scan → cache) ──
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if local_rank == 0:
        _ensure_features(args, dataset, cache_dir, device)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        torch.distributed.barrier()

    from utils.linear_probe import aggregate_by_scan

    train_f = np.load(os.path.join(cache_dir, "train_features.npy"))
    train_l = np.load(os.path.join(cache_dir, "train_labels.npy"))
    train_n = np.load(os.path.join(cache_dir, "train_names.npy"))
    train_f, train_l, _ = aggregate_by_scan(train_f, train_l, train_n)

    val_f = np.load(os.path.join(cache_dir, "valid_features.npy"))
    val_l = np.load(os.path.join(cache_dir, "valid_labels.npy"))
    val_n = np.load(os.path.join(cache_dir, "valid_names.npy"))
    val_f, val_l, _ = aggregate_by_scan(val_f, val_l, val_n)

    print(
        f"Scan features: train {train_f.shape}, val {val_f.shape}, "
        f"labels {len(dataset.label_names)}"
    )

    # ── Train via HF Trainer ──
    from transformers import Trainer, TrainingArguments

    from utils.linear_probe import (
        LinearClassifier,
        build_compute_metrics,
        build_feature_dataset,
    )

    train_ds = build_feature_dataset(train_f, train_l)
    val_ds = build_feature_dataset(val_f, val_l)

    pos_counts = train_l.sum(axis=0).clip(min=1)
    pos_weight = (len(train_l) - pos_counts) / pos_counts

    model = LinearClassifier(
        hidden_size=train_f.shape[1],
        num_labels=train_l.shape[1],
        pos_weight=pos_weight,
    )

    trainer_dir = os.path.join(output, "trainer")
    training_args = TrainingArguments(
        output_dir=trainer_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.head_batch_size,
        per_device_eval_batch_size=args.head_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="constant",
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="micro_auc_roc",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=5,
        seed=args.seed,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=f"linear-probe-{args.dataset}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=build_compute_metrics(),
    )

    print("Training linear head ...")
    trainer.train()
    print("Loaded best checkpoint (by eval micro_f1)")

    # ── Detailed per-label evaluation & save ──
    if local_rank == 0:
        from utils.linear_probe import evaluate_linear_head

        model_device = next(model.parameters()).device
        metrics = evaluate_linear_head(
            model, val_f, val_l, dataset.label_names, model_device
        )
        metrics["config"] = {
            k: v for k, v in vars(args).items() if not k.startswith("_")
        }

        os.makedirs(output, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output, "linear_head.pt"))
        with open(os.path.join(output, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        _print_summary(metrics, output)


# ── Helpers ──────────────────────────────────────────────────────────


def _load_dataset(args):
    from utils.CT_RATE.CT_RATE_LinearProbe import CT_RATE_LinearProbe

    ds = CT_RATE_LinearProbe(
        dataset_path=args.dataset_path,
        train_label_csv=args.train_label_csv,
        val_label_csv=args.val_label_csv,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    ds.load_data()
    return ds


def _ensure_features(args, dataset, cache_dir, device):
    """Extract, aggregate by scan, and cache pooled features."""
    os.makedirs(cache_dir, exist_ok=True)

    def _cached(split):
        return (
            not args.force_extract
            and os.path.exists(os.path.join(cache_dir, f"{split}_features.npy"))
            and os.path.exists(os.path.join(cache_dir, f"{split}_labels.npy"))
            and os.path.exists(os.path.join(cache_dir, f"{split}_names.npy"))
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
        feats, labs, names = extract_features(
            encoder, records, pp, device, args.batch_size, args.num_workers
        )
        np.save(os.path.join(cache_dir, f"{split}_features.npy"), feats)
        np.save(os.path.join(cache_dir, f"{split}_labels.npy"), labs)
        np.save(os.path.join(cache_dir, f"{split}_names.npy"), names)
        print(f"Cached {split} scan features ({feats.shape}) → {cache_dir}")

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
