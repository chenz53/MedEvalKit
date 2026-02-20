"""CT-RATE dataset adapter for linear-probe evaluation of 3D vision encoders.

Expects a local copy of the CT-RATE dataset
(https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
with multi-abnormality labels in CSV format and NIfTI CT volumes.

Supported directory layouts (auto-detected)::

    # Layout A – nested (official HuggingFace download)
    dataset_path/
      train/
        train_<pid>/<scan>/<volume>.nii.gz
      valid/
        valid_<pid>/<scan>/<volume>.nii.gz
      multi_abnormality_labels/
        train_predicted_labels.csv
        valid_predicted_labels.csv

    # Layout B – flat
    dataset_path/
      train/ or volumes_train/   *.nii.gz
      valid/ or volumes_valid/   *.nii.gz
      *train*label*.csv
      *valid*label*.csv

Records are ``(volume_name, nifti_path, labels_array)`` tuples consumed
by :class:`~utils.linear_probe.VolumeDataset` for lazy loading.
"""

import glob
import os

import numpy as np
import pandas as pd

from ..base_dataset import BaseDataset


class CT_RATE_LinearProbe(BaseDataset):

    ABNORMALITY_LABELS = [
        "Medical material",
        "Arterial wall calcification",
        "Cardiomegaly",
        "Pericardial effusion",
        "Coronary artery wall calcification",
        "Hiatal hernia",
        "Lymphadenopathy",
        "Emphysema",
        "Atelectasis",
        "Lung nodule",
        "Lung opacity",
        "Pulmonary fibrotic sequela",
        "Pleural effusion",
        "Mosaic attenuation pattern",
        "Peribronchial thickening",
        "Consolidation",
        "Bronchiectasis",
        "Interlobular septal thickening",
    ]

    def __init__(
        self,
        dataset_path,
        max_train_samples=None,
        max_val_samples=None,
        **_kwargs,
    ):
        super().__init__()
        self.dataset_path = dataset_path or "./datas/CT_RATE"
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples

        self.train_records = []
        self.val_records = []
        self.label_names = list(self.ABNORMALITY_LABELS)
        self.samples = []

    # ── Data loading ─────────────────────────────────────────────────

    def load_data(self):
        """Discover label CSVs and match them against NIfTI volumes."""
        train_csv, val_csv = self._find_label_csvs()
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)

        self.label_names = self._detect_label_columns(train_df)
        print(f"Detected {len(self.label_names)} abnormality labels")

        self.train_records = self._match_records(
            train_df, self._build_volume_map("train"), self.max_train_samples
        )
        self.val_records = self._match_records(
            val_df, self._build_volume_map("valid"), self.max_val_samples
        )

        print(
            f"CT-RATE loaded: {len(self.train_records)} train, "
            f"{len(self.val_records)} val volumes"
        )
        return self.train_records

    # ── File discovery ───────────────────────────────────────────────

    def _find_label_csvs(self):
        root = self.dataset_path
        candidates = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)

        train_csv = val_csv = None
        for p in candidates:
            low = os.path.basename(p).lower()
            if "train" in low and "label" in low:
                train_csv = p
            elif ("valid" in low or "val" in low) and "label" in low:
                val_csv = p

        if train_csv is None or val_csv is None:
            for p in candidates:
                low = os.path.basename(p).lower()
                if "train" in low and train_csv is None:
                    train_csv = p
                elif ("valid" in low or "val" in low) and val_csv is None:
                    val_csv = p

        if train_csv is None or val_csv is None:
            raise FileNotFoundError(
                f"Could not find train/val label CSVs under {root}. "
                f"Found: {[os.path.basename(c) for c in candidates]}"
            )

        print(f"Train labels: {train_csv}")
        print(f"Val labels:   {val_csv}")
        return train_csv, val_csv

    def _detect_label_columns(self, df):
        present = [c for c in self.ABNORMALITY_LABELS if c in df.columns]
        if present:
            return present
        id_like = {"volumename", "volume_name", "id", "filename", "name", "index"}
        return [
            c
            for c in df.columns
            if c.strip().lower() not in id_like
            and df[c].dropna().isin([0, 1, 0.0, 1.0]).all()
        ]

    def _build_volume_map(self, split_prefix):
        vol_map = {}
        root = self.dataset_path
        for search_dir in [
            os.path.join(root, split_prefix),
            os.path.join(root, "dataset", split_prefix),
            os.path.join(root, f"volumes_{split_prefix}"),
            root,
        ]:
            if not os.path.isdir(search_dir):
                continue
            for ext in ("*.nii.gz", "*.nii"):
                for path in glob.glob(
                    os.path.join(search_dir, "**", ext), recursive=True
                ):
                    base = os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")
                    vol_map.setdefault(base, path)
            if vol_map:
                break
        return vol_map

    def _match_records(self, df, vol_map, max_samples):
        id_col = self._detect_id_column(df)
        records = []
        for _, row in df.iterrows():
            name = str(row[id_col]).strip().replace(".nii.gz", "").replace(".nii", "")
            path = vol_map.get(name)
            if path is None:
                continue
            labels = np.array(
                [float(row[c]) for c in self.label_names], dtype=np.float32
            )
            records.append((name, path, labels))
            if max_samples and len(records) >= max_samples:
                break
        return records

    def _detect_id_column(self, df):
        for c in ["VolumeName", "volume_name", "Volume_Name", "filename", "id", "ID", "Name"]:
            if c in df.columns:
                return c
        return df.columns[0]

    # ── Compatibility stubs ──────────────────────────────────────────

    def construct_messages(self, sample):
        return sample

    def cal_metrics(self, out_samples):
        return {}, out_samples
