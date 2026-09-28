"""Read the official files, retaining row alignment and subject separation.

This is an integration adapter, not the data owner's PCA/clustering analysis.
Official test values are not even loaded unless include_test=True is explicit.
"""
from pathlib import Path
import numpy as np

CHANNELS = [f"{base}_{axis}" for base in ("body_acc", "body_gyro", "total_acc")
            for axis in ("x", "y", "z")]
CLASSES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
           "SITTING", "STANDING", "LAYING"]
VALIDATION_SUBJECTS = [1, 6, 14, 21, 23]
SPLIT_ID = "uci-har-subject-v1-val-1-6-14-21-23"


def _read_split(root, split):
    folder = Path(root) / split
    features = np.loadtxt(folder / f"X_{split}.txt", dtype=np.float32)
    labels = np.loadtxt(folder / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(folder / f"subject_{split}.txt", dtype=np.int64)
    sequence = np.stack([
        np.loadtxt(folder / "Inertial Signals" / f"{channel}_{split}.txt",
                   dtype=np.float32) for channel in CHANNELS
    ], axis=-1)
    expected = 7352 if split == "train" else 2947
    if features.shape != (expected, 561) or sequence.shape != (expected, 128, 9):
        raise ValueError(f"Unexpected official {split} input dimensions")
    if labels.shape != (expected,) or subjects.shape != (expected,):
        raise ValueError("Labels/subjects are not aligned with input rows")
    if set(labels.tolist()) != set(range(6)):
        raise ValueError("Expected all six original labels 1..6")
    if not np.isfinite(features).all() or not np.isfinite(sequence).all():
        raise ValueError("Non-finite sensor values")
    # Original row number is one-based, as in the team plan.
    sample_ids = np.array([f"{split}:{i + 1:06d}" for i in range(expected)])
    return dict(X_features=features, X_seq=sequence, y=labels,
                subject=subjects, sample_id=sample_ids)


def load_dataset(root, include_test=False):
    """Return (samples, split_indices), matching the team interface proposal."""
    root = Path(root)
    if (root / "UCI HAR Dataset").is_dir():
        root = root / "UCI HAR Dataset"
    samples = _read_split(root, "train")
    is_val = np.isin(samples["subject"], VALIDATION_SUBJECTS)
    indices = {"fit": np.flatnonzero(~is_val), "validation": np.flatnonzero(is_val),
               "test": np.array([], dtype=np.int64)}
    if len(indices["fit"]) != 5577 or len(indices["validation"]) != 1775:
        raise ValueError("Subject split differs from the agreed 5577/1775 split")
    if include_test:
        test = _read_split(root, "test")
        if set(samples["subject"]) & set(test["subject"]):
            raise ValueError("Official train/test subjects overlap")
        samples = {key: np.concatenate([samples[key], test[key]]) for key in samples}
        indices["test"] = np.arange(7352, 10299, dtype=np.int64)
    return samples, indices
