"""Load only fit/val from the team's unstandardized, already split cache."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import numpy as np
from ..contracts import TrainingData, row_binding_digest

SCHEMA = "team_sequence_npz_v1"
EXPECTED_DATA_SHA256 = "00e8eae9edbcf37e1b01bb5a76b4f1da7bd21ca8d42a9beeccf4a483e7d3e459"
EXPECTED_STATS_SHA256 = "42b1b895d420a21ed7cce3b3636a4cbed96350953b27057c0670db9062d79fb2"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


@dataclass
class TeamSequencePreprocessor:
    mean: np.ndarray
    scale: np.ndarray
    input_kind: str = "sequence"

    def __post_init__(self):
        for name, value in (("mean", self.mean), ("scale", self.scale)):
            if value.shape != (1, 1, 9) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Invalid team {name}: expected finite float32 (1,1,9)")
        if np.any(self.scale <= 0):
            raise ValueError("Team scale must be positive")

    def transform(self, raw):
        raw = np.asarray(raw)
        if raw.dtype != np.float32 or raw.ndim != 3 or raw.shape[1:] != (128, 9) or len(raw) == 0:
            raise ValueError("Expected nonempty float32 (N,128,9) input")
        if not np.isfinite(raw).all():
            raise ValueError("Input contains NaN or Inf")
        transformed = (raw - self.mean) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("Standardization produced non-finite values")
        return transformed

    def save(self, path):
        np.savez_compressed(path, schema=np.array(SCHEMA), seq_mean=self.mean,
                            seq_std=self.scale, input_state=np.array("unstandardized"))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if "schema" in z and str(z["schema"]) != SCHEMA:
                raise ValueError("Unsupported team preprocessing schema")
            return cls(z["seq_mean"], z["seq_std"])


def load_processed_data(cache_path, stats_path, *, smoke=False,
                        expected_data_sha256=None, expected_stats_sha256=None):
    data_hash, stats_hash = sha256(cache_path), sha256(stats_path)
    if expected_data_sha256 and data_hash != expected_data_sha256:
        raise ValueError("Processed data checksum differs from the frozen source")
    if expected_stats_sha256 and stats_hash != expected_stats_sha256:
        raise ValueError("Preprocessing checksum differs from the frozen source")
    pre = TeamSequencePreprocessor.load(stats_path)
    accessed = []
    arrays = {}
    # NpzFile is lazy. No test_* array is indexed or materialized here.
    with np.load(cache_path, allow_pickle=False) as z:
        for part in ("fit", "val"):
            for suffix in ("Xs", "y", "subject", "sample_id"):
                key = f"{part}_{suffix}"
                arrays[key] = z[key]
                accessed.append(key)
    for part in ("fit", "val"):
        x, y, ids = (arrays[f"{part}_{key}"] for key in ("Xs", "y", "sample_id"))
        pre.transform(x)  # Input validation before any statistics are inspected.
        if y.dtype != np.int64 or y.shape != (len(x),) or not np.isin(y, range(6)).all():
            raise ValueError("Labels must be aligned int64 0..5")
        if set(y.tolist()) != set(range(6)):
            raise ValueError("Each development split must contain all six classes")
        if ids.shape != (len(x),) or ids.dtype.kind != "U" or any(
                re.fullmatch(r"train:\d{6}", str(sid)) is None for sid in ids):
            raise ValueError("Expected zero-based train:NNNNNN sample IDs")
    # Verify existing statistics, never fit a new preprocessor.
    raw_fit = arrays["fit_Xs"]
    mean = raw_fit.mean(axis=(0, 1), keepdims=True)
    scale = np.maximum(raw_fit.std(axis=(0, 1), keepdims=True), np.float32(1e-8))
    if not (np.array_equal(mean, pre.mean) and np.array_equal(scale, pre.scale)):
        raise ValueError("Saved statistics do not match the unstandardized full fit cache")
    provenance = {
        "preprocessing": SCHEMA, "sample_row_number_base": 0,
        "data_sha256": data_hash, "source_stats_sha256": stats_hash,
        "data_filename": Path(cache_path).name, "source_stats_filename": Path(stats_path).name,
        "input_state": "unstandardized", "standardization_applications": 1,
        "statistics_refitted": False, "statistics_match_full_fit": True,
        "loaded_array_keys": accessed, "test_arrays_loaded": False,
    }
    if smoke:
        for part in ("fit", "val"):
            y = arrays[f"{part}_y"]
            rows = np.concatenate([np.flatnonzero(y == c)[:10 if part == "fit" else 5] for c in range(6)])
            for suffix in ("Xs", "y", "subject", "sample_id"):
                key = f"{part}_{suffix}"
                arrays[key] = arrays[key][rows]
    data = TrainingData(pre.transform(arrays["fit_Xs"]), arrays["fit_y"],
                        pre.transform(arrays["val_Xs"]), arrays["val_y"],
                        arrays["fit_sample_id"], arrays["val_sample_id"],
                        arrays["fit_subject"], arrays["val_subject"], pre,
                        arrays["val_Xs"], split_id="team_s0_val_1_6_14_21_23", smoke=smoke,
                        provenance=provenance)
    provenance["fit_row_binding_sha256"] = row_binding_digest(
        data.x_fit, data.y_fit, data.fit_ids, data.fit_subjects)
    provenance["validation_row_binding_sha256"] = row_binding_digest(
        data.raw_validation, data.y_validation, data.validation_ids, data.validation_subjects)
    data.validate()
    return data
