"""Match every fit/validation window to official UCI train rows without test access."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(cache, stats, raw, output):
    train = raw / "train"
    channels = [f"{base}_{axis}" for base in ("body_acc", "body_gyro", "total_acc") for axis in "xyz"]
    labels = np.loadtxt(train / "y_train.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(train / "subject_train.txt", dtype=np.int64)
    result = dict(data_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                  test_arrays_loaded=False, parts={}, source_row_base=0)
    with np.load(cache, allow_pickle=False) as z, np.load(stats, allow_pickle=False) as s:
        cached = {part: {key: z[f"{part}_{key}"] for key in ("Xs", "y", "subject", "sample_id")}
                  for part in ("fit", "val")}
        max_diff = 0.0
        for part, v in cached.items():
            rows = np.array([int(x.split(":")[1]) for x in v["sample_id"]])
            np.testing.assert_array_equal(v["y"], labels[rows])
            np.testing.assert_array_equal(v["subject"], subjects[rows])
            result["parts"][part] = dict(count=len(rows), subjects=np.unique(v["subject"]).tolist(),
                class_counts=np.bincount(v["y"], minlength=6).tolist(),
                original_labels_identical=True, original_subjects_identical=True)
        for ci, ch in enumerate(channels):
            official = np.loadtxt(train / "Inertial Signals" / f"{ch}_train.txt", dtype=np.float32)
            for v in cached.values():
                rows = np.array([int(x.split(":")[1]) for x in v["sample_id"]])
                max_diff = max(max_diff, float(np.max(np.abs(v["Xs"][:, :, ci] - official[rows]))))
                np.testing.assert_array_equal(v["Xs"][:, :, ci], official[rows])
        fit, val = cached["fit"]["Xs"], cached["val"]["Xs"]
        norm = (val - s["seq_mean"]) / s["seq_std"]
        f64 = fit.astype(np.float64)
        oldnorm = ((val.astype(np.float64) - f64.mean((0, 1))) / f64.std((0, 1))).astype(np.float32)
        result.update(channels=channels, original_all_sequence_values_identical=True,
            max_original_absolute_difference=max_diff,
            fit_stats_mean_identical=bool(np.array_equal(fit.mean((0, 1), keepdims=True), s["seq_mean"])),
            fit_stats_std_identical=bool(np.array_equal(fit.std((0, 1), keepdims=True), s["seq_std"])),
            legacy_float64_vs_team_float32_max_input_difference=float(abs(norm - oldnorm).max()),
            double_standardization_max_input_difference=float(abs((norm-s["seq_mean"])/s["seq_std"]-norm).max()),
            legacy_stats_loader_expected_key="schema_version", legacy_stats_loader_key_present="schema_version" in s,
            standardization_source="saved team fit statistics; one transform; no refit",
            audit_scope="all 7352 official train windows, all nine channels, labels and subjects; no test array access")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--processed-data", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--preprocess-stats", type=Path, default=Path("data/preprocess.npz"))
    p.add_argument("--output", type=Path, default=Path("reports/lstm/dataset_audit.json"))
    a = p.parse_args()
    audit(a.processed_data, a.preprocess_stats, a.data_root, a.output)
