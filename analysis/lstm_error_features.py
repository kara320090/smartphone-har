"""Describe validation errors using observable input summaries; no causal inference."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from src.adapters.processed_npz import load_processed_data


def analyze(cache, stats, predictions, output):
    data = load_processed_data(cache, stats)
    with predictions.open(encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    np.testing.assert_array_equal([r["sample_id"] for r in rows], data.validation_ids)
    np.testing.assert_array_equal([int(r["true_label"]) for r in rows], data.y_validation)
    errors = []
    for i, row in enumerate(rows):
        y, pred = int(row["true_label"]), int(row["predicted_label"])
        if y == pred:
            continue
        raw = data.raw_validation[i]
        result = dict(run_id=predictions.parent.name, sample_id=row["sample_id"],
                      subject=int(data.validation_subjects[i]), true_label=y, predicted_label=pred,
                      confidence=float(row[f"probability_{pred}"]))
        channels = [f"{base}_{axis}" for base in ("body_acc", "body_gyro", "total_acc") for axis in "xyz"]
        for channel, name in enumerate(channels):
            signal = raw[:, channel].astype(np.float64)
            result[f"{name}_mean"] = float(signal.mean())
            result[f"{name}_std"] = float(signal.std())
            result[f"{name}_rms"] = float(np.sqrt(np.mean(signal ** 2)))
        errors.append(result)
    errors.sort(key=lambda r: (-r["confidence"], r["sample_id"]))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "error_signal_features.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(errors[0]))
        writer.writeheader(); writer.writerows(errors)
    (output / "error_case_selection.json").write_text(json.dumps({
        "run_id": predictions.parent.name,
        "selection": "L01 seed 2026 fixed in advance; top ten erroneous predictions by confidence",
        "scope": "original UCI preprocessed window values, before our standardization; no test arrays",
        "features": "per-channel population standard deviation, mean and RMS over 128 time points",
        "interpretation": "descriptive observations; no claim about sensor placement or causal explanation",
        "total_errors": len(errors), "cases": errors[:10]
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--processed-data", required=True, type=Path)
    p.add_argument("--preprocess-stats", default=Path("data/preprocess.npz"), type=Path)
    p.add_argument("--predictions", default=Path("runs/lstm/L01_seed2026/validation_predictions.csv"), type=Path)
    p.add_argument("--output-dir", default=Path("reports/lstm"), type=Path)
    a = p.parse_args()
    analyze(a.processed_data, a.preprocess_stats, a.predictions, a.output_dir)
