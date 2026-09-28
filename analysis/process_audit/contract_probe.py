"""Run the same controlled adapter probes before and after the contract repair."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.contracts import prepare_data


def sample_fixture():
    rng = np.random.default_rng(1107)
    return {"X_features": rng.normal(size=(18, 561)).astype(np.float32),
            "y": np.tile(np.arange(6), 3), "subject": np.repeat([3, 1, 5], 6),
            "sample_id": np.array([f"fixture:{i}" for i in range(18)])}


def probe():
    config = {"input_kind": "features", "standardize": True}
    baseline = {"fit": np.arange(6), "validation": np.arange(6, 12)}
    cases = {}
    cases["valid_control"] = lambda: prepare_data(sample_fixture(), baseline, config)
    for key, values in {
        "fractional_fit_indices": np.arange(6) + 0.8,
        "negative_fit_indices": np.arange(-6, 0),
        "boolean_fit_indices": np.array([True, False]),
        "duplicate_fit_indices": np.array([0, 0, 1, 2, 3, 4]),
        "out_of_range_fit_indices": np.array([18]),
        "empty_fit_indices": np.array([], dtype=int),
    }.items():
        cases[key] = lambda values=values: prepare_data(sample_fixture(), {**baseline, "fit": values}, config)
    def raw_mismatch():
        data = prepare_data(sample_fixture(), baseline, config)
        data.raw_validation = data.raw_validation[::-1].copy()
        data.validate()
        return data
    cases["reordered_raw_validation"] = raw_mismatch
    def noninteger_subjects():
        data = prepare_data(sample_fixture(), baseline, config)
        data.fit_subjects = data.fit_subjects.astype(float) + .2
        data.validate()
        return data
    cases["fractional_subject_ids"] = noninteger_subjects
    records = []
    for name, fn in cases.items():
        try:
            data = fn()
            records.append({"case": name, "accepted": True,
                            "selected_fit_ids": data.fit_ids.tolist(),
                            "raw_to_prepared_max_difference": float(np.max(np.abs(
                                data.preprocessor.transform(data.raw_validation) - data.x_validation)))})
        except Exception as error:
            records.append({"case": name, "accepted": False,
                            "exception": type(error).__name__, "message": str(error)})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    result = {"phase": args.phase, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "fixture": "Synthetic 18-row adapter fixture, not HAR performance data",
              "cases": probe()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
