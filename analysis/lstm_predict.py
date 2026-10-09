"""Predict saved LSTM bundles from unstandardized (N,128,9) float32 windows."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.adapters.processed_npz import sha256
from src.lstm_analysis import predict_raw_bundle
from src.runtime import configure_runtime


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run-directory", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-npy", type=Path, help="Unstandardized windows, correct channel order")
    source.add_argument("--processed-data", type=Path, help="Demonstration using val_Xs only")
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.count < 1:
        p.error("count must be positive")
    if a.output.exists():
        p.error("output exists; use a new output filename")
    configure_runtime(2026, 4)
    if a.processed_data:
        metadata = json.loads((a.run_directory / "metadata.json").read_text(encoding="utf8"))
        if sha256(a.processed_data) != metadata["data_provenance"]["data_sha256"]:
            raise ValueError("Demonstration cache checksum differs from the trained source")
        with np.load(a.processed_data, allow_pickle=False) as z:
            raw = z["val_Xs"][:a.count]
            ids = z["val_sample_id"][:a.count].tolist()
    else:
        raw = np.load(a.input_npy, allow_pickle=False)[:a.count]
        ids = [str(i) for i in range(len(raw))]
    if not len(raw):
        raise ValueError("At least one input window is required")
    probabilities, predicted = predict_raw_bundle(a.run_directory, raw)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({
        "run_id": a.run_directory.name, "input_state": "unstandardized",
        "standardization_applications": 1, "count": len(raw),
        "class_order": ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"],
        "predictions": [dict(sample_id=sid, predicted_label=int(pred), probabilities=prob.tolist())
                        for sid, pred, prob in zip(ids, predicted, probabilities)]
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(f"Predicted {len(raw)} windows; output: {a.output}")


if __name__ == "__main__":
    main()
