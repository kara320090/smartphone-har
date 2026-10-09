"""Independent fresh-process verification of saved preprocessing and model.

This supplies training handoff evidence. Uses validation only, never test metrics.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from tensorflow import keras
from .adapters.preprocessing_reference import Preprocessor
from .adapters.uci_reference import CHANNELS, CLASSES, load_dataset
from .runtime import configure_runtime, file_sha256, write_json
from .train import predict_batches
from .training_metrics import classification_metrics, softmax


def verify_bundle(run_directory, samples, indices):
    out = Path(run_directory)
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    if json.loads((out / "status.json").read_text())["status"] != "complete":
        raise ValueError("Cannot verify an incomplete run")
    if metadata["channels"] != CHANNELS or metadata["classes"] != CLASSES:
        raise ValueError("Channel/class order differs")
    if file_sha256(out / "model.keras") != metadata["model_sha256"]:
        raise ValueError("Model checksum differs")
    if file_sha256(out / "preprocess.npz") != metadata["preprocess_sha256"]:
        raise ValueError("Preprocessing checksum differs")
    if metadata.get("preprocessing") == "team_sequence_npz_v1":
        from .adapters.processed_npz import TeamSequencePreprocessor
        pre = TeamSequencePreprocessor.load(out / "preprocess.npz")
        for field in ("data_sha256", "source_stats_sha256"):
            if metadata["data_provenance"][field] != samples["provenance"][field]:
                raise ValueError(f"Source {field} differs")
    elif metadata.get("preprocessing") == "reference_npz_v1":
        pre = Preprocessor.load(out / "preprocess.npz")
    else:
        raise ValueError("Unknown saved preprocessing format")
    model = keras.models.load_model(out / "model.keras", compile=False)
    split = json.loads((out / "split.json").read_text(encoding="utf-8"))
    by_id = {sample_id: i for i, sample_id in enumerate(samples["sample_id"])}
    rows = np.array([by_id[sample_id] for sample_id in split["validation_ids"]])
    if not set(rows.tolist()).issubset(set(indices["validation"].tolist())):
        raise ValueError("Saved validation IDs do not match agreed subject split")
    raw = samples["X_features" if pre.input_kind == "features" else "X_seq"][rows]
    x = pre.transform(raw)
    if list(x.shape[1:]) != metadata["model_input_shape"]:
        raise ValueError("Preprocessed shape differs from model metadata")
    logits = predict_batches(model, x)
    current = softmax(logits)
    with (out / "validation_predictions.csv").open(encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    if [r["sample_id"] for r in saved] != split["validation_ids"]:
        raise ValueError("Saved predictions are not in the recorded sample order")
    np.testing.assert_array_equal([int(r["true_label"]) for r in saved], samples["y"][rows])
    np.testing.assert_array_equal([int(r["subject"]) for r in saved], samples["subject"][rows])
    previous = np.array([[float(r[f"probability_{i}"]) for i in range(6)] for r in saved])
    np.testing.assert_allclose(current, previous, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(current.argmax(1), previous.argmax(1))
    np.testing.assert_allclose(current.sum(axis=1), 1, rtol=0, atol=1e-6)
    metrics = classification_metrics(samples["y"][rows], logits)
    previous_metrics = json.loads((out / "metrics.json").read_text())
    for field in ("accuracy", "macro_f1"):
        np.testing.assert_allclose(metrics[field], previous_metrics[field], rtol=0, atol=1e-12)
    return {"run_id": out.name, "passed": True, "count": len(rows),
            "max_probability_difference": float(abs(current - previous).max()),
            "labels_identical": True, "metrics_recomputed": metrics,
            "test_evaluated": False}


def processed_verification_samples(cache, stats, smoke=False):
    from .adapters.processed_npz import load_processed_data
    data = load_processed_data(cache, stats, smoke=smoke)
    samples = {"X_seq": data.raw_validation, "y": data.y_validation,
               "sample_id": data.validation_ids, "subject": data.validation_subjects,
               "provenance": data.provenance}
    return samples, {"validation": np.arange(len(data.y_validation))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-root", type=Path)
    group.add_argument("--processed-data", type=Path)
    parser.add_argument("--preprocess-stats", type=Path)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/formal"))
    parser.add_argument("--report", type=Path, default=Path("reports/fresh_process_verification.json"))
    args = parser.parse_args()
    configure_runtime()
    if args.processed_data:
        if not args.preprocess_stats:
            parser.error("--preprocess-stats is required")
        samples, indices = processed_verification_samples(args.processed_data, args.preprocess_stats, args.smoke)
    else:
        samples, indices = load_dataset(args.data_root)
    paths = [args.run_directory] if args.run_directory else sorted(args.runs_dir.iterdir())
    results = [verify_bundle(path, samples, indices)
               for path in paths if (path / "metadata.json").exists()]
    if not results:
        raise ValueError("No completed runs found")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, {"passed": True, "runs": results})
    print(f"Verified {len(results)} runs from original validation inputs in a fresh process")
