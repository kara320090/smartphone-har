"""Predict raw float32 sensor windows with a verified saved sequence MLP."""
import argparse
import json
from pathlib import Path
import numpy as np
from tensorflow import keras
from .adapters.processed_npz import TeamSequencePreprocessor
from .adapters.uci_reference import CHANNELS, CLASSES
from .runtime import configure_runtime, file_sha256, write_json
from .train import predict_batches
from .training_metrics import softmax


def load_bundle(run_directory):
    run = Path(run_directory)
    if json.loads((run / "status.json").read_text())["status"] != "complete":
        raise ValueError("Cannot predict with an incomplete run")
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if (config.get("model_key") != "mlp" or metadata.get("preprocessing") != "team_sequence_npz_v1"
            or metadata["model_input_shape"] != [128, 9]
            or metadata["channels"] != CHANNELS or metadata["classes"] != CLASSES):
        raise ValueError("Expected a six-class team sequence MLP bundle")
    for name, field in (("model.keras", "model_sha256"), ("preprocess.npz", "preprocess_sha256")):
        if file_sha256(run / name) != metadata[field]:
            raise ValueError(f"Bundle checksum differs: {name}")
    pre = TeamSequencePreprocessor.load(run / "preprocess.npz")
    model = keras.models.load_model(run / "model.keras", compile=False)
    if model.input_shape[1:] != (128, 9) or model.output_shape[1:] != (6,):
        raise ValueError("Model shape disagrees with the bundle contract")
    return model, pre


def predict_raw_bundle(run_directory, raw):
    # Reject malformed inputs before loading the model or running TensorFlow.
    raw = np.asarray(raw)
    if raw.dtype != np.float32 or raw.ndim != 3 or raw.shape[1:] != (128, 9) or len(raw) == 0:
        raise ValueError("Expected nonempty float32 (N,128,9) unstandardized input")
    if not np.isfinite(raw).all():
        raise ValueError("Input contains NaN or Inf")
    model, pre = load_bundle(run_directory)
    probabilities = softmax(predict_batches(model, pre.transform(raw)))
    return probabilities, probabilities.argmax(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--processed-data", type=Path)
    inputs.add_argument("--input-npy", type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    configure_runtime()
    if args.processed_data:
        metadata = json.loads((args.run_directory / "metadata.json").read_text(encoding="utf-8"))
        if file_sha256(args.processed_data) != metadata["data_provenance"]["data_sha256"]:
            raise ValueError("Prediction cache checksum differs from the training source")
        with np.load(args.processed_data, allow_pickle=False) as cache:
            raw = cache["val_Xs"][:args.count]
            ids = cache["val_sample_id"][:args.count].tolist()
    else:
        raw = np.load(args.input_npy, allow_pickle=False)[:args.count]
        ids = list(range(len(raw)))
    probabilities, predictions = predict_raw_bundle(args.run_directory, raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"run_id": args.run_directory.name, "input_state": "unstandardized",
               "test_evaluated": False, "predictions": [
                   {"sample_id": sid, "predicted_label": int(label), "class": CLASSES[int(label)],
                    "probabilities": probability.tolist()}
                   for sid, label, probability in zip(ids, predictions, probabilities)]})
    print(args.output)


if __name__ == "__main__":
    main()
