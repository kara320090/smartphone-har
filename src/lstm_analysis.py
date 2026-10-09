"""Recompute development metrics and benchmark verified LSTM bundles."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras
from .adapters.processed_npz import load_processed_data, TeamSequencePreprocessor, sha256
from .runtime import configure_runtime, environment_info, write_json


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predict_raw_bundle(run, raw):
    run = Path(run)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf8"))
    if metadata.get("preprocessing") != "team_sequence_npz_v1":
        raise ValueError("Expected a team sequence bundle")
    for name, key in (("model.keras", "model_sha256"), ("preprocess.npz", "preprocess_sha256")):
        if sha256(run / name) != metadata[key]:
            raise ValueError(f"Bundle checksum differs: {name}")
    pre = TeamSequencePreprocessor.load(run / "preprocess.npz")
    x = pre.transform(raw)
    model = keras.models.load_model(run / "model.keras", compile=False)
    logits = model(x, training=False).numpy()
    probability = tf.nn.softmax(logits).numpy()
    return probability, probability.argmax(1)


def benchmark(run, prepared, raw, warmup=50, repetitions=500):
    keras.backend.clear_session()
    configure_runtime(2026, 4)
    model = keras.models.load_model(run / "model.keras", compile=False)
    pre = TeamSequencePreprocessor.load(run / "preprocess.npz")

    @tf.function(input_signature=[tf.TensorSpec((1, 128, 9), tf.float32)])
    def infer(tensor):
        return model(tensor, training=False)

    tensor = tf.convert_to_tensor(prepared[:1])
    raw_one = raw[:1]
    for _ in range(warmup):
        infer(tensor).numpy()
        pre.transform(raw_one)
    rows = []
    for i in range(repetitions):
        start = time.perf_counter_ns()
        infer(tensor).numpy()  # Materialize output before ending timing.
        rows.append({"run_id": run.name, "scope": "compiled_model_output_numpy", "iteration": i,
                     "milliseconds": (time.perf_counter_ns() - start) / 1e6})
        start = time.perf_counter_ns()
        pre.transform(raw_one)
        rows.append({"run_id": run.name, "scope": "preprocess_validation_and_transform", "iteration": i,
                     "milliseconds": (time.perf_counter_ns() - start) / 1e6})
    summary = []
    for scope in ("compiled_model_output_numpy", "preprocess_validation_and_transform"):
        values = [r["milliseconds"] for r in rows if r["scope"] == scope]
        summary.append({"run_id": run.name, "scope": scope, "batch": 1, "warmup": warmup,
                        "repetitions": repetitions, "median_ms": float(np.median(values)),
                        "p95_ms": float(np.percentile(values, 95)),
                        "min_ms": float(min(values)), "max_ms": float(max(values))})
    return rows, summary


def analyze(runs_dir, output_dir, data, do_benchmark=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    results, class_rows, subject_rows, errors, latency, latency_summary = [], [], [], [], [], []
    paths = [runs_dir / f"{experiment}_seed{seed}" for experiment in ("L01", "L02", "L03") for seed in (2026, 2027, 2028)]
    for run in paths:
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf8"))
        config = json.loads((run / "config.json").read_text(encoding="utf8"))
        verification = json.loads((run / "fresh_process_verification.json").read_text(encoding="utf8"))
        assert verification["passed"] and verification["runs"][0]["count"] == len(data.y_validation)
        assert not metrics["smoke"] and not metrics["test_evaluated"]
        with (run / "validation_predictions.csv").open(encoding="utf8") as f:
            pred_rows = list(csv.DictReader(f))
        np.testing.assert_array_equal([r["sample_id"] for r in pred_rows], data.validation_ids)
        y = np.array([int(r["true_label"]) for r in pred_rows])
        pred = np.array([int(r["predicted_label"]) for r in pred_rows])
        np.testing.assert_array_equal(y, data.y_validation)
        f1 = float(f1_score(y, pred, labels=list(range(6)), average="macro", zero_division=0))
        accuracy = float(accuracy_score(y, pred))
        assert abs(f1 - metrics["macro_f1"]) < 1e-12 and abs(accuracy - metrics["accuracy"]) < 1e-12
        results.append({"run_id": run.name, "experiment": config["experiment_id"], "seed": config["seed"],
                        "accuracy": accuracy, "macro_f1": f1, "parameters": metrics["parameters"],
                        "best_epoch": metrics["best_epoch"], "epochs_ran": metrics["epochs_ran"],
                        "train_seconds": metrics["train_seconds"], "validation_count": len(y)})
        precision, recall, scores, support = precision_recall_fscore_support(y, pred, labels=list(range(6)), zero_division=0)
        for c in range(6):
            class_rows.append(dict(run_id=run.name, label=c, precision=precision[c], recall=recall[c], f1=scores[c], support=int(support[c])))
        for subject in np.unique(data.validation_subjects):
            mask = data.validation_subjects == subject
            subject_rows.append(dict(run_id=run.name, subject=int(subject), support=int(mask.sum()),
                accuracy=float(accuracy_score(y[mask], pred[mask])),
                macro_f1=float(f1_score(y[mask], pred[mask], labels=list(range(6)), average="macro", zero_division=0))))
        for i in np.flatnonzero(y != pred):
            errors.append(dict(run_id=run.name, sample_id=str(data.validation_ids[i]), subject=int(data.validation_subjects[i]),
                               true_label=int(y[i]), predicted_label=int(pred[i]),
                               predicted_probability=float(pred_rows[i][f"probability_{pred[i]}"])))
        write_json(output_dir / f"{run.name}_confusion.json", {"rows": "true", "columns": "predicted", "matrix": confusion_matrix(y,pred,labels=list(range(6))).tolist()})
        if do_benchmark:
            raw_rows, summaries = benchmark(run, data.x_validation, data.raw_validation)
            latency.extend(raw_rows)
            latency_summary.extend(summaries)
        print("ANALYZED", run.name, f"F1={f1:.6f}", flush=True)
    summary, paired = [], []
    for experiment in ("L01", "L02", "L03"):
        rows = [r for r in results if r["experiment"] == experiment]
        summary.append({"experiment": experiment, "repetitions": 3,
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in rows])),
            "macro_f1_sample_sd": float(np.std([r["macro_f1"] for r in rows], ddof=1)),
            "accuracy_mean": float(np.mean([r["accuracy"] for r in rows])),
            "accuracy_sample_sd": float(np.std([r["accuracy"] for r in rows], ddof=1)),
            "parameters": rows[0]["parameters"], "train_seconds_mean": float(np.mean([r["train_seconds"] for r in rows]))})
    for experiment in ("L02", "L03"):
        for seed in (2026, 2027, 2028):
            baseline = next(r for r in results if r["experiment"] == "L01" and r["seed"] == seed)
            candidate = next(r for r in results if r["experiment"] == experiment and r["seed"] == seed)
            paired.append(dict(experiment=experiment, seed=seed, macro_f1_delta=candidate["macro_f1"]-baseline["macro_f1"],
                               accuracy_delta=candidate["accuracy"]-baseline["accuracy"]))
    for name, rows in (("runs.csv",results),("summary.csv",summary),("paired_differences.csv",paired),
                       ("class_metrics.csv",class_rows),("subject_metrics.csv",subject_rows),("errors.csv",errors)):
        if rows:
            write_csv(output_dir / name, rows)
    if latency:
        write_csv(output_dir / "latency_samples.csv", latency)
        write_csv(output_dir / "latency_summary.csv", latency_summary)
    write_json(output_dir / "analysis.json", {"runs": results, "summary": summary, "paired": paired,
        "test_evaluated": False, "data_provenance": data.provenance,
        "latency": latency_summary, "benchmark_environment": environment_info(),
        "benchmark_scope": "tf.function compiled model, prepared Tensor, .numpy output; preprocessing timed separately",
        "statistics_scope": "sample SD across three training seeds on one fixed subject split"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--processed-data", type=Path, required=True)
    p.add_argument("--preprocess-stats", type=Path, default=Path("data/preprocess.npz"))
    p.add_argument("--runs-dir", type=Path, default=Path("runs/lstm"))
    p.add_argument("--output-dir", type=Path, default=Path("reports/lstm"))
    p.add_argument("--skip-benchmark", action="store_true")
    args = p.parse_args()
    configure_runtime(2026, 4)
    data = load_processed_data(args.processed_data, args.preprocess_stats)
    analyze(args.runs_dir, args.output_dir, data, not args.skip_benchmark)
