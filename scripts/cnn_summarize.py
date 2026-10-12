"""Summarize verified C01 validation runs; never access official test data.

Uses metrics, history and validation prediction CSVs from src.train.
Run after scripts/run_cnn_suite.py with the same --runs-dir.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _save_csv(path, rows):
    if not rows:
        raise ValueError("No rows for " + str(path))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for run, history in runs:
        epochs = np.arange(1, len(history) + 1)
        ax.plot(epochs, [float(h["val_loss"]) for h in history], label=f"{run} val")
        ax.plot(epochs, [float(h["loss"]) for h in history], ls="--", alpha=.6, label=f"{run} train")
    ax.set(title="CNN C01 learning curves (validation S0)", xlabel="Epoch", ylabel="Cross-entropy loss")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_confusion(cm, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
    normalized = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center",
                    color="white" if normalized[i, j] > .55 else "black", fontsize=9)
    ax.set(xticks=np.arange(6), yticks=np.arange(6), xticklabels=labels,
           yticklabels=labels, xlabel="Predicted activity", ylabel="True activity",
           title="CNN C01 validation confusion (pooled over seeds; row normalized)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.colorbar(img, ax=ax, label="Proportion of true activity")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def summarize(runs_dir, output_dir, seeds, provisional=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    records, person_records, all_cms, histories = [], [], [], []
    aligned_ids = None
    for seed in seeds:
        run = runs_dir / (f"C01_seed{seed}_torch" if provisional else f"C01_seed{seed}")
        if not run.is_dir():
            raise FileNotFoundError(f"Formal run not found: {run}; do not mix smoke with formal runs")
        config = json.loads((run / "config.json").read_text(encoding="utf-8")) if not provisional else {"model_key":"cnn","seed":seed}
        status = json.loads((run / "status.json").read_text(encoding="utf-8")) if not provisional else {"status":"complete"}
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        check = json.loads((run / ("reload_check.json" if provisional else "fresh_process_verification.json")).read_text(encoding="utf-8"))
        if (status.get("status") != "complete" or config.get("model_key") != "cnn" or
                config.get("seed") != seed or metrics.get("smoke") or metrics.get("test_evaluated") or
                not check.get("passed")):
            raise ValueError(f"Invalid/unfinished/test-touched run: {run}")
        rows = _read_csv(run / "validation_predictions.csv")
        ids = [r["sample_id"] for r in rows]
        if aligned_ids is not None and ids != aligned_ids:
            raise ValueError("Seeds use different validation samples or row orders")
        aligned_ids = ids
        y = np.array([int(r["true_label"]) for r in rows])
        pred = np.array([int(r["predicted_label"]) for r in rows])
        person = np.array([int(r["subject"]) for r in rows])
        if len(y) != 1775 or not set(np.unique(y)) == set(range(6)) or set(np.unique(person)) != {1, 6, 14, 21, 23}:
            raise ValueError("Validation contract broken")
        probs = np.array([[float(r[f"probability_{i}"]) for i in range(6)] for r in rows])
        if not np.isfinite(probs).all() or not np.allclose(probs.sum(1), 1, rtol=0, atol=1e-5):
            raise ValueError("Invalid saved class probabilities")
        np.testing.assert_array_equal(pred, probs.argmax(1))
        accuracy = accuracy_score(y, pred)
        macro_f1 = f1_score(y, pred, average="macro", labels=np.arange(6), zero_division=0)
        np.testing.assert_allclose([accuracy, macro_f1], [metrics["accuracy"], metrics["macro_f1"]], atol=1e-12)
        cm = confusion_matrix(y, pred, labels=range(6))
        all_cms.append(cm)
        precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=range(6), zero_division=0)
        rec = {"run_id": run.name, "seed": seed, "validation_count": len(y),
               "accuracy": accuracy, "macro_f1": macro_f1, "epochs_ran": metrics["epochs_ran"],
               "best_epoch": metrics["best_epoch"], "parameters": metrics["parameters"],
               "train_seconds": metrics["train_seconds"], "fresh_process_verified": not provisional, "model_reload_verified": bool(check["passed"]),
               "backend": "keras3-torch" if provisional else "tensorflow"}
        records.append(rec)
        for p in sorted(set(person)):
            mask = person == p
            person_records.append({"run_id": run.name, "seed": seed, "subject": int(p),
                                   "support": int(mask.sum()), "accuracy": accuracy_score(y[mask], pred[mask]),
                                   "macro_f1": f1_score(y[mask], pred[mask], average="macro",
                                                       labels=range(6), zero_division=0)})
        with (output_dir / f"{run.name}_per_class.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["label", "support", "precision", "recall", "f1"])
            for cls in range(6):
                writer.writerow([cls, int(support[cls]), float(precision[cls]), float(recall[cls]), float(f1[cls])])
        histories.append((run.name, _read_csv(run / "history.csv")))
    acc = [r["accuracy"] for r in records]
    f1 = [r["macro_f1"] for r in records]
    summary = {"model": "cnn", "experiment_id": "C01", "split": "validation_s0",
               "test_evaluated": False, "official_tensorflow_run": not provisional,
               "backend": "keras3-torch" if provisional else "tensorflow", "seed_count": len(seeds), "seeds": seeds,
               "validation_count_per_seed": 1775,
               "accuracy_mean": statistics.mean(acc), "accuracy_sample_sd": statistics.stdev(acc) if len(acc)>1 else None,
               "macro_f1_mean": statistics.mean(f1), "macro_f1_sample_sd": statistics.stdev(f1) if len(f1)>1 else None,
               "parameters": records[0]["parameters"], "runs": records,
               "confusion_matrix_pooled_seeds": np.sum(all_cms, axis=0).tolist(),
               "interpretation": "Sample SD describes fixed-split training-seed variability, not confidence about unseen people."}
    _save_csv(output_dir / "cnn_runs.csv", records)
    _save_csv(output_dir / "cnn_per_subject.csv", person_records)
    (output_dir / "cnn_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot_curves(histories, output_dir / "cnn_learning_curves.png")
    _plot_confusion(np.sum(all_cms, axis=0), output_dir / "cnn_confusion.png")
    print(json.dumps({k: v for k,v in summary.items() if k in ("macro_f1_mean", "macro_f1_sample_sd", "accuracy_mean", "accuracy_sample_sd")}, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/cnn"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/cnn"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--provisional-torch", action="store_true", help="Analyze separately labeled Keras/Torch validation runs")
    args = parser.parse_args()
    summarize(args.runs_dir, args.output_dir, args.seeds, args.provisional_torch)
