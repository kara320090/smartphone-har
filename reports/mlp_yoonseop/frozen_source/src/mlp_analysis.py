"""Audit and summarize the three frozen M01 validation runs; never evaluate test."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "work/mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from .adapters.processed_npz import load_processed_data
from .adapters.uci_reference import CHANNELS, CLASSES
from .mlp_inference import load_bundle, predict_raw_bundle
from .runtime import configure_runtime, environment_info, write_json
from .train import predict_batches
from .training_metrics import classification_metrics, softmax


SEEDS = (2026, 2027, 2028)


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def grouped_summary(rows, group_key, measures):
    results = []
    for group in sorted(set(row[group_key] for row in rows)):
        selected = [row for row in rows if row[group_key] == group]
        if len(selected) != 3:
            raise ValueError("Each summary must contain all three seeds")
        result = {group_key: group, "support_per_seed": selected[0]["support"], "seeds": 3}
        for field in measures:
            values = np.array([row[field] for row in selected], dtype=np.float64)
            result[field + "_mean"] = float(values.mean())
            result[field + "_sample_sd"] = float(values.std(ddof=1))
        results.append(result)
    return results


def cross_entropy(y, logits):
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y]))


def plot_artifacts(out, histories, matrices, class_summary, subject_summary, structure):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                        "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for column, (seed, history, best_epoch) in enumerate(histories):
        epochs = np.arange(1, len(history) + 1)
        for row, fields, ylabel in ((0, ("loss", "val_loss"), "Cross-entropy loss"),
                                    (1, ("accuracy", "val_accuracy"), "Accuracy")):
            ax = axes[row, column]
            for field, label, color in zip(fields, ("Training", "Validation"), ("#2563eb", "#dc6b23")):
                ax.plot(epochs, [float(h[field]) for h in history], label=label, color=color, linewidth=1.8)
            ax.axvline(best_epoch, color="#64748b", linestyle="--", linewidth=1,
                       label=f"Selected epoch {best_epoch}")
            ax.set(xlabel="Epoch (1-based)", ylabel=ylabel, title=f"M01 / seed {seed}")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("Sequence MLP: fixed subject split, three training seeds", fontsize=14)
    fig.savefig(out / "learning_curves.png", dpi=200)
    fig.savefig(out / "learning_curves.svg")
    plt.close(fig)

    short = ("Walk", "Upstairs", "Downstairs", "Sit", "Stand", "Lay")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    vmax = max(int(matrix.max()) for matrix in matrices)
    for seed, matrix, ax in zip(SEEDS, matrices, axes):
        img = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if matrix[i, j] > vmax * .55 else "#162138")
        ax.set(xticks=range(6), yticks=range(6), xticklabels=short, yticklabels=short,
               xlabel="Predicted label", ylabel="True label", title=f"M01 / seed {seed}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(img, ax=axes, shrink=.7, label="Windows per seed")
    fig.savefig(out / "confusion_matrices.png", dpi=200)
    fig.savefig(out / "confusion_matrices.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, rows, labels, title in ((axes[0], class_summary, short, "Validation F1 by class"),
                                  (axes[1], subject_summary,
                                   [str(r["subject"]) for r in subject_summary], "Validation Macro F1 by subject")):
        field = "f1" if "label" in rows[0] else "macro_f1"
        ax.bar(range(len(rows)), [r[field + "_mean"] for r in rows], color="#2563eb",
               yerr=[r[field + "_sample_sd"] for r in rows], capsize=4, alpha=.85)
        ax.set(xticks=range(len(rows)), xticklabels=labels, ylim=(0, 1.05), title=title, ylabel="Score")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Mean and sample SD across 3 seeds on the same validation windows", fontsize=11)
    fig.savefig(out / "class_subject_f1.png", dpi=200)
    fig.savefig(out / "class_subject_f1.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set(xlim=(0, 12), ylim=(0, 2.8))
    ax.axis("off")
    labels = ["Sensor window\n128 x 9", "Flatten\n1,152", "Dense + ReLU\n128", "Dense + ReLU\n64", "Linear logits\n6"]
    params = [0] + [r["parameters"] for r in structure]
    for i, label in enumerate(labels):
        x = .2 + i * 2.4
        box = FancyBboxPatch((x, .75), 2, 1.2, boxstyle="round,pad=0.08",
                             facecolor="#eff6ff" if i < 4 else "#dbeafe", edgecolor="#2563eb")
        ax.add_patch(box)
        ax.text(x + 1, 1.4, label, ha="center", va="center", fontsize=11)
        ax.text(x + 1, .37, f"{params[i]:,} parameters", ha="center", fontsize=9, color="#475569")
        if i < 4:
            ax.annotate("", xy=(x + 2.3, 1.35), xytext=(x + 2.07, 1.35),
                        arrowprops={"arrowstyle": "->", "color": "#475569"})
    ax.text(6, 2.48, "M01 sequence MLP | 156,230 trainable parameters | Dropout 0", ha="center", fontsize=13)
    fig.savefig(out / "architecture.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "architecture.svg", bbox_inches="tight")
    plt.close(fig)


def analyze(runs_dir, out, data):
    out.mkdir(parents=True, exist_ok=True)
    runs, class_rows, subject_rows, errors, histories, matrices, checks, error_votes = [], [], [], [], [], [], [], {}
    structure = None
    for seed in SEEDS:
        run = runs_dir / f"M01_seed{seed}"
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "metrics.json").read_text())
        verified = json.loads((run / "fresh_process_verification.json").read_text())
        if (config["seed"] != seed or config["experiment_id"] != "M01" or metrics["smoke"]
                or metrics["test_evaluated"] or not verified["passed"]
                or verified["runs"][0]["count"] != len(data.y_validation)):
            raise ValueError(f"Invalid formal M01 run: {run}")
        for field in ("data_sha256", "source_stats_sha256", "validation_row_binding_sha256"):
            if metadata["data_provenance"][field] != data.provenance[field]:
                raise ValueError(f"Data provenance differs: {field}")
        predictions = read_csv(run / "validation_predictions.csv")
        np.testing.assert_array_equal([r["sample_id"] for r in predictions], data.validation_ids)
        np.testing.assert_array_equal([int(r["subject"]) for r in predictions], data.validation_subjects)
        y = np.array([int(r["true_label"]) for r in predictions])
        pred = np.array([int(r["predicted_label"]) for r in predictions])
        logits = np.array([[float(r[f"logit_{c}"]) for c in range(6)] for r in predictions])
        probability = np.array([[float(r[f"probability_{c}"]) for c in range(6)] for r in predictions])
        np.testing.assert_array_equal(y, data.y_validation)
        np.testing.assert_array_equal(pred, probability.argmax(axis=1))
        np.testing.assert_allclose(probability, softmax(logits), rtol=0, atol=1e-12)
        recomputed = classification_metrics(y, logits)
        for field in ("accuracy", "macro_f1", "recall_per_class", "confusion_matrix"):
            np.testing.assert_allclose(recomputed[field], metrics[field], rtol=0, atol=1e-12)
        model, pre = load_bundle(run)
        fresh_logits = predict_batches(model, pre.transform(data.raw_validation))
        np.testing.assert_allclose(softmax(fresh_logits), probability, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(predict_batches(model, data.x_validation), fresh_logits)
        fit_logits = predict_batches(model, data.x_fit)
        fit_metrics = classification_metrics(data.y_fit, fit_logits)
        fit_metrics["loss"] = cross_entropy(data.y_fit, fit_logits)
        write_json(out / f"{run.name}_restored_fit_metrics.json", fit_metrics)
        history = read_csv(run / "history.csv")
        best = metrics["best_epoch"]
        if best != int(np.argmin([float(h["val_loss"]) for h in history])) + 1:
            raise ValueError("Reported best epoch disagrees with the training history")
        runs.append({"run_id": run.name, "contributor": "한윤섭", "config_id": "M01", "seed": seed,
                     "split_id": data.split_id, "validation_count": len(y), "accuracy": metrics["accuracy"],
                     "macro_f1": metrics["macro_f1"], "parameters": metrics["parameters"],
                     "best_epoch": best, "epochs_ran": metrics["epochs_ran"], "best_val_loss": metrics["best_val_loss"],
                     "train_seconds": metrics["train_seconds"], "restored_fit_accuracy": fit_metrics["accuracy"],
                     "restored_fit_macro_f1": fit_metrics["macro_f1"], "restored_fit_loss": fit_metrics["loss"],
                     "restored_accuracy_gap": fit_metrics["accuracy"] - metrics["accuracy"],
                     "git_base_commit": metadata["environment"]["git_commit"],
                     "code_sha256": metadata["environment"]["code_sha256"],
                     "data_sha256": data.provenance["data_sha256"],
                     "stats_sha256": data.provenance["source_stats_sha256"]})
        precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=range(6), zero_division=0)
        for c in range(6):
            class_rows.append({"run_id": run.name, "seed": seed, "label": c, "class": CLASSES[c],
                               "precision": float(precision[c]), "recall": float(recall[c]),
                               "f1": float(f1[c]), "support": int(support[c])})
        for subject in np.unique(data.validation_subjects):
            mask = data.validation_subjects == subject
            submetrics = classification_metrics(y[mask], logits[mask])
            subject_rows.append({"run_id": run.name, "seed": seed, "subject": int(subject),
                                 "support": int(mask.sum()), "accuracy": submetrics["accuracy"],
                                 "macro_f1": submetrics["macro_f1"]})
        for i in np.flatnonzero(y != pred):
            row = {"run_id": run.name, "seed": seed, "sample_id": str(data.validation_ids[i]),
                   "subject": int(data.validation_subjects[i]), "true_label": int(y[i]),
                   "predicted_label": int(pred[i]), "predicted_probability": float(probability[i, pred[i]])}
            errors.append(row)
            votes = error_votes.setdefault(row["sample_id"], {"sample_id": row["sample_id"], "subject": row["subject"],
                      "true_label": row["true_label"], "wrong_seed_count": 0, "wrong_seed_predictions": []})
            votes["wrong_seed_count"] += 1
            votes["wrong_seed_predictions"].append(f"{seed}:{pred[i]}")
        matrix = confusion_matrix(y, pred, labels=range(6))
        matrices.append(matrix)
        write_json(out / f"{run.name}_confusion.json", {"rows": "true_label", "columns": "predicted_label",
                   "classes": CLASSES, "matrix": matrix.tolist(), "support": len(y)})
        histories.append((seed, history, best))
        if structure is None:
            structure = [{"layer": layer.name, "type": layer.__class__.__name__,
                          "output_shape": str(tuple(layer.output.shape)), "parameters": layer.count_params()}
                         for layer in model.layers]
        checks.append({"run_id": run.name, "full_validation_recomputed": True,
                       "evaluation_mode_deterministic": True, "count": len(y),
                       "max_probability_difference": float(np.max(np.abs(softmax(fresh_logits) - probability)))})
        print(f"ANALYZED {run.name} accuracy={metrics['accuracy']:.6f} macro_f1={metrics['macro_f1']:.6f}", flush=True)
    summary = {"model": "Sequence MLP", "config_id": "M01", "contributor": "한윤섭", "seeds": list(SEEDS),
               "split": "validation", "repetitions": 3, "parameters": runs[0]["parameters"],
               "accuracy_mean": float(np.mean([r["accuracy"] for r in runs])),
               "accuracy_sample_sd": float(np.std([r["accuracy"] for r in runs], ddof=1)),
               "macro_f1_mean": float(np.mean([r["macro_f1"] for r in runs])),
               "macro_f1_sample_sd": float(np.std([r["macro_f1"] for r in runs], ddof=1)),
               "test_evaluated": False, "scope": "three training seeds on one fixed subject split"}
    class_summary = grouped_summary(class_rows, "label", ("precision", "recall", "f1"))
    subject_summary = grouped_summary(subject_rows, "subject", ("accuracy", "macro_f1"))
    consensus = sorted(error_votes.values(), key=lambda r: (-r["wrong_seed_count"], r["sample_id"]))
    for row in consensus:
        row["wrong_seed_predictions"] = ";".join(row["wrong_seed_predictions"])
    for name, rows in (("runs.csv", runs), ("summary.csv", [summary]), ("class_metrics.csv", class_rows),
                       ("class_metrics_summary.csv", class_summary), ("subject_metrics.csv", subject_rows),
                       ("subject_metrics_summary.csv", subject_summary), ("errors.csv", errors),
                       ("error_consensus.csv", consensus), ("model_structure.csv", structure)):
        write_csv(out / name, rows)
    distribution = []
    for part, labels, subjects in (("fit", data.y_fit, data.fit_subjects),
                                   ("validation", data.y_validation, data.validation_subjects)):
        for c in range(6):
            distribution.append({"split": part, "label": c, "class": CLASSES[c], "count": int(np.sum(labels == c))})
    write_csv(out / "data_distribution.csv", distribution)
    stats = [{"channel": channel, "mean": float(data.preprocessor.mean[0, 0, c]),
              "std": float(data.preprocessor.scale[0, 0, c])} for c, channel in enumerate(CHANNELS)]
    write_csv(out / "sequence_statistics.csv", stats)
    audit = {"data_provenance": data.provenance, "fit_count": len(data.y_fit), "validation_count": len(data.y_validation),
             "fit_subjects": np.unique(data.fit_subjects).tolist(),
             "validation_subjects": np.unique(data.validation_subjects).tolist(), "input_shape": [128, 9],
             "channels": CHANNELS, "classes": CLASSES, "Xf_used": False, "PCA_used": False,
             "test_evaluated": False,
             "preliminary_inspection": "All NPZ array shapes/dtypes and test subject IDs were inspected before training; no test metrics or model selection used test data."}
    write_json(out / "dataset_audit.json", audit)
    write_json(out / "summary.json", summary)
    # Input checks target the public prediction boundary, including dtype and emptiness.
    invalid = {"wrong_time_axis": np.zeros((1, 127, 9), np.float32),
               "wrong_channels": np.zeros((1, 128, 8), np.float32),
               "flattened": np.zeros((1, 1152), np.float32), "empty": np.zeros((0, 128, 9), np.float32),
               "float64": np.zeros((1, 128, 9), np.float64),
               "NaN": np.full((1, 128, 9), np.nan, np.float32), "Inf": np.full((1, 128, 9), np.inf, np.float32)}
    rejected = {}
    for name, raw in invalid.items():
        try:
            predict_raw_bundle(runs_dir / "M01_seed2026", raw)
        except ValueError as error:
            rejected[name] = str(error)
        else:
            raise AssertionError(f"Malformed inference input was accepted: {name}")
    write_json(out / "verification.json", {"passed": True, "runs": checks, "invalid_inputs_rejected": rejected,
               "test_evaluated": False, "environment": environment_info()})
    plot_artifacts(out, histories, matrices, class_summary, subject_summary, structure)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-data", type=Path, default=Path("data/processed_data.npz"))
    parser.add_argument("--preprocess-stats", type=Path, default=Path("data/preprocess.npz"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/mlp_yoonseop"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/mlp_yoonseop"))
    args = parser.parse_args()
    configure_runtime()
    data = load_processed_data(args.processed_data, args.preprocess_stats)
    print(analyze(args.runs_dir, args.output_dir, data))


if __name__ == "__main__":
    main()
