"""Prepare reviewable MLP evidence, then build an immutable release from a commit.

Run --prepare before committing. Run --build after committing a clean checkout.
No GitHub writes or model training are performed by this tool.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from verify_mlp_release import verify_archive

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/mlp_yoonseop"
TAG = "yoonseop-mlp-v1"
SEEDS = (2026, 2027, 2028)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def measure(predictions):
    matrix = [[0] * 6 for _ in range(6)]
    for row in predictions:
        matrix[int(row["true_label"])][int(row["predicted_label"])] += 1
    f1 = []
    for c in range(6):
        true_positive = matrix[c][c]
        denominator = sum(matrix[c]) + sum(line[c] for line in matrix)
        f1.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(matrix[c][c] for c in range(6)) / len(predictions), statistics.mean(f1), matrix, f1


def copy_evidence():
    for seed in SEEDS:
        source = ROOT / f"runs/mlp_yoonseop/M01_seed{seed}"
        target = REPORT / f"run_evidence/M01_seed{seed}"
        target.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.is_file() and path.name != "model.keras":
                shutil.copyfile(path, target / path.name)
                require(sha(path) == sha(target / path.name), "Evidence copy differs")
        shutil.copyfile(source.parent / f"{source.name}.log", target / "training.log")
    smoke = REPORT / "smoke_evidence"
    smoke.mkdir(parents=True, exist_ok=True)
    source = ROOT / "runs/mlp_yoonseop_smoke_fixed/M01_seed2026_smoke"
    target = smoke / "completed"
    target.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.name != "model.keras":
            shutil.copyfile(path, target / path.name)
    shutil.copyfile(source.parent / "M01_seed2026_smoke.log", target / "training.log")
    target = smoke / "initial_interrupted"
    target.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "runs/mlp_yoonseop_smoke/M01_seed2026_smoke/status.json", target / "status.json")
    shutil.copyfile(ROOT / "runs/mlp_yoonseop_smoke/M01_seed2026_smoke.log", target / "training.log")
    prior = ROOT / "MLP_HANDOFF_MANIFEST.json"
    preserved = REPORT / "local_handoff_manifest_20261012.json"
    if prior.exists():
        require(not preserved.exists(), "Initial handoff manifest already archived")
        prior.rename(preserved)


def prepare():
    copy_evidence()
    freeze = read_json(REPORT / "protocol_freeze.json")
    for entry in freeze["files"]:
        require(sha(ROOT / entry["path"]) == entry["sha256"], f"Original training file changed: {entry['path']}")
        require(sha(REPORT / "frozen_source" / entry["path"]) == entry["sha256"], "Frozen snapshot changed")
    digest = hashlib.sha256()
    for folder, suffix in (("src", ".py"), ("configs", ".json"), ("scripts", ".py")):
        for path in sorted((ROOT / folder).rglob(f"*{suffix}")):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    require(digest.hexdigest() == freeze["code_sha256"], "Training code fingerprint changed")
    suite = ET.parse(REPORT / "pytest_prepublication.xml").find("testsuite")
    require(int(suite.attrib["tests"]) == 48 and
            all(int(suite.attrib[field]) == 0 for field in ("errors", "failures", "skipped")), "Prepublication tests did not all pass")
    data_hash, stats_hash = sha(ROOT / "data/processed_data.npz"), sha(ROOT / "data/preprocess.npz")
    accuracy_values, f1_values, run_checks, all_errors = [], [], [], defaultdict(list)
    class_rows = rows(REPORT / "class_metrics.csv")
    subject_rows = rows(REPORT / "subject_metrics.csv")
    for seed in SEEDS:
        run = ROOT / f"runs/mlp_yoonseop/M01_seed{seed}"
        metadata, metrics = read_json(run / "metadata.json"), read_json(run / "metrics.json")
        check, config = read_json(run / "fresh_process_verification.json"), read_json(run / "config.json")
        require(read_json(run / "status.json")["status"] == "complete", "Incomplete formal run")
        require(config["seed"] == seed and config["experiment_id"] == "M01" and
                not metrics["test_evaluated"] and not metrics["smoke"] and metrics["parameters"] == 156230, "Wrong formal config")
        require(metadata["data_provenance"]["data_sha256"] == data_hash and
                metadata["data_provenance"]["source_stats_sha256"] == stats_hash and
                metadata["environment"]["code_sha256"] == digest.hexdigest(), "Run provenance differs")
        require(sha(run / "model.keras") == metadata["model_sha256"] and
                sha(run / "preprocess.npz") == metadata["preprocess_sha256"], "Saved model or preprocessing changed")
        require(check["passed"] and check["runs"][0]["count"] == 1775 and
                check["runs"][0]["labels_identical"], "Fresh-process verification failed")
        predictions = rows(run / "validation_predictions.csv")
        require(len(predictions) == 1775 and len({row["sample_id"] for row in predictions}) == 1775, "Prediction count or IDs differ")
        split = read_json(run / "split.json")
        require([row["sample_id"] for row in predictions] == split["validation_ids"], "Prediction order differs")
        accuracy, macro_f1, matrix, class_f1 = measure(predictions)
        require(abs(accuracy - metrics["accuracy"]) < 1e-12 and abs(macro_f1 - metrics["macro_f1"]) < 1e-12 and
                matrix == metrics["confusion_matrix"], "Metrics do not reproduce from the CSV")
        for row in class_rows:
            if row["run_id"] == run.name:
                label = int(row["label"])
                true_positive = matrix[label][label]
                predicted_count = sum(line[label] for line in matrix)
                true_count = sum(matrix[label])
                require(abs(float(row["f1"]) - class_f1[label]) < 1e-12 and
                        int(row["support"]) == true_count and
                        abs(float(row["precision"]) - (true_positive / predicted_count if predicted_count else 0)) < 1e-12 and
                        abs(float(row["recall"]) - (true_positive / true_count if true_count else 0)) < 1e-12, "Class metric differs")
        for subject in (1, 6, 14, 21, 23):
            subset = [row for row in predictions if int(row["subject"]) == subject]
            acc_subject, f1_subject, _, _ = measure(subset)
            saved = next(row for row in subject_rows if row["run_id"] == run.name and int(row["subject"]) == subject)
            require(abs(float(saved["accuracy"]) - acc_subject) < 1e-12 and
                    abs(float(saved["macro_f1"]) - f1_subject) < 1e-12 and int(saved["support"]) == len(subset), "Subject metric differs")
        for row in predictions:
            probabilities = [float(row[f"probability_{c}"]) for c in range(6)]
            logits = [float(row[f"logit_{c}"]) for c in range(6)]
            shifted = [math.exp(value - max(logits)) for value in logits]
            require(all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities) and
                    abs(sum(probabilities) - 1) < 1e-12 and probabilities.index(max(probabilities)) == int(row["predicted_label"]), "Invalid saved probabilities")
            require(max(abs(p - z / sum(shifted)) for p, z in zip(probabilities, shifted)) < 1e-12, "Logits and probabilities differ")
            if row["true_label"] != row["predicted_label"]:
                all_errors[row["sample_id"]].append((seed, int(row["predicted_label"])))
        history = rows(run / "history.csv")
        require(metrics["epochs_ran"] == len(history) and metrics["best_epoch"] ==
                min(range(len(history)), key=lambda i: float(history[i]["val_loss"])) + 1, "Selected epoch differs")
        accuracy_values.append(accuracy)
        f1_values.append(macro_f1)
        run_checks.append({"run_id": run.name, "validation_count": len(predictions), "accuracy": accuracy,
                           "macro_f1": macro_f1, "max_probability_reload_difference": check["runs"][0]["max_probability_difference"],
                           "model_sha256": metadata["model_sha256"], "all_saved_metrics_recomputed": True})
    summary = read_json(REPORT / "summary.json")
    for field, actual in (("accuracy_mean", statistics.mean(accuracy_values)), ("accuracy_sample_sd", statistics.stdev(accuracy_values)),
                          ("macro_f1_mean", statistics.mean(f1_values)), ("macro_f1_sample_sd", statistics.stdev(f1_values))):
        require(abs(actual - summary[field]) < 1e-12, f"Summary differs: {field}")
    consensus = rows(REPORT / "error_consensus.csv")
    require({row["sample_id"]: int(row["wrong_seed_count"]) for row in consensus} ==
            {sample_id: len(values) for sample_id, values in all_errors.items()}, "Repeated error counts differ")
    for row in consensus:
        require(row["wrong_seed_predictions"] == ";".join(f"{seed}:{prediction}" for seed, prediction in all_errors[row["sample_id"]]),
                "Repeated error predictions differ")
    for filename, source, key, fields in (("class_metrics_summary.csv", class_rows, "label", ("precision", "recall", "f1")),
                                         ("subject_metrics_summary.csv", subject_rows, "subject", ("accuracy", "macro_f1"))):
        for row in rows(REPORT / filename):
            selected = [value for value in source if value[key] == row[key]]
            require(len(selected) == 3 and {int(value["seed"]) for value in selected} == set(SEEDS), "Missing summary seed")
            require(all(int(value["support"]) == int(row["support_per_seed"]) for value in selected), "Summary support differs")
            for field in fields:
                values = [float(value[field]) for value in selected]
                require(abs(float(row[field + "_mean"]) - statistics.mean(values)) < 1e-12 and
                        abs(float(row[field + "_sample_sd"]) - statistics.stdev(values)) < 1e-12, "Group mean or SD differs")
    audit_path = REPORT / "publication_audit.json"
    audit = {"date_korea": "2026-10-12", "stage": "prepublication_review", "passed": False,
             "training_code_sha256": digest.hexdigest(), "unchanged_training_snapshot_files": len(freeze["files"]),
             "pytest_passed": 48, "runs": run_checks, "summary_recomputed": True,
             "class_and_subject_metrics_recomputed": True, "consensus_error_windows": sum(len(v) == 3 for v in all_errors.values()),
             "data_sha256": data_hash, "original_statistics_sha256": stats_hash,
             "models_or_results_changed_for_publication": False, "test_evaluated": False}
    write_json(audit_path, audit)
    links = 0
    for path in list((ROOT / "docs/team-paper/contributions/02_mlp").glob("*.md")) + [REPORT / "README.md"]:
        text = path.read_text(encoding="utf-8")
        require("[기입]" not in text and "[측정 예정]" not in text, "Unfilled MLP document")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("http:", "https:", "#")):
                continue
            require((path.parent / target.split("#")[0]).exists(), f"Broken link: {path.name}: {target}")
            links += 1
    audit.update(passed=True, local_document_links_checked=links)
    write_json(audit_path, audit)
    print("PREPUBLICATION AUDIT PASSED", len(run_checks), "models, 48 tests,", links, "document links")


def build(output):
    require(read_json(REPORT / "publication_audit.json")["passed"], "Prepare and review evidence first")
    state = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout
    require(not state.strip(), "Commit the reviewed changes before packaging")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8").split("\0")
    paths = {ROOT / name for name in tracked if name}
    paths.update({ROOT / "data/processed_data.npz", ROOT / "data/preprocess.npz"})
    for folder in ("runs/mlp_yoonseop", "runs/mlp_yoonseop_smoke_fixed"):
        paths.update(path for path in (ROOT / folder).rglob("*") if path.is_file())
    paths.update({ROOT / "runs/mlp_yoonseop_smoke/M01_seed2026_smoke.log",
                  ROOT / "runs/mlp_yoonseop_smoke/M01_seed2026_smoke/status.json"})
    entries = [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
               for path in sorted(paths)]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "MLP_RELEASE_MANIFEST.json"
    manifest = {"schema_version": 1, "date_korea": "2026-10-12", "contributor": "한윤섭", "release_tag": TAG,
                "repository": "kara320090/smartphone-har", "source_commit": commit,
                "training_code_sha256": read_json(REPORT / "protocol_freeze.json")["code_sha256"],
                "test_evaluated": False, "publication_audit_passed": True, "files": entries}
    write_json(manifest_path, manifest)
    archive_path = output / f"{TAG}-full.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            archive.write(path, "smartphone-har/" + path.relative_to(ROOT).as_posix())
        archive.write(manifest_path, "smartphone-har/MLP_RELEASE_MANIFEST.json")
    verification = verify_archive(archive_path)
    for path in (archive_path, manifest_path):
        require(path.stat().st_size > 0, "Empty release asset")
    sums_path = output / "SHA256SUMS.txt"
    sums_path.write_text("".join(f"{sha(path)}  {path.name}\n" for path in (archive_path, manifest_path)), encoding="utf-8")
    write_json(output / "build_verification.json", dict(verification, zip_sha256=sha(archive_path), zip_bytes=archive_path.stat().st_size))
    print("RELEASE BUILT AND VERIFIED", commit, len(entries), "files,", archive_path.stat().st_size, "bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--prepare", action="store_true")
    operation.add_argument("--build", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work/mlp_release")
    args = parser.parse_args()
    prepare() if args.prepare else build(args.output_dir)


if __name__ == "__main__":
    main()
