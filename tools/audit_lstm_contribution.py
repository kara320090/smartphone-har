"""Audit existing LSTM evidence for a paper discussion; no training or test access."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/lstm"
OUT = ROOT / "docs/research-contribution-review-20261015"


def read_csv(name):
    with (SOURCE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    runs = read_csv("runs.csv")
    published = {r["experiment"]: r for r in read_csv("summary.csv")}
    paired = {(r["experiment"], int(r["seed"])): r for r in read_csv("paired_differences.csv")}
    people = read_csv("subject_metrics.csv")
    sources = [SOURCE / n for n in ("runs.csv", "summary.csv", "paired_differences.csv", "subject_metrics.csv", "initialization_audit.json", "dataset_audit.json", "delivery_verification.json")]
    audited = []
    assert len(runs) == 9
    expected_counts = [322, 256, 241, 309, 327, 320]
    for row in runs:
        path = SOURCE / (row["run_id"] + "_confusion.json")
        sources.append(path)
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["rows"] == "true" and saved["columns"] == "predicted"
        cm = saved["matrix"]
        assert len(cm) == 6 and all(len(r) == 6 for r in cm)
        assert [sum(r) for r in cm] == expected_counts
        support = sum(map(sum, cm))
        accuracy = sum(cm[i][i] for i in range(6)) / support
        scores = [2 * cm[i][i] / (sum(cm[i]) + sum(r[i] for r in cm)) for i in range(6)]
        f1 = statistics.mean(scores)
        assert math.isclose(f1, float(row["macro_f1"]), abs_tol=1e-12, rel_tol=0)
        assert math.isclose(accuracy, float(row["accuracy"]), abs_tol=1e-12, rel_tol=0)
        assert support == int(row["validation_count"]) == 1775
        audited.append({"run_id": row["run_id"], "condition": row["experiment"], "seed": int(row["seed"]), "macro_f1": f1, "accuracy": accuracy, "parameters": int(row["parameters"])})

    summaries = {}
    by_seed = {(r["condition"], r["seed"]): r for r in audited}
    configs = {}
    for condition in ("L01", "L02", "L03"):
        cfg_path = ROOT / "configs" / (condition + ".json")
        sources.append(cfg_path)
        cfg = configs[condition] = json.loads(cfg_path.read_text(encoding="utf-8"))
        group = [r for r in audited if r["condition"] == condition]
        assert sorted(r["seed"] for r in group) == [2026, 2027, 2028]
        u = cfg["units"]
        count = 4 * u * (9 + u + 1) + (u + 1) * 32 + (32 + 1) * 6
        assert all(r["parameters"] == count for r in group)
        mean = statistics.mean(r["macro_f1"] for r in group)
        sd = statistics.stdev(r["macro_f1"] for r in group)
        assert math.isclose(mean, float(published[condition]["macro_f1_mean"]), abs_tol=1e-12, rel_tol=0)
        assert math.isclose(sd, float(published[condition]["macro_f1_sample_sd"]), abs_tol=1e-12, rel_tol=0)
        subject_means = {}
        for sid in (1, 6, 14, 21, 23):
            group_subject = [r for r in people if r["run_id"].startswith(condition + "_") and int(r["subject"]) == sid]
            assert len(group_subject) == 3
            subject_means[str(sid)] = statistics.mean(float(r["macro_f1"]) for r in group_subject)
        summaries[condition] = {"units": u, "dropout": cfg["dropout"], "parameters": count, "macro_f1_mean": mean, "macro_f1_seed_sample_sd": sd, "subject_macro_f1_seed_means": subject_means, "lowest_subject": min(subject_means, key=subject_means.get)}
    assert {k: v for k, v in configs["L01"].items() if k not in ("units", "experiment_id")} == {k: v for k, v in configs["L02"].items() if k not in ("units", "experiment_id")}
    deltas = {}
    for condition in ("L02", "L03"):
        values = []
        for seed in (2026, 2027, 2028):
            difference = by_seed[(condition, seed)]["macro_f1"] - by_seed[("L01", seed)]["macro_f1"]
            assert math.isclose(difference, float(paired[(condition, seed)]["macro_f1_delta"]), abs_tol=1e-12, rel_tol=0)
            values.append({"seed": seed, "delta_macro_f1": difference})
        deltas[condition + "_minus_L01"] = {"per_seed": values, "mean": statistics.mean(r["delta_macro_f1"] for r in values), "sample_sd": statistics.stdev(r["delta_macro_f1"] for r in values), "positive_seeds": sum(r["delta_macro_f1"] > 0 for r in values)}
    initialization = json.loads((SOURCE / "initialization_audit.json").read_text(encoding="utf-8"))
    assert initialization["passed"] and all(p["all_values_identical"] for p in initialization["pairs"])
    dataset = json.loads((SOURCE / "dataset_audit.json").read_text(encoding="utf-8"))
    assert dataset["test_arrays_loaded"] is False
    delivery = json.loads((SOURCE / "delivery_verification.json").read_text(encoding="utf-8"))
    assert delivery["passed"] and delivery["formal_runs"] == 9 and delivery["test_evaluated"] is False
    sources.extend([ROOT / "docs/LSTM_EXECUTION_PLAN_KO.md", ROOT / "docs/dcs-team-paper-research-v7/DCS_TEAM_PAPER_KO.md"])
    result = {
        "review_date_korea": "2026-10-15", "passed": True,
        "scope": "Recomputed nine existing global confusion matrices; aggregated existing audited subject metrics. No new training, inference, or access to official test.",
        "independent_subject_prediction_reaudit_this_revision": False,
        "conditions": summaries, "paired_seed_differences": deltas,
        "L02_parameter_reduction_vs_L01_percent": 100 * (1 - summaries["L02"]["parameters"] / summaries["L01"]["parameters"]),
        "L01_L02_config_difference": ["units", "experiment_id"],
        "L03_pretraining_initialization_matches_L01": True,
        "L02_initialization_matches_L01": False,
        "runs_recomputed_from_confusion": audited,
        "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "interpretation": "Exploratory development evidence. Same-seed labels are not identical weights across different sizes. Seed SD is not a population confidence interval. L02 is not substituted into the four-model baseline ranking."
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "EVIDENCE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"passed": True, "recomputed_confusion_matrices": len(audited), "parameter_reduction_percent": result["L02_parameter_reduction_vs_L01_percent"], "deltas": deltas, "subject14_L02": summaries["L02"]["subject_macro_f1_seed_means"]["14"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
