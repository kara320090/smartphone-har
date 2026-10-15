"""Audit post-hoc output grouping of existing development predictions.

Grouping changes the task. It is not a model improvement, new training,
probability aggregation, a confidence threshold, or an official test score.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import statistics
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
LEVELS = {
    "six": {"mapping": [0,1,2,3,4,5], "labels": ["걷기","계단 오르기","계단 내리기","앉기","서기","눕기"]},
    "three": {"mapping": [0,0,0,1,1,2], "labels": ["이동","앉거나 서 있음","누워 있음"]},
    "two": {"mapping": [0,0,0,1,1,1], "labels": ["이동","정적 자세"]},
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_out(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def evaluate(rows, mapping):
    size=max(mapping)+1
    cm=[[0]*size for _ in range(size)]
    y=[mapping[int(r["true_label"])] for r in rows]
    p=[mapping[int(r["predicted_label"])] for r in rows]
    for true,pred in zip(y,p):cm[true][pred]+=1
    errors=len(y)-sum(cm[i][i] for i in range(size))
    accuracy=1-errors/len(y)
    macro=statistics.mean(2*cm[i][i]/(sum(cm[i])+sum(r[i] for r in cm)) if sum(cm[i])+sum(r[i] for r in cm) else 0 for i in range(size))
    assert abs(accuracy-accuracy_score(y,p))<1e-12
    assert abs(macro-f1_score(y,p,labels=list(range(size)),average="macro",zero_division=0))<1e-12
    return {"count":len(y),"errors":errors,"accuracy":accuracy,"macro_f1":macro},cm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"reports/output_granularity")
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory; previous analyses are preserved")
    output.mkdir(parents=True,exist_ok=True)
    specs=[
        ("MLP","M01","baseline",ROOT/"reports/mlp_yoonseop/run_evidence"),
        ("CNN","C01","baseline",ROOT/"artifacts/cnn_tensorflow"),
        ("RNN","R01","baseline",ROOT/"reports/rnn/run_evidence"),
        ("LSTM","L01","baseline",ROOT/"runs/lstm"),
        ("LSTM","L02","supplementary",ROOT/"runs/lstm"),
        ("LSTM","L03","supplementary",ROOT/"runs/lstm"),
    ]
    refs={(r["config_id"],int(r["seed"])):r for r in read_csv(ROOT/"reports/team_comparison/runs_recomputed.csv")}
    refs.update({(r["experiment"],int(r["seed"])):r for r in read_csv(ROOT/"reports/lstm/runs.csv")})
    binding=None
    runs=[];subjects=[];evidence=[];matrices={};examples=[]
    for model,config,role,folder in specs:
        for seed in (2026,2027,2028):
            path=folder/f"{config}_seed{seed}"/"validation_predictions.csv"
            rows=read_csv(path)
            current=[(r["sample_id"],int(r["subject"]),int(r["true_label"])) for r in rows]
            assert len(rows)==1775 and len(set(r[0] for r in current))==1775
            assert set(r[1] for r in current)=={1,6,14,21,23}
            assert all(0<=int(r[k])<6 for r in rows for k in ("true_label","predicted_label"))
            if binding is None:binding=current
            assert binding==current
            identity={"model":model,"condition":config,"role":role,"seed":seed}
            fine_errors=None
            for level,spec in LEVELS.items():
                metrics,cm=evaluate(rows,spec["mapping"])
                if level=="six":
                    fine_errors=metrics["errors"]
                    for key in ("accuracy","macro_f1"):
                        assert abs(metrics[key]-float(refs[(config,seed)][key]))<1e-12
                run={**identity,"level":level,**metrics,
                     "fine_errors":fine_errors,"within_group_fine_errors":fine_errors-metrics["errors"],
                     "within_group_fraction":(fine_errors-metrics["errors"])/fine_errors if fine_errors else 0.0}
                runs.append(run)
                matrices[f"{config}_seed{seed}_{level}"]=cm
                for sid in (1,6,14,21,23):
                    person,_=evaluate([r for r in rows if int(r["subject"])==sid],spec["mapping"])
                    subjects.append({**identity,"level":level,"subject":sid,**person})
            evidence.append({"file":path.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
            if config=="L02":
                for r in rows:
                    fine_true,fine_pred=int(r["true_label"]),int(r["predicted_label"])
                    if fine_true!=fine_pred:
                        mapping=LEVELS["three"]["mapping"]
                        examples.append({**identity,"sample_id":r["sample_id"],"subject":int(r["subject"]),"true_label":fine_true,"predicted_label":fine_pred,"true_three_group":mapping[fine_true],"predicted_three_group":mapping[fine_pred],"three_group_error":mapping[fine_true]!=mapping[fine_pred]})
    summaries=[]
    for model,config,role,_ in specs:
        for level in LEVELS:
            group=[r for r in runs if r["condition"]==config and r["level"]==level]
            by_subject={sid:statistics.mean(r["macro_f1"] for r in subjects if r["condition"]==config and r["level"]==level and r["subject"]==sid) for sid in (1,6,14,21,23)}
            summaries.append({"model":model,"condition":config,"role":role,"level":level,"repetitions":3,
                "macro_f1_mean":statistics.mean(r["macro_f1"] for r in group),
                "macro_f1_seed_sample_sd":statistics.stdev(r["macro_f1"] for r in group),
                "accuracy_mean":statistics.mean(r["accuracy"] for r in group),
                "errors_each_seed":",".join(str(r["errors"]) for r in group),
                "lowest_subject_by_seed_mean":min(by_subject,key=by_subject.get),
                "lowest_subject_macro_f1_seed_mean":min(by_subject.values()),
                "fine_error_events_across_seeds":sum(r["fine_errors"] for r in group),
                "coarse_error_events_across_seeds":sum(r["errors"] for r in group),
                "within_group_fraction_of_fine_error_events":sum(r["within_group_fine_errors"] for r in group)/sum(r["fine_errors"] for r in group),
            })
    csv_out(output/"runs.csv",runs)
    csv_out(output/"subject_metrics.csv",subjects)
    csv_out(output/"summary.csv",summaries)
    csv_out(output/"l02_error_examples.csv",examples)
    report={"date_korea":"2026-10-15","passed":True,
        "post_hoc":True,"new_training":False,"test_evaluated":False,"fresh_model_inference":False,
        "grouping_rule":"Map both saved six-class argmax predictions and ground truth to a fixed group. No sum-of-probabilities decision or confidence threshold.",
        "task_levels":LEVELS,"unique_validation_windows":1775,"prediction_rows_checked":18*1775,
        "runs_checked":18,"global_metric_cases":len(runs),"subject_metric_cases":len(subjects),
        "sklearn_independent_metric_checks":len(runs)+len(subjects),
        "same_sample_subject_label_order":True,"original_six_class_metrics_matched":True,
        "repeated_seeds_are_not_independent_new_users":True,
        "app_scope":"Prepared-window development evidence for activity-state summaries, not continuous activity logging, sleep, sedentary-duration, transition or real-phone validation.",
        "limitations":["Tasks differ; grouped score is not improvement over six-class recognition.","Mappings and application framing are post-hoc, not preregistered.","Coarser tasks also need simple feature-based or threshold baselines before claiming that LSTM is required.","One development subject split used for checkpoint selection; official test remains withheld.","No unknown-activity, phone-placement, transition, continuous-time or end-user study."],
        "input_sha256":evidence,"confusion_matrices":matrices,"summaries":summaries}
    (output/"analysis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({"passed":True,"global_cases":len(runs),"subject_cases":len(subjects),"summaries":summaries},ensure_ascii=False))


if __name__=="__main__":main()
