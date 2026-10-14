"""Diagnose validation composition sensitivity without refitting any model.

All five prespecified development subjects are retained in the headline score.
The five leave-one-validation-subject-out summaries are post-hoc diagnostics,
not leave-one-subject-out training, independent tests, or improved model scores.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics

SUBJECTS=(1,6,14,21,23)
SEEDS=(2026,2027,2028)

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as file:
        return list(csv.DictReader(file))

def score(rows):
    cm=[[0]*6 for _ in range(6)]
    for row in rows:
        cm[int(row['true_label'])][int(row['predicted_label'])]+=1
    f1=[]
    for c in range(6):
        den=sum(cm[c])+sum(row[c] for row in cm)
        f1.append(2*cm[c][c]/den if den else 0.0)
    return statistics.mean(f1),sum(cm[c][c] for c in range(6))/len(rows),cm

def write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lstm-runs',type=Path,default=Path('runs/lstm'))
    parser.add_argument('--output',type=Path,default=Path('reports/subject_composition'))
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    specs=[('MLP','M01',repo/'reports/mlp_yoonseop/run_evidence'),('CNN','C01',repo/'artifacts/cnn_tensorflow'),('RNN','R01',repo/'reports/rnn/run_evidence'),('LSTM','L01',args.lstm_runs.resolve())]
    refs={(r['config_id'],int(r['seed'])):r for r in read(repo/'reports/team_comparison/runs_recomputed.csv') if r['role']=='baseline'}
    baseline={r['config_id']:r for r in read(repo/'reports/team_comparison/baseline_summary.csv')}
    runs=[];evidence=[];binding=None;sklearn_checks=0
    try:
        from sklearn.metrics import f1_score,accuracy_score
    except ImportError:
        f1_score=accuracy_score=None
    for model,config,folder in specs:
        for seed in SEEDS:
            path=folder/f'{config}_seed{seed}/validation_predictions.csv'
            rows=read(path)
            assert len(rows)==1775 and len({r['sample_id'] for r in rows})==1775
            assert {int(r['subject']) for r in rows}==set(SUBJECTS)
            assert all(0<=int(r['true_label'])<6 and 0<=int(r['predicted_label'])<6 for r in rows)
            current=[(r['sample_id'],r['subject'],r['true_label']) for r in rows]
            if binding is None:binding=current
            assert current==binding
            evidence.append({'run_id':f'{config}_seed{seed}','file':path.relative_to(repo).as_posix() if path.is_relative_to(repo) else f'external_lstm/{config}_seed{seed}/validation_predictions.csv','sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            for removed in (None,)+SUBJECTS:
                subset=[r for r in rows if removed is None or int(r['subject'])!=removed]
                macro,accuracy,cm=score(subset)
                assert all(sum(row)>0 for row in cm), 'All six true labels must remain'
                if removed is None:
                    assert math.isclose(macro,float(refs[(config,seed)]['macro_f1']),abs_tol=1e-12)
                    assert math.isclose(accuracy,float(refs[(config,seed)]['accuracy']),abs_tol=1e-12)
                else:
                    assert len(subset)==1775-sum(int(r['subject'])==removed for r in rows)
                if f1_score is not None:
                    true=[int(r['true_label']) for r in subset];pred=[int(r['predicted_label']) for r in subset]
                    assert math.isclose(macro,f1_score(true,pred,labels=list(range(6)),average='macro',zero_division=0),abs_tol=1e-12)
                    assert math.isclose(accuracy,accuracy_score(true,pred),abs_tol=1e-12)
                    sklearn_checks+=1
                runs.append({'model':model,'config_id':config,'seed':seed,'removed_subject':'none' if removed is None else removed,'retained_subjects':','.join(str(s) for s in SUBJECTS if s!=removed),'count':len(subset),'macro_f1':macro,'accuracy':accuracy})
    summary=[]
    for removed in ('none',)+SUBJECTS:
        for model,config,_ in specs:
            group=[r for r in runs if r['config_id']==config and r['removed_subject']==removed]
            assert len(group)==3 and len({r['count'] for r in group})==1
            summary.append({'model':model,'config_id':config,'removed_subject':removed,'count_per_seed':group[0]['count'],'macro_f1_mean':statistics.mean(r['macro_f1'] for r in group),'macro_f1_seed_sample_sd':statistics.stdev(r['macro_f1'] for r in group)})
            if removed=='none':
                assert math.isclose(summary[-1]['macro_f1_mean'],float(baseline[config]['macro_f1_mean']),abs_tol=1e-12)
    gaps=[]
    for removed in ('none',)+SUBJECTS:
        for a,b in [('C01','L01'),('L01','M01')]:
            values=[]
            for seed in SEEDS:
                value=lambda c:next(r['macro_f1'] for r in runs if r['config_id']==c and r['seed']==seed and r['removed_subject']==removed)
                values.append(value(a)-value(b))
            gaps.append({'removed_subject':removed,'left_config':a,'right_config':b,'mean_macro_f1_difference':statistics.mean(values),'difference_seed_sample_sd':statistics.stdev(values),'seed_differences':','.join(str(v) for v in values),'mean_higher_config':a if statistics.mean(values)>0 else b})
    write(output/'runs.csv',runs);write(output/'summary.csv',summary);write(output/'paired_seed_gaps.csv',gaps)
    report={'date_korea':'2026-10-14','passed':True,'scope':'post-hoc development validation composition diagnostic; no refitting; every one of five subjects omitted in turn','headline_subjects':list(SUBJECTS),'headline_count':1775,'seeds':list(SEEDS),'prediction_rows_checked':21300,'metric_cases_recomputed':len(runs),'sklearn_independent_metric_checks':sklearn_checks,'full_scores_match_previous_audit':True,'all_six_labels_present_in_every_retained_subset':True,'prediction_id_subject_label_binding_same_for_all_runs':True,'headline_result_changed':False,'new_training_performed':False,'test_evaluated':False,'evidence':evidence,'interpretation_limits':['Not LOSO cross-validation: no model is retrained.','Removing a validation subject changes sample count and class composition as well as subject composition.','Omitted-person scores are diagnostic, not new headline performance or estimates for unseen users.','All five omissions are reported; low-scoring subjects are retained in the headline result.','Seed SD is descriptive on fixed validation subsets; no population interval or significance test.']}
    (output/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    (output/'README.md').write_text('''# 검증 피험자 구성 민감도 진단

기존 M01·C01·R01·L01의 세 seed, 12회 저장 예측으로 계산했다. 전체 5명·1,775구간은 대표 결과로 유지한다. 검증 피험자 1·6·14·21·23을 각각 한 명씩 제외해 나머지의 6클래스 Macro F1을 다시 계산했다.

이는 재학습 없는 사후 진단이며 LOSO 교차검증, 새로운 분할 학습, 독립 test 또는 저성능 피험자 제거에 의한 성능 향상이 아니다. 제외 시 표본 수와 클래스 구성도 함께 달라지므로 피험자 한 명의 인과 효과로 해석하지 않는다. 다섯 제외 경우를 모두 보고한다.

`runs.csv`: 4구성×3seed×(전체+5제외)의 72개 지표. `summary.csv`: 24개 조건의 seed 평균·표본 SD. `paired_seed_gaps.csv`: CNN-LSTM 및 LSTM-MLP의 같은 seed 차이. `analysis.json`: 원본 예측 해시·지표 대조·해석 범위.

```powershell
python tools/analyze_subject_composition.py --lstm-runs runs/lstm --output reports/subject_composition
```

LSTM 전체 예측은 기존 LSTM Release의 실행 폴더에서 준비한다. scikit-learn이 설치되어 있으면 72개 계산을 독립 지표 구현과 대조한다. 기본 계산은 Python 표준 라이브러리만 사용한다.
''',encoding='utf-8',newline='\n')
    print(json.dumps({'passed':True,'metric_cases':len(runs),'sklearn_checks':sklearn_checks,'cnn_lstm_gaps':[{k:r[k] for k in ['removed_subject','mean_macro_f1_difference','mean_higher_config']} for r in gaps if r['left_config']=='C01']},ensure_ascii=True))

if __name__=='__main__':main()
