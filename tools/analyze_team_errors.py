"""Describe class and held-out-subject errors of the four fixed configurations.

This is post-hoc development analysis of saved validation predictions, not an
additional model search or official test evaluation.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import statistics


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lstm-runs',type=Path,default=Path('runs/lstm'))
    parser.add_argument('--output',type=Path,default=Path('reports/team_error_analysis'))
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    specs=[('MLP','M01',repo/'reports/mlp_yoonseop/run_evidence'),('CNN','C01',repo/'artifacts/cnn_tensorflow'),('RNN','R01',repo/'reports/rnn/run_evidence'),('LSTM','L01',args.lstm_runs.resolve())]
    classes=['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
    summary={r['config_id']:r for r in read_csv(repo/'reports/team_comparison/baseline_summary.csv')}
    subject_source=repo/'reports/team_comparison/subject_metrics.csv'
    subject_rows=[r for r in read_csv(subject_source) if r['role']=='baseline']
    subject_reference={(r['config_id'],int(r['seed']),int(r['subject'])):r for r in subject_rows}
    run_reference={(r['config_id'],int(r['seed'])):r for r in read_csv(repo/'reports/team_comparison/runs_recomputed.csv') if r['role']=='baseline'}
    per_class,errors,evidence=[],[],[]
    matrices={};binding=None
    for model,config,folder in specs:
        matrices[config]=[]
        for seed in (2026,2027,2028):
            path=folder/f'{config}_seed{seed}'/'validation_predictions.csv'
            rows=read_csv(path)
            assert len(rows)==1775
            assert len({r['sample_id'] for r in rows})==1775
            assert all(0<=int(r['true_label'])<6 and 0<=int(r['predicted_label'])<6 for r in rows)
            assert {int(r['subject']) for r in rows}=={1,6,14,21,23}
            current=[(r['sample_id'],r['subject'],r['true_label']) for r in rows]
            if binding is None:binding=current
            assert current==binding
            cm=[[0]*6 for _ in range(6)]
            for r in rows:cm[int(r['true_label'])][int(r['predicted_label'])]+=1
            reference=run_reference[(config,seed)]
            accuracy=sum(cm[c][c] for c in range(6))/1775
            macro=statistics.mean(2*cm[c][c]/(sum(cm[c])+sum(row[c] for row in cm)) for c in range(6))
            assert abs(accuracy-float(reference['accuracy']))<1e-12
            assert abs(macro-float(reference['macro_f1']))<1e-12
            # Recompute each subject directly as well; do not trust a stale CSV.
            for sid in (1,6,14,21,23):
                subset=[r for r in rows if int(r['subject'])==sid]
                subject_cm=[[0]*6 for _ in range(6)]
                for r in subset:subject_cm[int(r['true_label'])][int(r['predicted_label'])]+=1
                person=subject_reference[(config,seed,sid)]
                subject_acc=sum(subject_cm[c][c] for c in range(6))/len(subset)
                subject_f1=statistics.mean(2*subject_cm[c][c]/(sum(subject_cm[c])+sum(row[c] for row in subject_cm)) if sum(subject_cm[c])+sum(row[c] for row in subject_cm) else 0.0 for c in range(6))
                assert len(subset)==int(person['count'])
                assert abs(subject_acc-float(person['accuracy']))<1e-12
                assert abs(subject_f1-float(person['macro_f1_six_classes']))<1e-12
            matrices[config].append(cm)
            for label,name in enumerate(classes):
                support=sum(cm[label]);predicted=sum(row[label] for row in cm)
                precision=cm[label][label]/predicted if predicted else 0.0
                recall=cm[label][label]/support
                f1=2*cm[label][label]/(support+predicted) if support+predicted else 0.0
                per_class.append({'model':model,'config_id':config,'seed':seed,'label':label,'class':name,'support':support,'precision':precision,'recall':recall,'f1':f1})
            errors.append({'model':model,'config_id':config,'seed':seed,'sitting_support':sum(cm[3]),'sitting_to_standing_count':cm[3][4],'sitting_to_standing_rate':cm[3][4]/sum(cm[3]),'standing_support':sum(cm[4]),'standing_to_sitting_count':cm[4][3],'standing_to_sitting_rate':cm[4][3]/sum(cm[4])})
            evidence.append({'run_id':f'{config}_seed{seed}','file':path.relative_to(repo).as_posix() if path.is_relative_to(repo) else f'external_lstm/{config}_seed{seed}/validation_predictions.csv','sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    class_summary=[];confusions=[]
    for model,config,_ in specs:
        for label,name in enumerate(classes):
            group=[r for r in per_class if r['config_id']==config and r['label']==label]
            class_summary.append({'model':model,'config_id':config,'label':label,'class':name,'support_per_seed':group[0]['support'],'f1_mean':statistics.mean(r['f1'] for r in group),'f1_sample_sd':statistics.stdev(r['f1'] for r in group),'precision_mean':statistics.mean(r['precision'] for r in group),'recall_mean':statistics.mean(r['recall'] for r in group)})
        assert abs(statistics.mean(r['f1_mean'] for r in class_summary if r['config_id']==config)-float(summary[config]['macro_f1_mean']))<1e-12
        for true in range(6):
            for pred in range(6):
                counts=[cm[true][pred] for cm in matrices[config]]
                support=sum(matrices[config][0][true])
                confusions.append({'model':model,'config_id':config,'true_label':true,'predicted_label':pred,'true_class':classes[true],'predicted_class':classes[pred],'support_per_seed':support,'mean_count_per_seed':statistics.mean(counts),'mean_conditional_rate':statistics.mean(counts)/support})
    subjects=[]
    for model,config,_ in specs:
        for subject in (1,6,14,21,23):
            group=[r for r in subject_rows if r['config_id']==config and int(r['subject'])==subject]
            assert len(group)==3
            subjects.append({'model':model,'config_id':config,'subject':subject,'support_per_seed':int(group[0]['count']),'macro_f1_mean':statistics.mean(float(r['macro_f1_six_classes']) for r in group),'macro_f1_sample_sd':statistics.stdev(float(r['macro_f1_six_classes']) for r in group),'accuracy_mean':statistics.mean(float(r['accuracy']) for r in group)})
    rates=[]
    for model,config,_ in specs:
        group=[r for r in errors if r['config_id']==config]
        rates.append({'model':model,'config_id':config,'sitting_to_standing_mean_count':statistics.mean(r['sitting_to_standing_count'] for r in group),'sitting_to_standing_rate_mean':statistics.mean(r['sitting_to_standing_rate'] for r in group),'sitting_to_standing_rate_sample_sd':statistics.stdev(r['sitting_to_standing_rate'] for r in group),'standing_to_sitting_rate_mean':statistics.mean(r['standing_to_sitting_rate'] for r in group)})
    behavior_groups=[]
    for model,config,_ in specs:
        for group_name,labels in [('dynamic',[0,1,2]),('static',[3,4,5])]:
            values=[statistics.mean(r['f1'] for r in per_class if r['config_id']==config and r['seed']==seed and r['label'] in labels) for seed in (2026,2027,2028)]
            behavior_groups.append({'model':model,'config_id':config,'group':group_name,'labels':','.join(map(str,labels)),'mean_class_f1':statistics.mean(values),'seed_sample_sd':statistics.stdev(values),'definition':'mean of three class F1 values from the full six-class confusion matrix; not a refitted three-class task'})
    quantities={
        'lstm_parameters_fewer_than_cnn_fraction':1-int(summary['L01']['parameters'])/int(summary['C01']['parameters']),
        'lstm_parameters_fewer_than_mlp_fraction':1-int(summary['L01']['parameters'])/int(summary['M01']['parameters']),
        'cnn_minus_lstm_macro_f1_mean':float(summary['C01']['macro_f1_mean'])-float(summary['L01']['macro_f1_mean']),
        'lstm_minus_mlp_macro_f1_mean':float(summary['L01']['macro_f1_mean'])-float(summary['M01']['macro_f1_mean']),
    }
    for name,rows in [('class_runs.csv',per_class),('class_summary.csv',class_summary),('confusion_mean.csv',confusions),('subject_summary.csv',subjects),('directional_error_runs.csv',errors),('directional_error_summary.csv',rates),('behavior_group_summary.csv',behavior_groups)]:write_csv(output/name,rows)
    report={'date_korea':'2026-10-13','passed':True,'scope':'post-hoc development analysis of M01 C01 R01 L01; same fixed subject split, three training seeds','prediction_rows_checked':12*1775,'classes':classes,'class_macro_matches_overall':True,'per_run_and_subject_metrics_recomputed_and_matched':True,'test_evaluated':False,'new_training':False,'quantities':quantities,'subject_metrics_source_sha256':hashlib.sha256(subject_source.read_bytes()).hexdigest(),'evidence':evidence,'interpretation_limits':['mean class F1 is mean of per-seed class F1, not F1 of pooled predictions','directional rates average three decisions on the same windows; not independent new samples','subject analysis is descriptive on five development subjects; no population confidence interval','parameter count does not measure phone latency, energy, or deployment suitability','no measured causal explanation of gate effects or gradient vanishing']}
    (output/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'passed':True,'prediction_rows':12*1775,'quantities':quantities,'directional_errors':rates},ensure_ascii=True))


if __name__=='__main__':main()
