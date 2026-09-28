"""Aggregate measured predictions without test access or significance claims."""
import csv,json,sys,shutil
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/process_audit'
IDS=['E02','E03','E07','E08','E09','E10','E11']
SEEDS=[2026,2027,2028]

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write_csv(p,rows):
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def preds(p):
    rows=read_csv(p/'validation_predictions.csv')
    return rows,np.array([int(r['true_label']) for r in rows]),np.array([int(r['predicted_label']) for r in rows]),np.array([[float(r[f'probability_{i}']) for i in range(6)] for r in rows])
flat=[];per_subject=[];by_run={};replay=[]
for directory,experiments in [('process_repeats',IDS),('process_matched',['E08W'])]:
    for exp in experiments:
        for seed in SEEDS:
            p=ROOT/'runs'/directory/f'{exp}_seed{seed}'
            assert json.loads((p/'status.json').read_text())['status']=='complete'
            m=json.loads((p/'metrics.json').read_text()); assert not m['test_evaluated']
            meta=json.loads((p/'metadata.json').read_text())
            rows,y,pred,prob=preds(p)
            assert len(y)==1775
            np.testing.assert_allclose(f1_score(y,pred,labels=list(range(6)),average='macro',zero_division=0),m['macro_f1'],atol=1e-12)
            record={'experiment':exp,'seed':seed,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],
                    'parameters':m['parameters'],'best_epoch':m['best_epoch'],'epochs_ran':m['epochs_ran'],
                    'best_val_loss':m['best_val_loss'],'code_sha256':meta['environment']['code_sha256'],
                    'git_commit':meta['environment']['git_commit']}
            flat.append(record);by_run[(exp,seed)]=record
            dest=OUT/'run_records'/f'{exp}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
            for name in ['metrics.json','config.json','history.csv','reload_check.json','metadata.json','initialization_match.json']:
                if (p/name).exists():
                    shutil.copy2(p/name,dest/name)
                    if name.endswith('.csv'):
                        # Normalize only the review copy; preserve original run bytes.
                        (dest/name).write_bytes((dest/name).read_bytes().replace(b'\r\r\n',b'\n').replace(b'\r\n',b'\n'))
            if exp=='E02':
                subjects=np.array([int(r['subject']) for r in rows])
                for subject in sorted(set(subjects)):
                    mask=subjects==subject
                    per_subject.append({'seed':seed,'subject':int(subject),'samples':int(mask.sum()),
                        'accuracy':float(accuracy_score(y[mask],pred[mask])),
                        'macro_f1':float(f1_score(y[mask],pred[mask],labels=list(range(6)),average='macro',zero_division=0)),
                        'class_count':len(set(y[mask]))})
            if seed==2026 and exp in IDS:
                old_rows,old_y,old_pred,old_prob=preds(ROOT/'runs/formal'/f'{exp}_seed2026')
                assert [r['sample_id'] for r in rows]==[r['sample_id'] for r in old_rows]
                replay.append({'experiment':exp,'count':len(y),'labels_identical':bool(np.array_equal(pred,old_pred)),
                    'probabilities_identical':bool(np.array_equal(prob,old_prob)),
                    'maximum_probability_difference':float(np.max(np.abs(prob-old_prob)))})
summaries=[]
for exp in IDS+['E08W']:
    rr=[by_run[(exp,s)] for s in SEEDS]
    f=np.array([r['macro_f1'] for r in rr]);a=np.array([r['accuracy'] for r in rr])
    summaries.append({'experiment':exp,'seeds':3,'f1_mean':float(f.mean()),'f1_sample_sd':float(f.std(ddof=1)),
        'f1_min':float(f.min()),'f1_max':float(f.max()),'accuracy_mean':float(a.mean()),
        'accuracy_sample_sd':float(a.std(ddof=1)), 'parameters':rr[0]['parameters'],
        'delta_vs_e02_pp':float((f-np.array([by_run[('E02',s)]['macro_f1'] for s in SEEDS])).mean()*100)})
paired=[{'experiment':exp,'seed':s,'delta_f1_pp':(by_run[(exp,s)]['macro_f1']-by_run[('E02',s)]['macro_f1'])*100} for exp in IDS[1:]+['E08W'] for s in SEEDS]
checkpoint=[]
for exp in ['E02','E07']:
    for seed in SEEDS:
        r=by_run[(exp,seed)];checkpoint.append({k:r[k] for k in ['experiment','seed','macro_f1','best_val_loss','best_epoch','epochs_ran']})
summary={'base_conditions':7,'base_repeats':21,'matched_control_runs':3,'total_new_training_runs':24,
    'seeds':SEEDS,'test_evaluated':False,'sample_sd_ddof':1,'summaries':summaries,'per_subject':per_subject,
    'original_seed_replay':replay,'paired_deltas':paired,'checkpoint_comparison':checkpoint,
    'training_code_hashes':sorted(set(r['code_sha256'] for r in flat)),
    'qualification':'Same person split. Seed SD describes initialization/shuffling variability only, not population uncertainty.'}
assert len(summary['training_code_hashes'])==1
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
write_csv(OUT/'repeat_results.csv',flat);write_csv(OUT/'condition_summary.csv',summaries)
write_csv(OUT/'e02_by_subject.csv',per_subject);write_csv(OUT/'paired_deltas.csv',paired)
print(json.dumps({'summaries':summaries,'replay':replay,'subjects':per_subject},indent=2))
