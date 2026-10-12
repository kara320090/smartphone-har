"""Recompute validation results from saved predictions; retain all seed runs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from .training_metrics import classification_metrics
from .runtime import write_json
from .adapters.uci_reference import CLASSES


def read_rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    if not rows:
        return
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze(runs_dir,out):
    out=Path(out)
    figs=out/'figures'
    figs.mkdir(parents=True,exist_ok=True)
    runs=[]; classes=[]; people=[]; errors=[]; identity=None
    for path in sorted(Path(runs_dir).iterdir()):
        if not path.is_dir() or not (path/'metadata.json').exists():
            continue
        config=json.loads((path/'config.json').read_text(encoding='utf-8'))
        if config.get('smoke'):
            continue
        meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        metric=json.loads((path/'metrics.json').read_text(encoding='utf-8'))
        status=json.loads((path/'status.json').read_text())
        verify=json.loads((path/'fresh_process_verification.json').read_text())
        if status['status']!='complete' or not verify['passed']:
            raise ValueError(f'Incomplete/unverified run {path}')
        common=(meta['data_provenance']['data_sha256'],meta['data_provenance']['source_stats_sha256'],
                meta['split_id'],config['epochs'],config['batch_size'],config['patience'])
        if identity is None:
            identity=common
        if common!=identity:
            raise ValueError('Incompatible data/split/training budget; separate comparisons required')
        pred=read_rows(path/'validation_predictions.csv')
        y=np.array([int(r['true_label']) for r in pred])
        subject=np.array([int(r['subject']) for r in pred])
        logits=np.array([[float(r[f'logit_{i}']) for i in range(6)] for r in pred])
        predicted=logits.argmax(1)
        current=classification_metrics(y,logits)
        for key in ('accuracy','macro_f1'):
            np.testing.assert_allclose(current[key],metric[key],rtol=0,atol=1e-12)
        row=dict(run_id=path.name,experiment=config['experiment_id'],model=config['model_key'],seed=config['seed'],
                 accuracy=current['accuracy'],macro_f1=current['macro_f1'],parameters=metric['parameters'],
                 best_epoch=metric['best_epoch'],epochs_ran=metric['epochs_ran'],train_seconds=metric['train_seconds'])
        runs.append(row)
        p,r,f,n=precision_recall_fscore_support(y,predicted,labels=range(6),zero_division=0)
        for i in range(6):
            classes.append(dict(run_id=path.name,class_name=CLASSES[i],precision=p[i],recall=r[i],f1=f[i],support=int(n[i])))
        for sid in np.unique(subject):
            mask=subject==sid
            m=classification_metrics(y[mask],logits[mask])
            people.append(dict(run_id=path.name,subject=int(sid),support=int(mask.sum()),accuracy=m['accuracy'],macro_f1=m['macro_f1']))
        for i in np.flatnonzero(y!=predicted):
            errors.append(dict(run_id=path.name,sample_id=pred[i]['sample_id'],subject=int(subject[i]),
                              true_label=int(y[i]),predicted_label=int(predicted[i]),
                              confidence=max(float(pred[i][f'probability_{j}']) for j in range(6))))
        history=read_rows(path/'history.csv')
        fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
        epoch=np.arange(1,len(history)+1)
        for key,label in [('loss','fit'),('val_loss','validation')]:
            axes[0].plot(epoch,[float(r[key]) for r in history],label=label)
        axes[0].axvline(metric['best_epoch'],color='gray',linestyle=':',label='best val loss')
        axes[0].set(xlabel='Epoch',ylabel='Cross entropy',title=path.name)
        axes[0].legend()
        axes[1].plot(epoch,[float(r['accuracy']) for r in history],label='fit accuracy')
        axes[1].plot(epoch,[float(r['val_accuracy']) for r in history],label='val accuracy')
        axes[1].plot(epoch,[float(r['val_macro_f1']) for r in history],label='val macro F1')
        axes[1].set(xlabel='Epoch',ylabel='Score',ylim=(0,1))
        axes[1].legend()
        fig.savefig(figs/f'{path.name}_history.png',dpi=160)
        plt.close(fig)
        cm=np.asarray(current['confusion_matrix'])
        fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
        im=ax.imshow(cm,cmap='Blues')
        for i in range(6):
            for j in range(6):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        labels=['Walk','Upstairs','Downstairs','Sit','Stand','Lay']
        ax.set(xticks=range(6),yticks=range(6),xticklabels=labels,yticklabels=labels,
               xlabel='Predicted class',ylabel='True class',title=f'{path.name} validation N={len(y)}')
        plt.setp(ax.get_xticklabels(),rotation=35,ha='right')
        fig.colorbar(im,ax=ax,label='Windows')
        fig.savefig(figs/f'{path.name}_confusion.png',dpi=160)
        plt.close(fig)
        write_json(out/f'{path.name}_confusion.json',dict(counts=cm.tolist(),row_normalized=(cm/np.maximum(cm.sum(1,keepdims=True),1)).tolist()))
    summary=[]
    for exp in sorted(set(r['experiment'] for r in runs)):
        selected=[r for r in runs if r['experiment']==exp]
        if sorted(r['seed'] for r in selected)!=[2026,2027,2028]:
            raise ValueError(f'{exp} requires all three agreed seeds')
        summary.append(dict(experiment=exp,model=selected[0]['model'],n=3,parameters=selected[0]['parameters'],
            accuracy_mean=float(np.mean([r['accuracy'] for r in selected])),
            accuracy_sd=float(np.std([r['accuracy'] for r in selected],ddof=1)),
            macro_f1_mean=float(np.mean([r['macro_f1'] for r in selected])),
            macro_f1_sd=float(np.std([r['macro_f1'] for r in selected],ddof=1))))
    # Freeze the candidate set before training: R01/R02/R03 only, same seed budget.
    candidates=[r for r in summary if r['experiment'] in ('R01','R02','R03')]
    chosen=sorted(candidates,key=lambda r:(-r['macro_f1_mean'],r['parameters'],r['experiment']))[0]
    result=dict(runs=runs,summary=summary,selected_rnn=chosen['experiment'],
                selection_rule='highest three-seed validation macro F1 mean; ties: fewer parameters then ID',
                test_evaluated=False,identity=list(identity),sd_scope='sample SD across training seeds on one fixed subject split')
    write_json(out/'analysis.json',result)
    baseline=[r for r in summary if r['experiment'] in ('M01','C01','R01','L01')]
    write_rows(out/'four_model_comparison.csv',baseline)
    paired=[]
    for exp in ('M01','C01','L01'):
        for seed in (2026,2027,2028):
            a=next((r for r in runs if r['experiment']==exp and r['seed']==seed),None)
            b=next((r for r in runs if r['experiment']=='R01' and r['seed']==seed),None)
            if a and b:
                paired.append(dict(comparator=exp,seed=seed,macro_f1_difference_comparator_minus_rnn=a['macro_f1']-b['macro_f1']))
    write_rows(out/'paired_differences.csv',paired)
    for name,rows in [('runs',runs),('summary',summary),('class_metrics',classes),('subject_metrics',people),('errors',errors)]:
        write_rows(out/f'{name}.csv',rows)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    ax.bar([r['experiment'] for r in summary],[r['macro_f1_mean'] for r in summary],
           yerr=[r['macro_f1_sd'] for r in summary],capsize=5,color='#31688e')
    ax.set(ylabel='Validation macro F1',xlabel='Configuration',ylim=(0,1),title='Three seeds, fixed S0 split; error bars = sample SD')
    fig.savefig(figs/'validation_comparison.png',dpi=180)
    plt.close(fig)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--runs-dir',type=Path,default=Path('runs/rnn'))
    p.add_argument('--out',type=Path,default=Path('reports/rnn'))
    args=p.parse_args()
    analyze(args.runs_dir,args.out)
