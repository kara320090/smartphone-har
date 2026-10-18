"""Fixed simple baselines and exhaustive grouping controls for development data."""
from pathlib import Path
import argparse,csv,hashlib,itertools,json,os,platform,statistics,sys,time,warnings
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,f1_score,classification_report
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.adapters.processed_npz import load_processed_data,EXPECTED_DATA_SHA256,EXPECTED_STATS_SHA256
GROUP=np.array([0,0,0,1,1,2])
PARAMS=dict(C=1.0,solver='lbfgs',max_iter=2000,tol=1e-8,fit_intercept=True,class_weight=None)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def read_csv(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def features(x):
    x=np.asarray(x,dtype=np.float64)
    assert x.ndim==3 and x.shape[1:]==(128,9) and np.isfinite(x).all()
    return np.concatenate([x.mean(axis=1),x.std(axis=1,ddof=0)],axis=1)
def metrics(y,p,k):
    cm=confusion_matrix(y,p,labels=range(k));den=cm.sum(0)+cm.sum(1)
    own=np.divide(2*np.diag(cm),den,out=np.zeros(k,dtype=float),where=den>0).mean()
    ref=f1_score(y,p,labels=range(k),average='macro',zero_division=0)
    assert abs(own-ref)<1e-12
    return dict(macro_f1=float(own),accuracy=float(np.trace(cm)/cm.sum()),errors=int(cm.sum()-np.trace(cm)),count=len(y),confusion_matrix=cm.tolist(),class_metrics=classification_report(y,p,labels=range(k),output_dict=True,zero_division=0))
def predict_saved(model,x):
    x=(features(x)-np.array(model['feature_mean']))/np.array(model['feature_scale'])
    z=x@np.array(model['coef']).T+np.array(model['intercept'])
    z-=z.max(axis=1,keepdims=True);q=np.exp(z);q/=q.sum(axis=1,keepdims=True)
    return q.argmax(axis=1),q

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--processed-data',type=Path,required=True)
    ap.add_argument('--preprocess-stats',type=Path,default=ROOT/'data/preprocess.npz')
    ap.add_argument('--output',type=Path,default=ROOT/'reports/paper_controls_20261019')
    ap.add_argument('--verify-only',action='store_true')
    a=ap.parse_args();out=a.output
    if not a.verify_only and out.exists() and any(out.iterdir()):raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    d=load_processed_data(a.processed_data,a.preprocess_stats,expected_data_sha256=EXPECTED_DATA_SHA256,expected_stats_sha256=EXPECTED_STATS_SHA256)
    assert d.x_fit.shape==(5577,128,9) and d.x_validation.shape==(1775,128,9)
    assert not d.provenance['test_arrays_loaded']
    if a.verify_only:
        checked=[]
        for name in ['LR6','LR3']:
            model=json.loads((out/f'{name}_model.json').read_text(encoding='utf-8'))
            p,q=predict_saved(model,d.x_validation)
            saved=read_csv(out/f'{name}_predictions.csv')
            assert [r['sample_id'] for r in saved]==list(d.validation_ids)
            assert np.array_equal(p,[int(r['prediction']) for r in saved])
            target=np.array([[float(r[f'p{i}']) for i in range(q.shape[1])] for r in saved])
            np.testing.assert_allclose(q,target,rtol=1e-10,atol=1e-12)
            checked.append({'condition':name,'windows':1775,'max_abs_probability_difference':float(np.max(np.abs(q-target)))})
        write_json(out/'reload_verification.json',{'passed':True,'separate_process':True,'checks':checked,'test_arrays_loaded':False})
        print(json.dumps({'reload_passed':True,'checks':checked}));return
    protocol=ROOT/'docs/paper-controls-20261019/PROTOCOL_KO.md'
    write_json(out/'execution_start.json',{'protocol_sha256':sha(protocol),'script_sha256':sha(__file__),'params':PARAMS,'penalty':'L2 default for lbfgs','data_provenance':d.provenance,'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'sklearn':sklearn.__version__,'started_unix':time.time(),'post_hoc_design':True})
    scaler=StandardScaler().fit(features(d.x_fit))
    x=scaler.transform(features(d.x_fit));v=scaler.transform(features(d.x_validation))
    summary=[];by_subject=[]
    with threadpool_limits(limits=4):
        for name,k in [('LR6',6),('LR3',3)]:
            y=d.y_fit if k==6 else GROUP[d.y_fit]
            model=LogisticRegression(**PARAMS)
            start=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(x,y)
            seconds=time.perf_counter()-start
            write_json(out/f'{name}_warnings.json',[str(w.message) for w in caught])
            assert not any(issubclass(w.category,ConvergenceWarning) for w in caught),'Optimizer did not converge'
            assert int(model.n_iter_[0])<PARAMS['max_iter']
            pred=model.predict(v);prob=model.predict_proba(v)
            record={'condition':name,'input_features':18,'feature_order':'channel means 0..8 then ddof=0 standard deviations 0..8','feature_mean':scaler.mean_.tolist(),'feature_scale':scaler.scale_.tolist(),'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'classes':model.classes_.tolist(),'params':PARAMS,'classifier_parameters':int(model.coef_.size+model.intercept_.size),'feature_scaler_constants':36,'team_sequence_scaler_constants':18,'iterations':int(model.n_iter_[0]),'fit_seconds':seconds,'training_subjects':sorted(set(map(int,d.fit_subjects))),'validation_subjects':sorted(set(map(int,d.validation_subjects))),'six_to_three_mapping':GROUP.tolist(),'test_evaluated':False}
            write_json(out/f'{name}_model.json',record)
            saved=[]
            for i in range(len(pred)):
                saved.append({'sample_id':str(d.validation_ids[i]),'subject':int(d.validation_subjects[i]),'true_six':int(d.y_validation[i]),'true_target':int(d.y_validation[i] if k==6 else GROUP[d.y_validation[i]]),'prediction':int(pred[i]),**{f'p{c}':float(prob[i,c]) for c in range(k)}})
            write_csv(out/f'{name}_predictions.csv',saved)
            tasks=[('six',d.y_validation,pred,6),('three',GROUP[d.y_validation],GROUP[pred],3)] if k==6 else [('three',GROUP[d.y_validation],pred,3)]
            for level,truth,p,n in tasks:
                m=metrics(truth,p,n)
                write_json(out/f'{name}_{level}_metrics.json',m)
                summary.append({'model':name,'level':level,'classifier_parameters':record['classifier_parameters'],'fits':1,'macro_f1':m['macro_f1'],'accuracy':m['accuracy'],'errors':m['errors'],'count':m['count']})
                for sid in sorted(set(d.validation_subjects)):
                    mask=d.validation_subjects==sid;q=metrics(truth[mask],p[mask],n)
                    by_subject.append({'model':name,'level':level,'subject':int(sid),**{key:q[key] for key in ['count','macro_f1','accuracy','errors']}})
    write_csv(out/'linear_summary.csv',summary);write_csv(out/'linear_subjects.csv',by_subject)
    maps=[]
    for singleton in range(6):
        for pair in itertools.combinations([i for i in range(6) if i!=singleton],2):
            m=np.zeros(6,dtype=int);m[list(pair)]=1;m[singleton]=2;maps.append(m)
    assert len(maps)==60 and len({tuple(m) for m in maps})==60
    semantic=next(i for i,m in enumerate(maps) if np.array_equal(m,GROUP))
    rows=[];sources=[]
    for c,folder in [('M01','reports/mlp_yoonseop/run_evidence'),('C01','artifacts/cnn_tensorflow'),('R01','reports/rnn/run_evidence'),('L01','runs/lstm')]:
        for seed in (2026,2027,2028):
            path=ROOT/folder/f'{c}_seed{seed}/validation_predictions.csv';saved=read_csv(path)
            assert [r['sample_id'] for r in saved]==list(d.validation_ids)
            y=np.array([int(r['true_label']) for r in saved]);p=np.array([int(r['predicted_label']) for r in saved])
            assert np.array_equal(y,d.y_validation) and np.array_equal([int(r['subject']) for r in saved],d.validation_subjects)
            sources.append({'file':path.relative_to(ROOT).as_posix(),'sha256':sha(path)})
            for i,m in enumerate(maps):
                q=metrics(m[y],m[p],3)
                rows.append({'config':c,'seed':seed,'mapping_id':i,'mapping':','.join(map(str,m)),'semantic':i==semantic,'macro_f1':q['macro_f1'],'errors':q['errors']})
    write_csv(out/'all_mapping_runs.csv',rows)
    grouped=[];rank_summary=[]
    for c in ['M01','C01','R01','L01']:
        means=[]
        for i,m in enumerate(maps):
            group=[r for r in rows if r['config']==c and r['mapping_id']==i]
            mean=statistics.mean(r['macro_f1'] for r in group);means.append(mean)
            grouped.append({'config':c,'mapping_id':i,'mapping':','.join(map(str,m)),'semantic':i==semantic,'macro_f1_mean':mean,'macro_f1_seed_sd':statistics.stdev(r['macro_f1'] for r in group),'errors_sum':sum(r['errors'] for r in group)})
        score=means[semantic]
        rank_summary.append({'config':c,'semantic_f1':score,'semantic_rank':1+sum(v>score+1e-12 for v in means),'ties':sum(abs(v-score)<1e-12 for v in means),'mapping_min':min(means),'mapping_median':statistics.median(means),'mapping_max':max(means),'alternative_median':statistics.median(v for i,v in enumerate(means) if i!=semantic)})
    write_csv(out/'all_mapping_summary.csv',grouped);write_csv(out/'mapping_rank_summary.csv',rank_summary)
    write_json(out/'verification.json',{'passed':True,'linear_fits':2,'linear_seed_repetitions':False,'linear_validation_rows':3550,'grouping_metric_checks':len(rows),'grouping_maps':60,'semantic_mapping_id':semantic,'prediction_sources':sources,'post_hoc':True,'new_neural_training':False,'test_arrays_loaded':False,'test_evaluated':False,'no_validation_hyperparameter_search':True,'source_data_sha256':EXPECTED_DATA_SHA256,'source_stats_sha256':EXPECTED_STATS_SHA256,'protocol_sha256':sha(protocol),'code_sha256':sha(__file__),'summary':summary,'mapping_ranks':rank_summary})
    print(json.dumps({'linear':summary,'mapping_controls':rank_summary},ensure_ascii=True))
if __name__=='__main__':main()
