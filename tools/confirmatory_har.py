"""Development controls, frozen evaluation and measured inference for paper v10.

No test array can be read until freeze() has checked development evidence.
Every formal fit and benchmark is run in a fresh child process by the driver.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, platform, re, statistics, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import psutil
import tensorflow as tf
from tensorflow import keras
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from paper_controls import features, metrics, sha, read_csv, write_csv, write_json, predict_saved
from src.adapters.processed_npz import load_processed_data, TeamSequencePreprocessor, EXPECTED_DATA_SHA256, EXPECTED_STATS_SHA256
from src.adapters.uci_reference import CHANNELS
from src.contracts import row_binding_digest
from src.models import build_model
from src.runtime import configure_runtime
from src.train import make_dataset, predict_batches, FiniteLoss

CFG_PATH = ROOT / 'configs/paper_confirmatory_20261019.json'
CFG = json.loads(CFG_PATH.read_text(encoding='utf-8'))
GROUP = np.array(CFG['mapping'], dtype=np.int64)
REP = ROOT / 'reports/confirmatory_20261019'
RUN = ROOT / 'runs/confirmatory_20261019'
PROTOCOL = ROOT / 'docs/paper-confirmatory-20261019/PROTOCOL_KO.md'
DATA = Path(os.environ.get('HAR_PROCESSED_DATA', 'C:/Users/soma/Desktop/processed_data.npz'))
STATS = ROOT / 'data/preprocess.npz'
OLDLR = ROOT / 'reports/paper_controls_20261019'

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(1, keepdims=True)
    q = np.exp(x)
    return q / q.sum(1, keepdims=True)

def dev():
    return load_processed_data(DATA, STATS, expected_data_sha256=EXPECTED_DATA_SHA256,
                               expected_stats_sha256=EXPECTED_STATS_SHA256)

def scaler():
    m = json.loads((OLDLR/'LR3_model.json').read_text(encoding='utf-8'))
    return np.array(m['feature_mean']), np.array(m['feature_scale'])

def feature_input(sequence):
    mean, scale = scaler()
    return ((features(sequence) - mean) / scale).astype(np.float32)

def build(condition, seed):
    if condition == 'FMLP3':
        return keras.Sequential([keras.Input((18,)), keras.layers.Dense(128, activation='relu'),
                                 keras.layers.Dense(64, activation='relu'), keras.layers.Dense(3)])
    source = {'MLP3':'M01','CNN3':'C01','RNN3':'R01','LSTM3':'L01'}[condition]
    config = json.loads((ROOT/f'configs/{source}.json').read_text(encoding='utf-8'))
    config['seed'] = seed
    model = build_model((128,9), config)
    model.pop()
    model.add(keras.layers.Dense(3, name='direct_three_logits'))
    return model

def save_predictions(path, ids, subjects, y6, prob, classes):
    rows = []
    pred = prob.argmax(1)
    for i in range(len(pred)):
        rows.append(dict(sample_id=str(ids[i]), subject=int(subjects[i]), true_six=int(y6[i]),
                         true_target=int(y6[i] if classes==6 else GROUP[y6[i]]),
                         prediction=int(pred[i]), **{f'p{c}':float(prob[i,c]) for c in range(classes)}))
    write_csv(path, rows)

def environment():
    return dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                tensorflow=tf.__version__, keras=keras.__version__, numpy=np.__version__,
                cpu_count=psutil.cpu_count(), intra_op_threads=4, inter_op_threads=1,
                devices=[str(x) for x in tf.config.list_physical_devices()])

def train(condition, seed):
    if (REP/'FINAL_LOCK.json').exists():
        raise RuntimeError('Development is locked; no further training')
    configure_runtime(seed,4)
    d=dev()
    out=RUN/f'{condition}_seed{seed}'
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'status.json',dict(status='running',started_unix=time.time()))
    write_json(out/'config.json',dict(CFG,condition=condition,seed=seed,protocol_sha256=sha(PROTOCOL),
                                     code_sha256=sha(__file__),data=d.provenance,environment=environment()))
    x,v=(feature_input(d.x_fit),feature_input(d.x_validation)) if condition=='FMLP3' else (d.x_fit,d.x_validation)
    y,vy=GROUP[d.y_fit],GROUP[d.y_validation]
    model=build(condition,seed)
    model.save(out/'initial.keras')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=CFG['learning_rate']),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
    start=time.perf_counter()
    history=model.fit(make_dataset(x,y,CFG['batch_size'],True,seed),
                      validation_data=make_dataset(v,vy,CFG['batch_size'],False,seed),
                      epochs=CFG['epochs'],verbose=2,
                      callbacks=[FiniteLoss(), keras.callbacks.CSVLogger(str(out/'history.csv')),
                                 keras.callbacks.EarlyStopping(monitor='val_loss',patience=CFG['patience'],
                                     min_delta=0,mode='min',restore_best_weights=True)])
    seconds=time.perf_counter()-start
    prob=softmax(predict_batches(model,v))
    # Save an inference-only model, not stale optimizer state after best-weight restoration.
    inference=keras.models.clone_model(model)
    inference.set_weights(model.get_weights())
    inference.save(out/'model.keras')
    save_predictions(out/'validation_predictions.csv',d.validation_ids,d.validation_subjects,d.y_validation,prob,3)
    m=metrics(vy,prob.argmax(1),3)
    m.update(condition=condition,seed=seed,parameters=int(model.count_params()),
             train_seconds=seconds,epochs_ran=len(history.history['loss']),
             best_epoch=int(np.argmin(history.history['val_loss'])+1),test_evaluated=False,
             model_sha256=sha(out/'model.keras'))
    write_json(out/'metrics.json',m)
    write_json(out/'status.json',dict(status='complete',finished_unix=time.time()))
    print(json.dumps({k:m[k] for k in ['condition','seed','macro_f1','best_epoch','train_seconds']}),flush=True)

def inventory():
    items=[]
    for c in CFG['conditions']:
        for seed in CFG['seeds']:
            folder=RUN/f'{c}_seed{seed}'
            assert json.loads((folder/'status.json').read_text())['status']=='complete'
            items.append(dict(id=folder.name,condition=c,seed=seed,kind='keras',classes=3,
                              input='features18' if c=='FMLP3' else 'sequence',
                              model=(folder/'model.keras').relative_to(ROOT).as_posix(),
                              validation=(folder/'validation_predictions.csv').relative_to(ROOT).as_posix()))
    for c,folder in [('M01','reports/mlp_yoonseop/run_evidence'),('C01','artifacts/cnn_tensorflow'),
                     ('R01','reports/rnn/run_evidence'),('L01','runs/lstm')]:
        for seed in CFG['seeds']:
            record=ROOT/folder/f'{c}_seed{seed}'
            model=record/'model.keras'
            if not model.exists():model=ROOT/'runs/confirmed_sources'/f'{c}_seed{seed}'/'model.keras'
            md=json.loads((record/'metadata.json').read_text(encoding='utf-8'))
            assert sha(model)==md['model_sha256']
            items.append(dict(id=f'{c}_seed{seed}',condition=c,seed=seed,kind='keras',classes=6,input='sequence',
                              model=model.relative_to(ROOT).as_posix(),
                              validation=(record/'validation_predictions.csv').relative_to(ROOT).as_posix()))
    for c,k in [('LR3',3),('LR6',6)]:
        items.append(dict(id=c,condition=c,seed=None,kind='linear',classes=k,input='features18',
                          model=(OLDLR/f'{c}_model.json').relative_to(ROOT).as_posix(),
                          validation=(OLDLR/f'{c}_predictions.csv').relative_to(ROOT).as_posix()))
    REP.mkdir(parents=True,exist_ok=True)
    write_json(REP/'inventory.json',items)
    return items

def items():return json.loads((REP/'inventory.json').read_text(encoding='utf-8'))
def item(runid):return next(e for e in items() if e['id']==runid)

def predictor(e):
    pre=TeamSequencePreprocessor.load(STATS)
    fmean,fscale=scaler()
    if e['kind']=='linear':
        m=json.loads((ROOT/e['model']).read_text(encoding='utf-8'))
        coef,bias=np.array(m['coef']),np.array(m['intercept'])
        def prepare(raw):return (features(pre.transform(raw))-fmean)/fscale
        def call(x):return softmax(x@coef.T+bias)
        nparams=coef.size+bias.size
        weightbytes=coef.nbytes+bias.nbytes
    else:
        m=keras.models.load_model(ROOT/e['model'],compile=False)
        signature=tf.TensorSpec((None,18) if e['input']=='features18' else (None,128,9),tf.float32)
        @tf.function(input_signature=[signature])
        def compiled(x):return tf.nn.softmax(m(x,training=False))
        def prepare(raw):
            x=pre.transform(raw)
            if e['input']=='features18':x=((features(x)-fmean)/fscale).astype(np.float32)
            return tf.convert_to_tensor(x)
        def call(x):return compiled(x).numpy()
        nparams=m.count_params()
        weightbytes=sum(w.numpy().nbytes for w in m.weights)
    def full(raw):
        prob=call(prepare(raw))
        return prob,prob.argmax(1)
    statbytes=pre.mean.nbytes+pre.scale.nbytes
    if e['input']=='features18':statbytes+=fmean.nbytes+fscale.nbytes
    return prepare,call,full,dict(parameters=int(nparams),weight_array_bytes=int(weightbytes),
                                preprocessing_array_bytes=int(statbytes),
                                stored_model_bytes=(ROOT/e['model']).stat().st_size)

def verify_development(runid):
    configure_runtime(2026,4)
    e=item(runid);d=dev();prepare,call,full,size=predictor(e)
    q=np.concatenate([full(d.raw_validation[i:i+256])[0] for i in range(0,len(d.y_validation),256)])
    saved=read_csv(ROOT/e['validation'])
    assert [r['sample_id'] for r in saved]==list(d.validation_ids)
    label_key='true_six' if 'true_six' in saved[0] else 'true_label'
    assert np.array_equal([int(r[label_key]) for r in saved],d.y_validation)
    assert np.array_equal([int(r['subject']) for r in saved],d.validation_subjects)
    pkey='prediction' if 'prediction' in saved[0] else 'predicted_label'
    prefix='p' if 'p0' in saved[0] else 'probability_'
    target=np.array([[float(r[f'{prefix}{c}']) for c in range(e['classes'])] for r in saved])
    np.testing.assert_allclose(q,target,rtol=1e-5,atol=1e-6)
    np.testing.assert_array_equal(q.argmax(1),[int(r[pkey]) for r in saved])
    dest=REP/'development';dest.mkdir(exist_ok=True)
    m=metrics(d.y_validation if e['classes']==6 else GROUP[d.y_validation],q.argmax(1),e['classes'])
    write_json(dest/f'{runid}.json',dict(passed=True,run=e,windows=len(q),metrics=m,size=size,
                                       probability_max_abs_diff=float(np.max(abs(q-target))),test_arrays_loaded=False))
    print('development verified',runid,flush=True)

def benchmark(runid):
    configure_runtime(2026,4)
    e=item(runid);d=dev();prepare,call,full,size=predictor(e)
    raw=d.raw_validation[:1].copy()
    del d
    prepared=prepare(raw)
    for _ in range(CFG['benchmark_warmup']):
        full(raw)
        q=call(prepared);q.argmax(1)
    process=psutil.Process();rss_start=process.memory_info().rss;rss_peak=rss_start
    rows=[]
    # Alternate scope order, deterministic, to reduce monotonic drift bias.
    for i in range(CFG['benchmark_repetitions']):
        scopes=('prepared_call','window_to_output') if i%2==0 else ('window_to_output','prepared_call')
        for scope in scopes:
            start=time.perf_counter_ns()
            if scope=='prepared_call':q=call(prepared);p=q.argmax(1)
            else:q,p=full(raw)
            elapsed=(time.perf_counter_ns()-start)/1e6
            assert np.isfinite(q).all() and len(p)==1
            rows.append(dict(run_id=runid,iteration=i,scope=scope,milliseconds=elapsed))
        rss_peak=max(rss_peak,process.memory_info().rss)
    out=REP/'benchmark';out.mkdir(exist_ok=True)
    write_csv(out/f'{runid}_samples.csv',rows)
    summary={}
    for scope in ('prepared_call','window_to_output'):
        values=[r['milliseconds'] for r in rows if r['scope']==scope]
        summary[scope]=dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)))
    write_json(out/f'{runid}.json',dict(run=e,summary=summary,size=size,rss_start_bytes=rss_start,
                  sampled_peak_rss_bytes=rss_peak,rss_scope='entire fresh process; not model-only or allocator peak',
                  environment=environment(),warmup=CFG['benchmark_warmup'],repetitions=CFG['benchmark_repetitions'],
                  batch=1,test_arrays_loaded=False,finished_unix=time.time()))
    print('benchmark',runid,summary['window_to_output'],flush=True)

def code_files():
    return [Path(__file__),ROOT/'tools/paper_controls.py',ROOT/'scripts/confirmatory_jobs.py',PROTOCOL,
            *sorted((ROOT/'configs').glob('*.json')),*sorted((ROOT/'src').rglob('*.py'))]

def freeze():
    if (REP/'FINAL_LOCK.json').exists():raise FileExistsError('Lock already exists')
    files=set(code_files()+[STATS,OLDLR/'LR3_model.json',OLDLR/'LR6_model.json',REP/'inventory.json'])
    files.update((REP/'training_sources').glob('*'))
    for e in items():
        for rel in [e['model'],e['validation']]:files.add(ROOT/rel)
        for name in [f"development/{e['id']}.json",f"benchmark/{e['id']}.json",f"benchmark/{e['id']}_samples.csv"]:
            p=REP/name;assert p.exists();files.add(p)
        assert json.loads((REP/f"development/{e['id']}.json").read_text())['passed']
    for p in RUN.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.csv','.keras'):files.add(p)
    lock=dict(frozen_unix=time.time(),test_evaluated_at_freeze=False,test_arrays_loaded_at_freeze=False,
              no_fit_val_refit=True,official_test_authorized_by_user=True,items=items(),
              data_sha256=EXPECTED_DATA_SHA256,stats_sha256=EXPECTED_STATS_SHA256,
              files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)})
    write_json(REP/'FINAL_LOCK.json',lock)
    print('Frozen',len(lock['items']),'models',len(files),'files',flush=True)

def verify_lock():
    lock=json.loads((REP/'FINAL_LOCK.json').read_text(encoding='utf-8'))
    if lock['test_evaluated_at_freeze'] or not lock['official_test_authorized_by_user']:
        raise ValueError('Invalid lock')
    for path,digest in lock['files'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Frozen file changed: '+path)
    if sha(DATA)!=lock['data_sha256']:raise ValueError('Data changed')
    return lock

def load_test():
    verify_lock()
    with np.load(DATA,allow_pickle=False) as z:
        a={k:z['test_'+k] for k in ('Xs','y','subject','sample_id')}
        dev_people=set(z['fit_subject'].tolist())|set(z['val_subject'].tolist())
    x,y,s,ids=(a[k] for k in ('Xs','y','subject','sample_id'))
    assert x.shape==(2947,128,9) and x.dtype==np.float32 and np.isfinite(x).all()
    assert y.shape==s.shape==ids.shape==(2947,)
    assert y.dtype==s.dtype==np.int64 and set(y)==set(range(6))
    assert len(set(s))==9 and not (set(s)&dev_people)
    assert list(ids)==[f'test:{i:06d}' for i in range(2947)]
    return x,y,s,ids

def validate_official_source():
    x,y,s,ids=load_test()
    original=Path(os.environ.get('HAR_OFFICIAL_ROOT', str(ROOT.parents[1]/'work/uci-har/UCI HAR Dataset')))
    files=[]
    for split in ('train','test'):
        labels=np.loadtxt(original/split/f'y_{split}.txt',dtype=np.int64)-1
        subjects=np.loadtxt(original/split/f'subject_{split}.txt',dtype=np.int64)
        sequence=np.stack([np.loadtxt(original/split/'Inertial Signals'/f'{c}_{split}.txt',dtype=np.float32) for c in CHANNELS],axis=-1)
        for p in [original/split/f'y_{split}.txt',original/split/f'subject_{split}.txt',*[original/split/'Inertial Signals'/f'{c}_{split}.txt' for c in CHANNELS]]:
            files.append(dict(file=p.relative_to(original).as_posix(),sha256=sha(p)))
        if split=='test':
            np.testing.assert_array_equal(x,sequence);np.testing.assert_array_equal(y,labels);np.testing.assert_array_equal(s,subjects)
        else:
            with np.load(DATA,allow_pickle=False) as z:
                for part in ('fit','val'):
                    indexes=np.array([int(i.split(':')[1]) for i in z[part+'_sample_id']])
                    np.testing.assert_array_equal(z[part+'_Xs'],sequence[indexes])
                    np.testing.assert_array_equal(z[part+'_y'],labels[indexes])
                    np.testing.assert_array_equal(z[part+'_subject'],subjects[indexes])
    write_json(REP/'official_source_check.json',dict(passed=True,count=len(y),subjects=sorted(map(int,set(s))),
                    raw_row_binding_sha256=row_binding_digest(x,y,ids,s),files=files,finished_unix=time.time()))

def evaluate(runid,verify_only=False):
    configure_runtime(2026,4)
    x,y,s,ids=load_test();e=item(runid);prepare,call,full,size=predictor(e)
    q=np.concatenate([full(x[i:i+256])[0] for i in range(0,len(y),256)])
    assert q.shape==(2947,e['classes']) and np.isfinite(q).all()
    np.testing.assert_allclose(q.sum(1),1,atol=1e-6)
    out=REP/'test'/runid
    if verify_only:
        saved=read_csv(out/'predictions.csv')
        assert [r['sample_id'] for r in saved]==list(ids)
        np.testing.assert_array_equal([int(r['subject']) for r in saved],s)
        np.testing.assert_array_equal([int(r['true_six']) for r in saved],y)
        target=np.array([[float(r[f'p{c}']) for c in range(e['classes'])] for r in saved])
        np.testing.assert_array_equal(q.argmax(1),target.argmax(1))
        np.testing.assert_allclose(q,target,rtol=1e-5,atol=1e-6)
        write_json(out/'reload.json',dict(passed=True,count=len(y),max_abs_diff=float(np.max(abs(q-target))),
                                         separate_process=True))
        print('test replay verified',runid,flush=True);return
    out.mkdir(parents=True,exist_ok=False)
    save_predictions(out/'predictions.csv',ids,s,y,q,e['classes'])
    tasks=[('six',y,q.argmax(1),6),('three',GROUP[y],GROUP[q.argmax(1)],3)] if e['classes']==6 else [('three',GROUP[y],q.argmax(1),3)]
    result=dict(run=e,size=size,lock_sha256=sha(REP/'FINAL_LOCK.json'),finished_unix=time.time(),levels={})
    subject_rows=[]
    for level,truth,p,k in tasks:
        result['levels'][level]=metrics(truth,p,k)
        for sid in sorted(set(s)):
            mask=s==sid;m=metrics(truth[mask],p[mask],k)
            subject_rows.append(dict(run_id=runid,level=level,subject=int(sid),
                                     **{key:m[key] for key in ('count','accuracy','macro_f1','errors')}))
    write_json(out/'metrics.json',result);write_csv(out/'subjects.csv',subject_rows)
    print('test complete',runid,{k:v['macro_f1'] for k,v in result['levels'].items()},flush=True)

def audit():
    verify_lock()
    x,y,s,ids=load_test();rows=[];subjects=[];bench=[]
    assert json.loads((REP/'official_source_check.json').read_text())['passed']
    for e in items():
        out=REP/'test'/e['id'];saved=read_csv(out/'predictions.csv')
        m=json.loads((out/'metrics.json').read_text());assert json.loads((out/'reload.json').read_text())['passed']
        assert [r['sample_id'] for r in saved]==list(ids)
        p=np.array([int(r['prediction']) for r in saved]);q=np.array([[float(r[f'p{c}']) for c in range(e['classes'])] for r in saved])
        np.testing.assert_array_equal(p,q.argmax(1));np.testing.assert_array_equal([int(r['true_six']) for r in saved],y)
        for level in m['levels']:
            truth=y if level=='six' else GROUP[y]
            pred=p if level=='six' or e['classes']==3 else GROUP[p]
            calc=metrics(truth,pred,6 if level=='six' else 3)
            assert abs(calc['macro_f1']-m['levels'][level]['macro_f1'])<1e-12
            assert calc['confusion_matrix']==m['levels'][level]['confusion_matrix']
            rows.append(dict(run_id=e['id'],condition=e['condition'],seed=e['seed'],level=level,
                             parameters=m['size']['parameters'],**{k:calc[k] for k in ('macro_f1','accuracy','errors','count')}))
        subjects.extend(read_csv(out/'subjects.csv'))
        b=json.loads((REP/'benchmark'/f"{e['id']}.json").read_text())
        bench.append(dict(run_id=e['id'],condition=e['condition'],seed=e['seed'],**b['size'],
                          **{scope+'_'+key:val for scope,vals in b['summary'].items() for key,val in vals.items()},
                          rss_start_bytes=b['rss_start_bytes'],sampled_peak_rss_bytes=b['sampled_peak_rss_bytes']))
    write_csv(REP/'test_runs.csv',rows);write_csv(REP/'test_subjects.csv',subjects);write_csv(REP/'benchmark_runs.csv',bench)
    summary=[]
    for c,level in dict.fromkeys((r['condition'],r['level']) for r in rows):
        a=[r for r in rows if r['condition']==c and r['level']==level]
        b=[r for r in bench if r['condition']==c]
        summary.append(dict(condition=c,level=level,runs=len(a),parameters=a[0]['parameters'],
                            f1_mean=statistics.mean(r['macro_f1'] for r in a),
                            f1_seed_sd=statistics.stdev(r['macro_f1'] for r in a) if len(a)>1 else None,
                            accuracy_mean=statistics.mean(r['accuracy'] for r in a),
                            errors_sum=sum(r['errors'] for r in a),
                            window_ms_median_across_runs=statistics.median(r['window_to_output_median_ms'] for r in b),
                            weight_and_preprocessing_bytes=b[0]['weight_array_bytes']+b[0]['preprocessing_array_bytes']))
    write_csv(REP/'summary.csv',summary)
    pairs=[]
    for base,direct in [('M01','MLP3'),('C01','CNN3'),('R01','RNN3'),('L01','LSTM3')]:
        for seed in CFG['seeds']:
            a=next(r for r in rows if r['condition']==base and r['seed']==seed and r['level']=='three')
            b=next(r for r in rows if r['condition']==direct and r['seed']==seed and r['level']=='three')
            pairs.append(dict(base=base,direct=direct,seed=seed,direct_minus_mapped_f1=b['macro_f1']-a['macro_f1']))
    write_csv(REP/'paired_differences.csv',pairs)
    write_json(REP/'AUDIT.json',dict(passed=True,models=len(items()),test_predictions=len(items())*2947,
                                   metric_recomputations=len(rows),fresh_process_replays=len(items()),
                                   test_subject_count=len(set(s)),lock_sha256=sha(REP/'FINAL_LOCK.json'),
                                   no_post_test_model_changes=True,summary=summary))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','inventory','verify-dev','benchmark','freeze','source-check','test','replay','audit'])
    ap.add_argument('--condition');ap.add_argument('--seed',type=int);ap.add_argument('--run');a=ap.parse_args()
    REP.mkdir(parents=True,exist_ok=True)
    if a.mode=='train':train(a.condition,a.seed)
    elif a.mode=='inventory':inventory()
    elif a.mode=='verify-dev':verify_development(a.run)
    elif a.mode=='benchmark':benchmark(a.run)
    elif a.mode=='freeze':freeze()
    elif a.mode=='source-check':validate_official_source()
    elif a.mode=='test':evaluate(a.run)
    elif a.mode=='replay':evaluate(a.run,True)
    else:audit()
if __name__=='__main__':main()
