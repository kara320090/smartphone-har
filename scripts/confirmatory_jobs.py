"""Fresh-process jobs; training and cost benchmarks must remain sequential."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument('phase',choices=['train','verify-dev','benchmark','test','replay'])
ap.add_argument('--workers',type=int,default=1,choices=[1,2,3])
ap.add_argument('--log-dir',type=Path,default=ROOT/'runs/confirmatory_logs')
a=ap.parse_args()
if a.phase in ('train','benchmark') and a.workers!=1:raise ValueError('Timing/training jobs must be sequential')
rep=ROOT/'reports/confirmatory_20261019'
logs=a.log_dir;logs.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/'configs/paper_confirmatory_20261019.json').read_text())
if a.phase=='train':jobs=[('train','--condition',c,'--seed',str(s)) for c in cfg['conditions'] for s in cfg['seeds']]
else:jobs=[(a.phase,'--run',e['id']) for e in json.loads((rep/'inventory.json').read_text())]

def run(args):
    tag='_'.join(v for v in args if not v.startswith('--'));log=logs/(tag+'.log')
    if log.exists():raise FileExistsError(log)
    print('START',tag,flush=True)
    with log.open('w',encoding='utf-8') as f:
        result=subprocess.run([sys.executable,'-u','tools/confirmatory_har.py',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text(encoding='utf-8')[-5000:],flush=True)
        raise RuntimeError(f'{tag} failed; log preserved')
    print('COMPLETE',tag,flush=True)

with ThreadPoolExecutor(max_workers=a.workers) as pool:
    list(pool.map(run,jobs))
print('PHASE COMPLETE',a.phase,flush=True)
