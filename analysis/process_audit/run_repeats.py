"""Prespecified 7-condition x 3-seed experiment; never evaluate official test."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[2]
IDS = ['E02','E03','E07','E08','E09','E10','E11']
SEEDS = [2026,2027,2028]
OUT = ROOT/'reports/process_audit'
RUNS = ROOT/'runs/process_repeats'
DATA = ROOT.parents[1]/'work/uci-har/UCI HAR Dataset'
if len(sys.argv)>1: DATA=Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True,exist_ok=True)
LOGS = ROOT.parents[1]/'work/process-build/logs'
LOGS.mkdir(parents=True,exist_ok=True)
started = datetime.now(timezone.utc).isoformat()
records=[]
for seed in SEEDS:
    for exp in IDS:
        env=dict(os.environ,TF_ENABLE_ONEDNN_OPTS='0',TF_CPP_MIN_LOG_LEVEL='2',PYTHONHASHSEED=str(seed))
        command=[sys.executable,'-m','src.train','--config',f'configs/{exp}.json',
                 '--data-root',str(DATA),'--runs-dir',str(RUNS),'--seed',str(seed)]
        run_id=f'{exp}_seed{seed}'
        if (RUNS/run_id).exists():
            raise FileExistsError(f'Use an empty run directory: {RUNS/run_id}')
        print(f'Start {run_id}',flush=True)
        with (LOGS/f'{run_id}.log').open('w',encoding='utf8') as log:
            process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record={'run_id':run_id,'seed':seed,'experiment':exp,'returncode':process.returncode,
                'finished_utc':datetime.now(timezone.utc).isoformat()}
        records.append(record)
        (OUT/'repeat_execution.json').write_text(json.dumps({'started_utc':started,'prespecified_seeds':SEEDS,
            'prespecified_experiments':IDS,'official_test_evaluated':False,'runs':records},indent=2)+'\n',encoding='utf8')
        if process.returncode: raise RuntimeError(f'{run_id} failed; see {LOGS/run_id}.log')
        metrics=json.loads((RUNS/run_id/'metrics.json').read_text())
        print(f'Complete {run_id}: F1={metrics["macro_f1"]:.6f}',flush=True)
