"""Follow-up control after finding same-seed initial-weight differences."""
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.runtime import configure_runtime
from src.models import build_mlp
from src.train import train_model
from src.contracts import prepare_data
from src.adapters.uci_reference import load_dataset
parser=argparse.ArgumentParser()
parser.add_argument('--seed',type=int,required=True,choices=[2026,2027,2028])
parser.add_argument('--data-root',required=True,type=Path)
args=parser.parse_args()
seed=args.seed
configure_runtime(seed)
reference=build_mlp(dropout=0)
reference_weights={l.name:l.get_weights() for l in reference.layers if l.get_weights()}
configure_runtime(seed)
model=build_mlp(dropout=.3)
for name,weights in reference_weights.items(): model.get_layer(name).set_weights(weights)
def digest(values):
    h=hashlib.sha256()
    for name,weights in values.items():
        h.update(name.encode())
        for w in weights: h.update(w.tobytes())
    return h.hexdigest()
matched={l.name:l.get_weights() for l in model.layers if l.get_weights()}
assert digest(reference_weights)==digest(matched)
for name in reference_weights:
    for a,b in zip(reference_weights[name],matched[name]): np.testing.assert_array_equal(a,b)
config=json.loads((ROOT/'configs/E08.json').read_text())
config.update(seed=seed,experiment_id='E08W',run_directory=str(ROOT/'runs/process_matched'/f'E08W_seed{seed}'))
samples,indices=load_dataset(args.data_root)
data=prepare_data(samples,indices,config)
out=train_model(model,data,config)
record={'seed':seed,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'control':'E08W: Dropout 0.3 with every Dense weight/bias copied from E02 initialization',
        'all_dense_parameters_initially_identical':True,'initial_dense_weights_sha256':digest(matched),
        'plan_sha256':hashlib.sha256((ROOT/'docs/PROCESS_AUDIT_AMENDMENT_KO.md').read_bytes()).hexdigest(),
        'remaining_difference':'Dropout layers and their training masks; different training trajectories and early stopping remain expected.'}
(out/'initialization_match.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
print(out)
