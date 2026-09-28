"""Check what same-seed pairing does and does not keep identical across models."""
from datetime import datetime, timezone
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.runtime import configure_runtime
from src.models import build_mlp
records=[]
for seed in [2026,2027,2028]:
    configure_runtime(seed); baseline=build_mlp(dropout=0)
    a={l.name:l.get_weights() for l in baseline.layers if l.get_weights()}
    configure_runtime(seed); dropout=build_mlp(dropout=.3)
    b={l.name:l.get_weights() for l in dropout.layers if l.get_weights()}
    records.append({'seed':seed,'dense_layers':[{'name':key,'identical':all(np.array_equal(u,v) for u,v in zip(a[key],b[key])),
        'max_weight_difference':max(float(np.max(np.abs(u-v))) for u,v in zip(a[key],b[key]))} for key in a]})
result={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Check initial weights only; no matched-weight training experiment is claimed.', 'results':records}
(ROOT/'reports/process_audit/initialization_control.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
