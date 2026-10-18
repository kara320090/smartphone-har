import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('confirmatory',Path(__file__).parents[1]/'tools/confirmatory_har.py')
c=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)

def test_three_class_heads_and_deterministic_mapping():
    for name in c.CFG['conditions']:
        c.keras.backend.clear_session()
        c.configure_runtime(2026,4)
        m=c.build(name,2026)
        x=np.zeros((2,18) if name=='FMLP3' else (2,128,9),np.float32)
        assert m(x).shape==(2,3)
        assert np.isfinite(m(x).numpy()).all()
    np.testing.assert_array_equal(c.GROUP[np.arange(6)],[0,0,0,1,1,2])

def test_features_and_metrics_independent_expected_values():
    x=np.zeros((2,128,9),np.float32)
    x[:,:,0]=np.arange(128)
    f=c.features(x)
    assert f.shape==(2,18)
    assert f[0,0]==63.5
    assert abs(f[0,9]-np.sqrt((128**2-1)/12))<1e-12
    m=c.metrics(np.array([0,1,2]),np.array([0,0,2]),3)
    assert abs(m['macro_f1']-5/9)<1e-12

def test_test_loader_refuses_missing_or_tampered_lock_before_data(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'REP',tmp_path)
    monkeypatch.setattr(c,'DATA',tmp_path/'must_not_be_opened.npz')
    with pytest.raises(FileNotFoundError,match='FINAL_LOCK'):
        c.load_test()
    frozen=tmp_path/'frozen.txt';frozen.write_text('original')
    lock={'test_evaluated_at_freeze':False,'official_test_authorized_by_user':True,
          'files':{str(frozen):c.sha(frozen)}}
    (tmp_path/'FINAL_LOCK.json').write_text(json.dumps(lock))
    frozen.write_text('changed')
    with pytest.raises(ValueError,match='Frozen file changed'):
        c.load_test()
