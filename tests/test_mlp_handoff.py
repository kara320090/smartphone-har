"""Guards against reusing a run for different conditions and invalid live inputs."""
import json
import numpy as np
import pytest
from scripts.run_mlp_suite import check_reusable
from src.mlp_inference import predict_raw_bundle


@pytest.mark.parametrize("bad", [np.zeros((1, 127, 9), np.float32), np.zeros((1, 128, 8), np.float32),
                                np.zeros((1, 1152), np.float32), np.zeros((0, 128, 9), np.float32),
                                np.zeros((1, 128, 9), np.float64), np.full((1, 128, 9), np.nan, np.float32),
                                np.full((1, 128, 9), np.inf, np.float32)])
def test_malformed_prediction_is_rejected_before_model_io(tmp_path, bad):
    # No model exists: validation must fail first, rather than accidentally loading it.
    with pytest.raises(ValueError):
        predict_raw_bundle(tmp_path / "nonexistent", bad)


def test_previous_run_cannot_be_reused_after_condition_change(tmp_path):
    run = tmp_path / "M01_seed2026"
    run.mkdir()
    (run / "status.json").write_text(json.dumps({"status": "complete"}))
    expected = {"experiment_id": "M01", "learning_rate": .001, "expected_data_sha256": "original"}
    (run / "config.json").write_text(json.dumps(dict(expected, seed=2026, smoke=False)))
    assert check_reusable(run, expected, 2026, False)
    for changed in ({**expected, "learning_rate": .01}, {**expected, "expected_data_sha256": "different"}):
        with pytest.raises(ValueError, match="config differs"):
            check_reusable(run, changed, 2026, False)
    with pytest.raises(ValueError, match="config differs"):
        check_reusable(run, expected, 2027, False)


def test_incomplete_run_is_preserved_for_diagnosis(tmp_path):
    run = tmp_path / "failed"
    run.mkdir()
    (run / "status.json").write_text(json.dumps({"status": "failed", "error": "interrupted"}))
    with pytest.raises(RuntimeError, match="Incomplete run preserved"):
        check_reusable(run, {}, 2026, False)
    assert json.loads((run / "status.json").read_text())["error"] == "interrupted"
