"""LSTM integration, source integrity and deliberately malformed input checks."""
import json
from pathlib import Path
import numpy as np
import pytest
from tensorflow import keras
from src.adapters.processed_npz import load_processed_data, TeamSequencePreprocessor
from src.models import build_model, build_lstm
from src.runtime import configure_runtime
from src.train import train_model
from src.verify_bundle import verify_bundle, processed_verification_samples


@pytest.fixture
def cache(tmp_path):
    rng = np.random.default_rng(15)
    fit = rng.normal(size=(60, 128, 9)).astype(np.float32)
    val = rng.normal(size=(30, 128, 9)).astype(np.float32)
    arrays = dict(fit_Xs=fit, fit_y=np.tile(np.arange(6, dtype=np.int64), 10),
                  fit_subject=np.full(60, 3, dtype=np.int64),
                  fit_sample_id=np.array([f"train:{i:06d}" for i in range(60)]),
                  val_Xs=val, val_y=np.tile(np.arange(6, dtype=np.int64), 5),
                  val_subject=np.full(30, 1, dtype=np.int64),
                  val_sample_id=np.array([f"train:{i:06d}" for i in range(60, 90)]),
                  # This array cannot be loaded with allow_pickle=False.
                  test_Xs=np.array([{"sentinel": "must never load"}], dtype=object))
    cp, sp = tmp_path / "processed.npz", tmp_path / "stats.npz"
    np.savez_compressed(cp, **arrays)
    np.savez(sp, seq_mean=fit.mean((0, 1), keepdims=True),
             seq_std=fit.std((0, 1), keepdims=True))
    return cp, sp, arrays


def test_team_transform_exact_and_test_arrays_unread(cache, tmp_path):
    cp, sp, arrays = cache
    data = load_processed_data(cp, sp)
    with np.load(sp) as z:
        expected = (arrays["val_Xs"] - z["seq_mean"]) / z["seq_std"]
    np.testing.assert_array_equal(expected, data.x_validation)
    assert data.fit_ids[0] == "train:000000"
    assert data.provenance["sample_row_number_base"] == 0
    assert not data.provenance["test_arrays_loaded"]
    assert all(not key.startswith("test") for key in data.provenance["loaded_array_keys"])
    path = tmp_path / "roundtrip.npz"
    data.preprocessor.save(path)
    np.testing.assert_array_equal(expected, TeamSequencePreprocessor.load(path).transform(arrays["val_Xs"]))


@pytest.mark.parametrize("defect", ["shape", "nan", "label", "duplicate", "overlap", "checksum", "stats"])
def test_malformed_cache_rejected(cache, defect):
    cp, sp, arrays = cache
    if defect == "shape":
        arrays["val_Xs"] = arrays["val_Xs"].transpose(0, 2, 1)
    elif defect == "nan":
        arrays["fit_Xs"][0, 0, 0] = np.nan
    elif defect == "label":
        arrays["val_y"][0] = 6
    elif defect == "duplicate":
        arrays["val_sample_id"][1] = arrays["val_sample_id"][0]
    elif defect == "overlap":
        arrays["val_subject"][:] = 3
    elif defect == "stats":
        with np.load(sp) as z:
            mean, scale = z["seq_mean"].copy(), z["seq_std"].copy()
        mean += np.float32(0.1)
        np.savez(sp, seq_mean=mean, seq_std=scale)
    np.savez_compressed(cp, **arrays)
    with pytest.raises(ValueError):
        load_processed_data(cp, sp, expected_data_sha256="wrong" if defect == "checksum" else None)


@pytest.mark.parametrize("target", ["validation_ids", "raw_validation", "y_validation", "x_fit"])
def test_changed_row_binding_rejected(cache, target):
    data = load_processed_data(*cache[:2])
    array = getattr(data, target)
    array[[0, 1]] = array[[1, 0]]
    with pytest.raises(ValueError, match="disagree|binding"):
        data.validate()


def test_lstm_sizes_independent_windows_and_initialization():
    configure_runtime(2026)
    m = build_lstm()
    assert m.count_params() == 21222
    assert build_lstm(units=32).count_params() == 6630
    assert m.output_shape == (None, 6)
    assert m.layers[0].stateful is False
    x = np.random.default_rng(1).normal(size=(2, 128, 9)).astype(np.float32)
    first = m(x, training=False).numpy()
    m(x * 2, training=False)
    np.testing.assert_array_equal(first, m(x, training=False).numpy())
    configure_runtime(2026)
    other = build_lstm()
    for a, b in zip(m.get_weights(), other.get_weights()):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        build_lstm((9, 128))
    with pytest.raises(ValueError):
        build_model((128, 9), {"model_key": "unknown"})


def test_matched_initialization_and_complete_reload(cache, tmp_path):
    data = load_processed_data(*cache[:2], smoke=True)
    config = dict(model_key="lstm", experiment_id="L01", units=64, seed=2026,
                  epochs=1, batch_size=30, verbose=0, run_directory=str(tmp_path / "base"))
    configure_runtime(2026)
    base = train_model(build_lstm(), data, config)
    configure_runtime(2026)
    config.update(experiment_id="L03", dropout=0.3, require_matched_initialization=True,
                  initial_weights_from=str(base), run_directory=str(tmp_path / "drop"))
    drop = train_model(build_lstm(dropout=0.3), data, config)
    with np.load(base / "initial_weights.npz") as a, np.load(drop / "initial_weights.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    assert json.loads((drop / "initialization.json").read_text())["matched"]
    metadata = json.loads((drop / "metadata.json").read_text())
    assert metadata["sample_row_number_base"] == 0
    assert metadata["preprocessing"] == "team_sequence_npz_v1"
    samples, indices = processed_verification_samples(*cache[:2], smoke=True)
    assert verify_bundle(drop, samples, indices)["count"] == 30


def test_dropout_cannot_run_without_initial_weights(cache, tmp_path):
    data = load_processed_data(*cache[:2])
    with pytest.raises(ValueError, match="requires L01"):
        train_model(build_lstm(dropout=0.3), data, dict(run_directory=str(tmp_path / "bad"),
                    model_key="lstm", require_matched_initialization=True))
