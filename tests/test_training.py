import json
from pathlib import Path
import numpy as np
import pytest
from tensorflow import keras
from src.adapters.preprocessing_reference import Preprocessor
from src.contracts import TrainingData
from src.math_checks import check_gradients, scalar_example
from src.models import build_mlp
from src.runtime import configure_runtime
from src.train import predict_batches, train_model
from src.training_metrics import classification_metrics, softmax


@pytest.fixture
def data():
    rng = np.random.default_rng(4)
    raw_fit = rng.normal(size=(60, 561)).astype(np.float32)
    raw_val = rng.normal(size=(30, 561)).astype(np.float32)
    pre = Preprocessor.fit(raw_fit)
    return TrainingData(pre.transform(raw_fit), np.tile(np.arange(6), 10),
                        pre.transform(raw_val), np.tile(np.arange(6), 5),
                        np.array([f"train:{i:06d}" for i in range(1, 61)]),
                        np.array([f"train:{i:06d}" for i in range(61, 91)]),
                        np.full(60, 3), np.full(30, 1), pre, raw_val, smoke=True)


def test_no_validation_leakage_and_preprocess_roundtrip(tmp_path):
    rng = np.random.default_rng(2026)
    fit = rng.normal(3, 2, (80, 561)).astype(np.float32)
    fit[:, 0] = 4
    val = rng.normal(200, 10, (8, 561)).astype(np.float32)
    pre = Preprocessor.fit(fit, pca_variance=0.95)
    np.testing.assert_allclose(pre.mean, fit.astype(np.float64).mean(0))
    assert pre.scale[0] == 1
    before_mean = pre.mean.copy()
    transformed = pre.transform(val)
    np.testing.assert_array_equal(pre.mean, before_mean)
    pre.save(tmp_path / "pre.npz")
    restored = Preprocessor.load(tmp_path / "pre.npz")
    np.testing.assert_array_equal(transformed, restored.transform(val))
    with pytest.raises(ValueError, match="NaN"):
        restored.transform(np.full((1, 561), np.nan))
    with pytest.raises(ValueError, match="Expected"):
        restored.transform(np.zeros((1, 560)))


def test_sequence_uses_global_per_channel_statistics():
    fit = np.random.default_rng(1).normal(size=(4, 128, 9)).astype(np.float32)
    pre = Preprocessor.fit(fit, "sequence")
    np.testing.assert_allclose(pre.mean, fit.astype(np.float64).mean(axis=(0, 1)))
    assert pre.transform(fit).shape == (4, 128, 9)
    assert not np.allclose(pre.transform(fit).mean(axis=1), 0)


def test_subject_and_id_overlap_rejected(data):
    data.fit_subjects[0] = 1
    with pytest.raises(ValueError, match="subjects overlap"):
        data.validate()
    data.fit_subjects[0] = 3
    data.validation_ids[0] = data.fit_ids[0]
    with pytest.raises(ValueError, match="sample IDs overlap"):
        data.validate()


def test_model_sizes_and_reproducible_initialization():
    configure_runtime(2026)
    feature = build_mlp()
    first_weights = feature.get_weights()
    assert feature.count_params() == 80582
    assert feature.output_shape == (None, 6)
    assert feature.layers[-1].activation == keras.activations.linear
    assert build_mlp((128, 9)).count_params() == 156230
    configure_runtime(2026)
    for first, second in zip(first_weights, build_mlp().get_weights()):
        np.testing.assert_array_equal(first, second)


def test_metrics_include_all_six_classes_and_stable_softmax():
    logits = np.full((2, 6), -10000.0)
    logits[0, 0] = 10000
    logits[1, 1] = 10000
    result = classification_metrics(np.array([0, 1]), logits)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == pytest.approx(2 / 6)
    np.testing.assert_allclose(softmax(logits).sum(1), 1)


def test_hand_numpy_autodiff_and_numerical_gradient():
    scalar = scalar_example()
    assert scalar["loss"] == 4.5
    assert scalar["updated_loss_by_learning_rate"]["0.1"] == pytest.approx(1.125)
    assert scalar["updated_loss_by_learning_rate"]["0.5"] == pytest.approx(10.125)
    summary, rows = check_gradients()
    assert summary["passed"] and summary["parameter_count"] == 17
    assert len(rows) == 51


def test_training_save_reload_and_overwrite_guard(data, tmp_path):
    configure_runtime()
    model = build_mlp(dropout=0.3)
    out = train_model(model, data, {"run_directory": str(tmp_path / "run"),
                                  "epochs": 2, "verbose": 0})
    restored = keras.models.load_model(out / "model.keras", compile=False)
    np.testing.assert_allclose(predict_batches(model, data.x_validation),
                               predict_batches(restored, data.x_validation), rtol=1e-5, atol=1e-6)
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["smoke"] and metrics["epochs_ran"] == 2 and not metrics["test_evaluated"]
    assert (out / "validation_predictions.csv").exists()
    with pytest.raises(FileExistsError):
        train_model(model, data, {"run_directory": str(out), "epochs": 2})


def test_common_trainer_accepts_external_classifier_and_reconstruction(data, tmp_path):
    configure_runtime()
    linear = keras.Sequential([keras.Input((561,)), keras.layers.Dense(6)])
    out = train_model(linear, data, {"run_directory": str(tmp_path / "linear"),
                                   "epochs": 1, "verbose": 0})
    assert json.loads((out / "metrics.json").read_text())["parameters"] == 3372
    reconstruction = keras.Sequential([keras.Input((561,)), keras.layers.Dense(8, activation="relu"),
                                       keras.layers.Dense(561)])
    out = train_model(reconstruction, data, {"run_directory": str(tmp_path / "ae"),
                         "task_type": "reconstruction", "epochs": 1, "verbose": 0})
    result = json.loads((out / "metrics.json").read_text())
    assert np.isfinite(result["reconstruction_mse"])
    assert not (out / "validation_predictions.csv").exists()
