"""CNN interface/shape test. Run with repository TensorFlow environment."""
import numpy as np
import pytest
from tensorflow import keras
from src.models.cnn import build_cnn
from src.models import build_model


def test_temporal_shapes_and_logits():
    model = build_cnn()
    assert model.input_shape == (None, 128, 9)
    assert model.output_shape == (None, 6)
    assert model.count_params() == 85574
    expected = {"conv1_time5": (None, 128, 64), "pool1": (None, 64, 64),
                "conv2_time3": (None, 64, 128), "pool2": (None, 32, 128),
                "conv3_time3": (None, 32, 128), "global_average_time": (None, 128),
                "hidden": (None, 64), "logits": (None, 6)}
    for layer in model.layers:
        if layer.name in expected:
            assert tuple(layer.output.shape) == expected[layer.name]
    assert model.layers[-1].activation == keras.activations.linear
    values = model(np.zeros((3, 128, 9), dtype=np.float32), training=False).numpy()
    assert values.shape == (3, 6)
    assert np.isfinite(values).all()


def test_model_registry():
    config = {"model_key": "cnn", "dropout": 0.3, "seed": 2026}
    assert build_model((128, 9), config).count_params() == 85574


@pytest.mark.parametrize("shape", [(9, 128), (561,), (128, 3)])
def test_reject_bad_shapes(shape):
    with pytest.raises(ValueError, match="128"):
        build_cnn(shape)


@pytest.mark.parametrize("rate", [-0.1, 1.0, float("nan"), None])
def test_reject_bad_dropout(rate):
    with pytest.raises(ValueError, match="dropout"):
        build_cnn(dropout=rate)
