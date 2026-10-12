import numpy as np
import pytest
from tensorflow import keras
from src.models import build_model, build_rnn
from src.runtime import configure_runtime
from src.adapters.processed_npz import TeamSequencePreprocessor


def test_rnn_structure_state_reset_and_parameter_count():
    configure_runtime(2026)
    m = build_rnn()
    assert isinstance(m.layers[0], keras.layers.SimpleRNN)
    assert m.count_params() == 1542
    assert build_rnn(units=64).count_params() == 5126
    assert m.input_shape == (None, 128, 9)
    assert m.output_shape == (None, 6)
    assert not m.layers[0].stateful
    assert not m.layers[0].return_sequences
    x = np.random.default_rng(12).normal(size=(2, 128, 9)).astype(np.float32)
    first = m(x, training=False).numpy()
    m(x * 2, training=False)
    np.testing.assert_array_equal(first, m(x, training=False).numpy())
    np.testing.assert_allclose(first[:1], m(x[:1], training=False).numpy(), atol=1e-6)



@pytest.mark.parametrize("kwargs", [dict(input_shape=(9,128)), dict(units=0), dict(units=True), dict(dropout=1)])
def test_invalid_rnn_config_rejected(kwargs):
    with pytest.raises(ValueError):
        build_rnn(**kwargs)


@pytest.mark.parametrize("defect", ["shape", "nan", "inf", "dtype"])
def test_bad_rnn_raw_input_rejected(defect):
    pre = TeamSequencePreprocessor(np.zeros((1,1,9), np.float32), np.ones((1,1,9), np.float32))
    x = np.zeros((2,128,9), np.float32)
    if defect == "shape":
        x = x.transpose(0,2,1)
    elif defect == "dtype":
        x = x.astype(np.float64)
    else:
        x[0,0,0] = np.nan if defect == "nan" else np.inf
    with pytest.raises(ValueError):
        pre.transform(x)


def test_registry_rnn():
    assert build_model((128,9), {"model_key":"rnn", "units":32}).count_params() == 1542
