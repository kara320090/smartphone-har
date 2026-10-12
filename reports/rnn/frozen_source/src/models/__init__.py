from .mlp import build_mlp
from .lstm import build_lstm
from .rnn import build_rnn
from .cnn import build_cnn

__all__ = ["build_mlp", "build_lstm", "build_rnn", "build_cnn", "build_model"]


def build_model(input_shape, config):
    key = config.get("model_key", "mlp")
    if key == "mlp":
        return build_mlp(input_shape, config.get("dropout", 0.0))
    if key == "lstm":
        return build_lstm(input_shape, config.get("units", 64),
                          config.get("dropout", 0.0), config.get("seed", 2026))
    if key == "rnn":
        return build_rnn(input_shape, config.get("units", 32),
                         config.get("dropout", 0.0), config.get("seed", 2026))
    if key == "cnn":
        return build_cnn(input_shape, config.get("seed", 2026))
    raise ValueError(f"Unsupported model_key: {key}")
