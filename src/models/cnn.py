"""1D CNN for user-independent smartphone HAR (six unnormalized logits).

Channels-last input: 128 time steps by 9 sensor channels. Convolutions slide
ONLY over the time dimension, not over subjects or feature columns.
"""
from keras import Input, Sequential, layers


def build_cnn(input_shape=(128, 9), dropout=0.3, seed=2026):
    """Return fixed-architecture temporal CNN with an explicit six-logit head."""
    if tuple(input_shape) != (128, 9):
        raise ValueError("CNN expects (128, 9), with time as axis 1")
    if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be numeric and in [0, 1)")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    return Sequential([
        Input(shape=(128, 9), dtype="float32", name="sensor_window"),
        layers.Conv1D(64, kernel_size=5, strides=1, padding="same",
                      activation="relu", name="conv1_time5"),
        layers.MaxPooling1D(pool_size=2, strides=2, padding="valid", name="pool1"),
        layers.Conv1D(128, kernel_size=3, strides=1, padding="same",
                      activation="relu", name="conv2_time3"),
        layers.MaxPooling1D(pool_size=2, strides=2, padding="valid", name="pool2"),
        layers.Conv1D(128, kernel_size=3, strides=1, padding="same",
                      activation="relu", name="conv3_time3"),
        layers.GlobalAveragePooling1D(name="global_average_time"),
        layers.Dense(64, activation="relu", name="hidden"),
        layers.Dropout(float(dropout), seed=seed + 1000, name="head_dropout"),
        layers.Dense(6, activation="linear", name="logits"),
    ], name="temporal_cnn_1d")
