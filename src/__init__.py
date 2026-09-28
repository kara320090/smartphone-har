"""Part 02: reproducible MLP training for smartphone HAR."""
import os

# Set before importing TensorFlow. CPU is the measured reference environment.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
