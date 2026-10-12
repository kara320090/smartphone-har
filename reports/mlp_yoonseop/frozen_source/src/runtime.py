import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
import sklearn
import tensorflow as tf
from tensorflow import keras


def configure_runtime(seed=2026, threads=4):
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(threads)
    except RuntimeError:
        # An existing notebook/pytest session may have initialized TensorFlow.
        pass
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                   allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment_info():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for folder, suffix in (("src", ".py"), ("configs", ".json"), ("scripts", ".py")):
        for path in sorted((root / folder).rglob(f"*{suffix}")):
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                         capture_output=True, text=True, check=False)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "tensorflow": tf.__version__,
            "keras": keras.__version__, "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "code_sha256": h.hexdigest(),
            "git_commit": rev.stdout.strip() if rev.returncode == 0 else None,
            "devices": [str(d) for d in tf.config.list_physical_devices()],
            "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
            "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
            "deterministic_ops": True}
