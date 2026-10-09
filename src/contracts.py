"""Training accepts arrays from any loader that obeys this documented contract."""
from dataclasses import dataclass, field
import hashlib
from typing import Any
import numpy as np
from .adapters.preprocessing_reference import Preprocessor
from .adapters.uci_reference import SPLIT_ID


def _subject_vector(subjects, count):
    values = np.asarray(subjects)
    if (values.ndim != 1 or len(values) != count or
            not np.issubdtype(values.dtype, np.integer) or np.any(values <= 0)):
        raise ValueError("Subject IDs must be aligned positive integer vectors")
    return values


def _row_indices(values, count, name):
    """Validate before casting: a float truncation or negative index changes rows."""
    values = np.asarray(values)
    if (values.ndim != 1 or len(values) == 0 or
            not np.issubdtype(values.dtype, np.integer)):
        raise ValueError(f"{name} indices must be a nonempty integer vector")
    if np.any(values < 0) or np.any(values >= count):
        raise ValueError(f"{name} row indices must be in [0, {count})")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{name} row indices contain duplicates")
    return values.astype(np.int64, copy=False)


def row_binding_digest(*arrays):
    """Bind array rows, labels, IDs and subjects without converting ID bases."""
    digest = hashlib.sha256()
    for value in arrays:
        value = np.asarray(value)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


@dataclass
class TrainingData:
    x_fit: np.ndarray
    y_fit: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    fit_ids: np.ndarray
    validation_ids: np.ndarray
    fit_subjects: np.ndarray
    validation_subjects: np.ndarray
    preprocessor: Any  # Must provide save(path), transform(raw), input_kind.
    raw_validation: np.ndarray
    split_id: str = SPLIT_ID
    smoke: bool = False
    provenance: dict = field(default_factory=dict)

    def validate(self):
        for x, y, ids, subjects in (
            (self.x_fit, self.y_fit, self.fit_ids, self.fit_subjects),
            (self.x_validation, self.y_validation, self.validation_ids, self.validation_subjects),
        ):
            if len(x) == 0 or any(len(a) != len(x) for a in (y, ids, subjects)):
                raise ValueError("Nonempty aligned input/label/ID/subject arrays required")
            if x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError("Inputs must be finite float32 arrays")
            if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or not np.isin(y, range(6)).all():
                raise ValueError("Labels must be integer vectors in 0..5")
            _subject_vector(subjects, len(x))
            if ids.ndim != 1:
                raise ValueError("Sample IDs must be aligned vectors")
            if len(set(ids.tolist())) != len(ids):
                raise ValueError("Duplicate sample IDs")
        if self.x_fit.shape[1:] != self.x_validation.shape[1:]:
            raise ValueError("Fit/validation input shapes differ")
        if set(self.fit_subjects.tolist()) & set(self.validation_subjects.tolist()):
            raise ValueError("Fit/validation subjects overlap")
        if set(self.fit_ids.tolist()) & set(self.validation_ids.tolist()):
            raise ValueError("Fit/validation sample IDs overlap")
        if len(self.raw_validation) != len(self.x_validation):
            raise ValueError("Raw validation rows do not align")
        replay = self.preprocessor.transform(self.raw_validation)
        if (replay.shape != self.x_validation.shape or not np.isfinite(replay).all() or
                not np.allclose(replay, self.x_validation, rtol=1e-5, atol=1e-6)):
            raise ValueError("Raw validation and prepared input disagree under the saved preprocessing state")
        for part, arrays in (
            ("fit", (self.x_fit, self.y_fit, self.fit_ids, self.fit_subjects)),
            ("validation", (self.raw_validation, self.y_validation,
                            self.validation_ids, self.validation_subjects)),
        ):
            expected = self.provenance.get(f"{part}_row_binding_sha256")
            if expected and row_binding_digest(*arrays) != expected:
                raise ValueError(f"{part} row binding differs from the loaded source")


def prepare_data(samples, split_indices, config, smoke=False):
    """Adapter from team load_dataset() output to fit-only TrainingData."""
    kind = config["input_kind"]
    if kind not in ("features", "sequence"):
        raise ValueError(f"Unknown input kind: {kind}")
    raw = samples["X_features" if kind == "features" else "X_seq"]
    fit = _row_indices(split_indices["fit"], len(raw), "Fit")
    val = _row_indices(split_indices["validation"], len(raw), "Validation")
    if len(np.intersect1d(fit, val)):
        raise ValueError("Fit/validation row indices overlap")
    subjects = _subject_vector(samples["subject"], len(raw))
    if set(subjects[fit].tolist()) & set(subjects[val].tolist()):
        raise ValueError("Fit/validation subjects overlap")
    if smoke:
        def small(index, n):
            return np.concatenate([index[samples["y"][index] == label][:n] for label in range(6)])
        fit, val = small(fit, 10), small(val, 5)
    pre = Preprocessor.fit(raw[fit], kind, config["standardize"], config.get("pca_variance"))
    data = TrainingData(pre.transform(raw[fit]), samples["y"][fit],
                        pre.transform(raw[val]), samples["y"][val],
                        samples["sample_id"][fit], samples["sample_id"][val],
                        samples["subject"][fit], samples["subject"][val], pre,
                        raw[val], smoke=smoke)
    data.validate()
    return data
