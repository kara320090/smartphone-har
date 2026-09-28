"""Training accepts arrays from any loader that obeys this documented contract."""
from dataclasses import dataclass
from typing import Any
import numpy as np
from .adapters.preprocessing_reference import Preprocessor
from .adapters.uci_reference import SPLIT_ID


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


def prepare_data(samples, split_indices, config, smoke=False):
    """Adapter from team load_dataset() output to fit-only TrainingData."""
    fit = np.asarray(split_indices["fit"], dtype=np.int64)
    val = np.asarray(split_indices["validation"], dtype=np.int64)
    if len(np.intersect1d(fit, val)):
        raise ValueError("Fit/validation row indices overlap")
    if smoke:
        def small(index, n):
            return np.concatenate([index[samples["y"][index] == label][:n] for label in range(6)])
        fit, val = small(fit, 10), small(val, 5)
    kind = config["input_kind"]
    raw = samples["X_features" if kind == "features" else "X_seq"]
    pre = Preprocessor.fit(raw[fit], kind, config["standardize"], config.get("pca_variance"))
    data = TrainingData(pre.transform(raw[fit]), samples["y"][fit],
                        pre.transform(raw[val]), samples["y"][val],
                        samples["sample_id"][fit], samples["sample_id"][val],
                        samples["subject"][fit], samples["subject"][val], pre,
                        raw[val], smoke=smoke)
    data.validate()
    return data
