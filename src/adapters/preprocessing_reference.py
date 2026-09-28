"""Fit-only reference preprocessing with a portable, pickle-free NPZ format."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class Preprocessor:
    input_kind: str
    standardize: bool
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    pca_mean: np.ndarray
    explained_variance_ratio: np.ndarray

    @classmethod
    def fit(cls, x_fit, input_kind="features", standardize=True, pca_variance=None):
        x_fit = np.asarray(x_fit, dtype=np.float64)
        cls._validate_input(x_fit, input_kind)
        if pca_variance is not None and (input_kind != "features" or not standardize):
            raise ValueError("PCA requires standardized feature input")
        axes = (0, 1) if input_kind == "sequence" else (0,)
        mean = x_fit.mean(axis=axes) if standardize else np.zeros(x_fit.shape[-1])
        scale = x_fit.std(axis=axes) if standardize else np.ones(x_fit.shape[-1])
        scale = np.where(scale == 0, 1.0, scale)
        obj = cls(input_kind, standardize, mean, scale, np.empty((0, x_fit.shape[-1])),
                  np.zeros(x_fit.shape[-1]), np.empty(0))
        if pca_variance is not None:
            pca = PCA(n_components=pca_variance, svd_solver="full")
            pca.fit((x_fit - mean) / scale)
            obj.components = pca.components_
            obj.pca_mean = pca.mean_
            obj.explained_variance_ratio = pca.explained_variance_ratio_
        return obj

    @staticmethod
    def _validate_input(x, kind):
        if kind not in ("features", "sequence"):
            raise ValueError(f"Unknown input kind: {kind}")
        expected = (561,) if kind == "features" else (128, 9)
        if x.ndim != len(expected) + 1 or x.shape[1:] != expected or len(x) == 0:
            raise ValueError(f"Expected nonempty (B,{','.join(map(str, expected))}) input")
        if not np.isfinite(x).all():
            raise ValueError("Input contains NaN or Inf")

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        self._validate_input(x, self.input_kind)
        z = (x - self.mean) / self.scale
        if len(self.components):
            z = (z - self.pca_mean) @ self.components.T
        z = z.astype(np.float32)
        if not np.isfinite(z).all():
            raise ValueError("Preprocessing produced non-finite values")
        return z

    def save(self, path):
        np.savez_compressed(path, schema_version=np.array(1),
                            input_kind=np.array(self.input_kind),
                            standardize=np.array(self.standardize),
                            mean=self.mean, scale=self.scale, components=self.components,
                            pca_mean=self.pca_mean,
                            explained_variance_ratio=self.explained_variance_ratio)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as z:
            if int(z["schema_version"]) != 1:
                raise ValueError("Unsupported preprocessing schema")
            return cls(str(z["input_kind"]), bool(z["standardize"]), z["mean"],
                       z["scale"], z["components"], z["pca_mean"],
                       z["explained_variance_ratio"])
