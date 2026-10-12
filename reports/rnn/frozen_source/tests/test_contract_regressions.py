"""Regression checks for real adapter-boundary failures discovered in the audit."""
import numpy as np
import pytest
from src.contracts import prepare_data


@pytest.fixture
def samples():
    rng = np.random.default_rng(1107)
    return {"X_features": rng.normal(size=(18, 561)).astype(np.float32),
            "y": np.tile(np.arange(6), 3), "subject": np.repeat([3, 1, 5], 6),
            "sample_id": np.array([f"fixture:{i}" for i in range(18)])}


CONFIG = {"input_kind": "features", "standardize": True}
SPLIT = {"fit": np.arange(6), "validation": np.arange(6, 12)}


@pytest.mark.parametrize("bad", [
    np.arange(6) + .8, np.arange(-6, 0), np.array([True, False]),
    np.array([0, 0, 1]), np.array([18]), np.array([], dtype=int),
    np.array(0), np.array([[0, 1]]), np.array([2**64-1], dtype=np.uint64),
])
def test_invalid_split_is_rejected_before_preprocessor_fit(samples, bad, monkeypatch):
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Preprocessor.fit must not receive an invalid split")
    monkeypatch.setattr("src.contracts.Preprocessor.fit", forbidden_fit)
    with pytest.raises(ValueError):
        prepare_data(samples, {**SPLIT, "fit": bad}, CONFIG)


def test_reordered_raw_validation_rejected(samples):
    data = prepare_data(samples, SPLIT, CONFIG)
    data.raw_validation = data.raw_validation[::-1].copy()
    with pytest.raises(ValueError):
        data.validate()


def test_wrong_preprocessing_state_rejected(samples):
    data = prepare_data(samples, SPLIT, CONFIG)
    data.preprocessor.mean += 20
    with pytest.raises(ValueError):
        data.validate()


@pytest.mark.parametrize("subjects", [np.full(6, 3.2), np.full(6, -3), np.full((6, 1), 3)])
def test_subject_ids_are_positive_integer_vectors(samples, subjects):
    data = prepare_data(samples, SPLIT, CONFIG)
    data.fit_subjects = subjects
    with pytest.raises(ValueError):
        data.validate()


def test_valid_subset_and_validation_shift_preserve_fit_statistics(samples):
    before = prepare_data(samples, SPLIT, CONFIG)
    samples["X_features"][SPLIT["validation"]] += 10000
    after = prepare_data(samples, SPLIT, CONFIG)
    np.testing.assert_array_equal(before.preprocessor.mean, after.preprocessor.mean)
    np.testing.assert_array_equal(before.preprocessor.scale, after.preprocessor.scale)
    np.testing.assert_array_equal(before.x_fit, after.x_fit)
    after.validate()


def test_subject_overlap_rejected_before_fit(samples, monkeypatch):
    samples["subject"][6] = 3
    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Preprocessor.fit must not receive overlapping subjects")
    monkeypatch.setattr("src.contracts.Preprocessor.fit", forbidden_fit)
    with pytest.raises(ValueError):
        prepare_data(samples, SPLIT, CONFIG)
