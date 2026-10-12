"""Minimal validation metrics for training; final evaluation belongs to part 04."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 6 or not np.isfinite(logits).all():
        raise ValueError("Expected finite (B,6) logits")
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def classification_metrics(y_true, logits):
    probabilities = softmax(logits)
    pred = probabilities.argmax(axis=1)
    return {"accuracy": float(accuracy_score(y_true, pred)),
            "macro_f1": float(f1_score(y_true, pred, labels=list(range(6)),
                                        average="macro", zero_division=0)),
            "recall_per_class": recall_score(y_true, pred, labels=list(range(6)),
                                               average=None, zero_division=0).tolist(),
            "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(6))).tolist()}
