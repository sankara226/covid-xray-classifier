"""Threshold selection and evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, roc_auc_score)


def select_threshold(y_true: np.ndarray, scores: np.ndarray, target_sensitivity: float) -> float:
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    positives = y_true == 1
    if not positives.any() or not 0 < target_sensitivity <= 1:
        raise ValueError("Validation labels need positives and target sensitivity must be in (0, 1]")
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    valid = [t for t in candidates if np.mean(scores[positives] >= t) >= target_sensitivity]
    if not valid:
        raise ValueError("Unable to meet target sensitivity")
    return float(max(valid))


def precision_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    if not 0 <= sensitivity <= 1 or not 0 <= specificity <= 1 or not 0 < prevalence <= 1:
        raise ValueError("Invalid sensitivity, specificity, or prevalence")
    denom = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    return sensitivity * prevalence / denom if denom else 0.0


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    tn, fp, fn, tp = confusion_matrix(y_true, scores >= threshold, labels=[0, 1]).ravel()
    return {"roc_auc": float(roc_auc_score(y_true, scores)),
            "pr_auc": float(average_precision_score(y_true, scores)), "threshold": float(threshold),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp))}
