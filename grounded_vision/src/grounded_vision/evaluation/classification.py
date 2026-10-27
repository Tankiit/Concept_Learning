"""
Classification Evaluation Metrics
=================================
Evaluates downstream species classifier performance:
Accuracy, Macro-F1, Micro-F1, and Expected Calibration Error (ECE).
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def evaluate_classifier_performance(
    y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray
) -> Dict[str, Any]:
    """
    Computes classification benchmark metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    ece = compute_ece(y_probs, y_true)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "expected_calibration_error": ece,
    }
