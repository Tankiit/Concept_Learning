"""
Concept Intervention and Counterfactual Evaluation
==================================================
Evaluates the causal effect of intervening on grounded concepts:
Tests model behavior when a concept is artificially flipped do(X_j = v),
measuring prediction sensitivity and counterfactual invariance.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np

from ..reasoning.tree_head import CertifiableTreeHead
from ..reasoning.features import FeatureCompiler


def evaluate_interventions(
    tree_head: CertifiableTreeHead,
    compiler: FeatureCompiler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    salient_feature_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Performs concept interventions and measures impact on predictions.
    """
    assert tree_head.is_fitted
    num_samples = len(X_test)
    base_preds = tree_head.predict(X_test)

    if salient_feature_indices is None:
        # Use top 5 features split in the tree
        tree = tree_head.clf.tree_
        used_feats = [f for f in tree.feature if f >= 0]
        salient_feature_indices = list(set(used_feats))[:5]

    flip_rates: Dict[str, float] = {}

    for f_idx in salient_feature_indices:
        f_name = compiler.idx_to_feature.get(f_idx, f"feat_{f_idx}")
        # Intervene: flip binary value 1.0 -> 0.0 or 0.0 -> 1.0
        X_intervened = X_test.copy()
        X_intervened[:, f_idx] = 1.0 - X_intervened[:, f_idx]

        new_preds = tree_head.predict(X_intervened)
        changed = np.sum(new_preds != base_preds)
        flip_rate = float(changed / num_samples)
        flip_rates[f_name] = flip_rate

    mean_flip_rate = float(np.mean(list(flip_rates.values()))) if flip_rates else 0.0

    return {
        "num_samples_evaluated": num_samples,
        "intervened_features_count": len(salient_feature_indices),
        "mean_prediction_flip_rate": mean_flip_rate,
        "feature_flip_rates": flip_rates,
    }
