"""
Certifiable Decision Tree Head
==============================
The deployed interpretable reasoning engine. Wraps a Decision Tree
classifier and exposes exact decision path extraction, node conditions,
rule exportation, and interfaces for formal sufficiency verification.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from sklearn.tree import DecisionTreeClassifier


@dataclass
class DecisionPathStep:
    """A single conditional test in a decision tree path."""
    node_id: int
    feature_idx: int
    feature_name: str
    threshold: float
    is_left: bool  # True if value <= threshold, False if value > threshold
    value: float  # The actual observed feature value in the instance

    @property
    def condition_str(self) -> str:
        op = "<=" if self.is_left else ">"
        return f"{self.feature_name} {op} {self.threshold:.3f}"


class CertifiableTreeHead:
    """
    Deployed decision tree classifier with certifiable path export.
    """
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.feature_names = feature_names or []
        self.class_names = class_names or []
        self.random_state = random_state

        self.clf = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=self.random_state,
        )
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
    ) -> CertifiableTreeHead:
        """Trains the decision tree on compiled features."""
        if feature_names is not None:
            self.feature_names = feature_names
        if class_names is not None:
            self.class_names = class_names

        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def get_decision_path(self, x: np.ndarray) -> List[DecisionPathStep]:
        """
        Extracts the explicit sequence of conditional splits taken for a single sample x.

        Args:
            x: 1D array of shape [num_features]

        Returns:
            List of DecisionPathStep instances leading to the leaf.
        """
        assert self.is_fitted, "Tree must be fitted before extracting decision path."
        if x.ndim == 1:
            x_2d = x.reshape(1, -1)
        else:
            x_2d = x

        tree = self.clf.tree_
        feature = tree.feature
        threshold = tree.threshold

        node_indicator = self.clf.decision_path(x_2d)
        node_indices = node_indicator.indices[
            node_indicator.indptr[0] : node_indicator.indptr[1]
        ]

        steps: List[DecisionPathStep] = []
        for i, node_id in enumerate(node_indices):
            feat_idx = feature[node_id]
            if feat_idx != -2:  # -2 denotes leaf node in sklearn
                thresh = float(threshold[node_id])
                val = float(x_2d[0, feat_idx])
                is_left = val <= thresh
                feat_name = (
                    self.feature_names[feat_idx]
                    if feat_idx < len(self.feature_names)
                    else f"feat_{feat_idx}"
                )
                steps.append(
                    DecisionPathStep(
                        node_id=int(node_id),
                        feature_idx=int(feat_idx),
                        feature_name=feat_name,
                        threshold=thresh,
                        is_left=is_left,
                        value=val,
                    )
                )

        return steps

    def export_rules(self) -> List[Dict[str, Any]]:
        """Exports all leaf paths as symbolic rules."""
        assert self.is_fitted
        tree = self.clf.tree_
        rules: List[Dict[str, Any]] = []

        def recurse(node_id: int, current_path: List[Tuple[int, str, float, bool]]):
            if tree.feature[node_id] == -2:  # Leaf
                class_counts = tree.value[node_id][0]
                pred_class = int(np.argmax(class_counts))
                samples = int(tree.n_node_samples[node_id])
                rules.append({
                    "leaf_id": int(node_id),
                    "predicted_class": pred_class,
                    "class_name": self.class_names[pred_class] if pred_class < len(self.class_names) else str(pred_class),
                    "samples": samples,
                    "conditions": [
                        {
                            "feature_idx": f_idx,
                            "feature_name": f_name,
                            "threshold": thresh,
                            "is_left": left,
                        }
                        for f_idx, f_name, thresh, left in current_path
                    ],
                })
            else:
                f_idx = tree.feature[node_id]
                thresh = float(tree.threshold[node_id])
                f_name = self.feature_names[f_idx] if f_idx < len(self.feature_names) else f"feat_{f_idx}"
                
                # Left branch (<= thresh)
                recurse(tree.children_left[node_id], current_path + [(f_idx, f_name, thresh, True)])
                # Right branch (> thresh)
                recurse(tree.children_right[node_id], current_path + [(f_idx, f_name, thresh, False)])

        recurse(0, [])
        return rules
