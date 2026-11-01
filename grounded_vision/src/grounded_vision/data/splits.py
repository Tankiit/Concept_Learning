"""
Dataset Split Utilities
=======================
Handles train/validation/test split generation, cross-validation folds,
and split verification for CUB-200.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def get_cub_split_indices(
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Computes class-stratified train, validation, and test splits.
    
    Args:
        labels: List of class labels for all samples.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion for validation.
        test_ratio: Proportion for testing.
        seed: Random seed for reproducibility.
        
    Returns:
        Dict mapping split name ('train', 'val', 'test') to list of sample indices.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    rng = np.random.RandomState(seed)
    
    labels_arr = np.array(labels)
    unique_classes = np.unique(labels_arr)
    
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    
    for cls in unique_classes:
        cls_idx = np.where(labels_arr == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n > 2 else 0
        
        train_part = cls_idx[:n_train]
        val_part = cls_idx[n_train : n_train + n_val]
        test_part = cls_idx[n_train + n_val :]
        
        train_indices.extend(train_part.tolist())
        val_indices.extend(val_part.tolist())
        test_indices.extend(test_part.tolist())
        
    return {
        "train": sorted(train_indices),
        "val": sorted(val_indices),
        "test": sorted(test_indices),
    }


def verify_splits(splits: Dict[str, List[int]], total_samples: int) -> bool:
    """Verifies that splits form a partition without leaks or missing items."""
    all_indices = set()
    for name, indices in splits.items():
        s = set(indices)
        if len(s) != len(indices):
            return False  # duplicate inside split
        if len(all_indices.intersection(s)) > 0:
            return False  # overlap between splits
        all_indices.update(s)
    return len(all_indices) == total_samples
