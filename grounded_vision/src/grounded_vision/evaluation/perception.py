"""
Perception Evaluation Metrics
=============================
Evaluates:
1. Part localization accuracy (Intersection-over-Union, PCK).
2. Attribute prediction metrics (Accuracy, Balanced Accuracy, Precision, Recall, Macro-F1).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from ..perception.regions import Region, SpatialHierarchy


def compute_bbox_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes IoU between two (ymin, xmin, ymax, xmax) boxes."""
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    inter_ymin = max(y1_min, y2_min)
    inter_xmin = max(x1_min, x2_min)
    inter_ymax = min(y1_max, y2_max)
    inter_xmax = min(x1_max, x2_max)

    inter_area = max(0.0, inter_ymax - inter_ymin) * max(0.0, inter_xmax - inter_xmin)
    area1 = max(0.0, y1_max - y1_min) * max(0.0, x1_max - x1_min)
    area2 = max(0.0, y2_max - y2_min) * max(0.0, x2_max - x2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_perception_performance(
    pred_boxes_list: List[Dict[str, Tuple[float, float, float, float]]],
    gt_boxes_list: List[Dict[str, Tuple[float, float, float, float]]],
    pred_attrs_matrix: np.ndarray,
    gt_attrs_matrix: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes overall perception benchmark metrics.
    """
    # 1. Part localization IoU
    part_ious: Dict[str, List[float]] = {}
    part_success: Dict[str, List[int]] = {}

    for pred_boxes, gt_boxes in zip(pred_boxes_list, gt_boxes_list):
        for part, gt_box in gt_boxes.items():
            if part in pred_boxes:
                iou = compute_bbox_iou(pred_boxes[part], gt_box)
                part_ious.setdefault(part, []).append(iou)
                part_success.setdefault(part, []).append(1 if iou >= iou_threshold else 0)

    mean_iou = float(np.mean([np.mean(vals) for vals in part_ious.values()])) if part_ious else 0.0
    mean_pck = float(np.mean([np.mean(vals) for vals in part_success.values()])) if part_success else 0.0

    # 2. Attribute prediction metrics
    pred_binary = (pred_attrs_matrix >= 0.5).astype(int)
    gt_binary = (gt_attrs_matrix >= 0.5).astype(int)

    acc = float(accuracy_score(gt_binary.flatten(), pred_binary.flatten()))
    prec, rec, f1, _ = precision_recall_fscore_support(
        gt_binary.flatten(), pred_binary.flatten(), average="binary", zero_division=0
    )

    return {
        "mean_part_iou": mean_iou,
        "part_localization_accuracy_pck": mean_pck,
        "attribute_accuracy": acc,
        "attribute_precision": float(prec),
        "attribute_recall": float(rec),
        "attribute_f1": float(f1),
        "part_breakdown_iou": {p: float(np.mean(v)) for p, v in part_ious.items()},
    }
