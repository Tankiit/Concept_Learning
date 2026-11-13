"""Crop before encoding; fixed boxes are part of the inference contract.

Guarantee starts at the supplied RGB tensor. Global preprocessing, rerunning a
localizer, batch statistics, or attention across crops need separate analysis.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def validate_boxes(boxes, shape, m):
    b = np.asarray(boxes)
    h, w = shape
    if b.shape != (m, 4) or not np.issubdtype(b.dtype, np.integer):
        raise ValueError('Boxes must be integer [M,4], ordered (y0,y1,x0,x1)')
    if not ((b[:, 0] >= 0) & (b[:, 1] <= h) & (b[:, 2] >= 0) &
            (b[:, 3] <= w) & (b[:, 0] < b[:, 1]) & (b[:, 2] < b[:, 3])).all():
        raise ValueError('Invalid/empty image crop')
    return b


class CropPredicateReader(nn.Module):
    """One small learned score predictor per named crop.

    Independent modules keep the dependency contract easy to inspect. Weight
    sharing would also be valid if inference never mixes data across crops.
    Returns logits, for stable masked BCE training.
    """
    def __init__(self, n_predicates, channels=8):
        super().__init__()
        if n_predicates < 1:
            raise ValueError('Need at least one predicate')
        self.networks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, channels, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 1))
            for _ in range(n_predicates)])

    def forward(self, images, boxes):
        if images.ndim != 4 or images.shape[1] != 3 or len(images) == 0:
            raise ValueError('Expected nonempty [N,3,H,W] image tensors')
        b = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
        if b.shape != (len(images), len(self.networks), 4):
            raise ValueError('Expected per-image fixed boxes [N,M,4]')
        rows = []
        for image, image_boxes in zip(images, b):
            validate_boxes(image_boxes, image.shape[-2:], len(self.networks))
            # Crucial: slice original pixels before ANY learned encoding.
            row = [net(image[None, :, y0:y1, x0:x1]).reshape(())
                   for net, (y0, y1, x0, x1) in zip(self.networks, image_boxes)]
            rows.append(torch.stack(row))
        return torch.stack(rows)


def masked_concept_loss(logits, labels, observed, pos_weight=None):
    """Unknown/unobserved annotations are excluded, never converted to negatives.

    `pos_weight` optionally reweights positives per predicate (shape [M] for
    logits [B,M]); it broadcasts to the masked elements. It changes training
    emphasis only, never the observation mask or the label semantics.
    """
    if logits.shape != labels.shape or logits.shape != observed.shape or observed.dtype != torch.bool:
        raise ValueError('Logits, labels and Boolean observation mask must have the same shape')
    if not bool(observed.any()):
        raise ValueError('Batch has no observed concept labels')
    target = labels[observed]
    if not torch.isfinite(target).all() or not ((target == 0) | (target == 1)).all():
        raise ValueError('Observed labels must be binary and finite')
    weight = None
    if pos_weight is not None:
        w = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
        if w.ndim != 1 or w.shape[0] != logits.shape[-1] or not (w > 0).all():
            raise ValueError('pos_weight must be positive with one entry per predicate')
        weight = w.expand(logits.shape)[observed]
    return F.binary_cross_entropy_with_logits(logits[observed], target.to(logits.dtype), pos_weight=weight)


def region_union(boxes, reason, shape):
    b = validate_boxes(boxes, shape, len(boxes))
    keep = np.zeros(shape, dtype=bool)
    seen = set()
    for lit in reason:
        if not isinstance(lit, (int, np.integer)) or isinstance(lit, bool) or not 1 <= abs(lit) <= len(b):
            raise ValueError('Invalid predicate literal')
        if abs(lit) in seen:
            raise ValueError('Duplicate/contradictory predicate literal')
        seen.add(abs(lit))
        # Negative literals also depend on their crop. Use union, not box sums.
        y0, y1, x0, x1 = b[abs(lit)-1]
        keep[y0:y1, x0:x1] = True
    return keep


def replace_outside(image, keep, seed=0):
    if tuple(keep.shape) != tuple(image.shape[-2:]):
        raise ValueError('Mask/image shapes do not agree')
    rng = np.random.default_rng(seed)
    replacement = torch.as_tensor(rng.random(tuple(image.shape)), device=image.device, dtype=image.dtype)
    mask = torch.as_tensor(keep, device=image.device, dtype=torch.bool)[None]
    return torch.where(mask, image, replacement)
