"""Evaluation metrics for segmentation."""

import numpy as np
from .labels import LABELS


def compute_dices(pred, gt):
    """Compute per-muscle Dice scores between prediction and ground truth arrays."""
    dices = {}
    for lid, name in sorted(LABELS.items()):
        p = pred == lid; g = gt == lid
        inter = np.sum(p & g); total = np.sum(p) + np.sum(g)
        if total > 0:
            dices[name] = float(2 * inter / total)
    return dices
