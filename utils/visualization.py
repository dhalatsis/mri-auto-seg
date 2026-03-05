"""Shared visualization helpers."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .labels import LABELS

# Generate distinct colors for each label using tab20 colormap
COLORS = {}
_cmap = plt.cm.tab20(np.linspace(0, 1, 20))
for _lid in sorted(LABELS.keys()):
    COLORS[_lid] = _cmap[_lid % 20][:3]


def make_overlay(mri_slice, label_slice, alpha=0.55):
    """Create RGB overlay of labels on MRI."""
    mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
    rgb = np.stack([mri_norm] * 3, axis=-1)

    for lid, color in COLORS.items():
        mask = label_slice == lid
        if not np.any(mask):
            continue
        for c in range(3):
            rgb[:, :, c][mask] = mri_norm[mask] * (1 - alpha) + color[c] * alpha

    return rgb
