#!/usr/bin/env python3
"""Visualize MRI segmentation results.

Usage:
    python plotting/visualize.py data/sub-01_ext/mri.nii.gz \
                        data/sub-01_ext/labels.nii.gz \
                        --output figures/sub-01.png

    # Compare prediction vs ground truth
    python plotting/visualize.py data/sub-02_ext/mri.nii.gz \
                        results/atlas_baseline/single_sub-02_ext.nii.gz \
                        --gt data/sub-02_ext/labels.nii.gz \
                        --output figures/comparison.png
"""

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path


# Distinct colors for up to 25 labels
LABEL_COLORS = [
    [0, 0, 0],        # 0: background
    [255, 0, 0],      # 1: skin/other
    [0, 128, 255],    # 2: ANC
    [255, 165, 0],    # 3: APL
    [0, 255, 0],      # 4: ECRB
    [128, 0, 255],    # 5: ECRL
    [255, 255, 0],    # 6: ECU
    [0, 255, 255],    # 7: ED
    [255, 0, 255],    # 8: EDM
    [128, 255, 0],    # 9: EPL
    [0, 128, 128],    # 10: FCR
    [255, 128, 0],    # 11: FCU
    [128, 0, 128],    # 12: FDP
    [64, 64, 255],    # 13: (unused)
    [255, 64, 128],   # 14: FDS
    [0, 200, 100],    # 15: FPL
    [200, 200, 0],    # 16: PL
    [100, 0, 200],    # 17: PQ
    [200, 100, 0],    # 18: PT
    [0, 100, 200],    # 19: SUP
    [150, 150, 150],  # 20+
    [200, 50, 50],
    [50, 200, 50],
    [50, 50, 200],
    [200, 200, 100],
    [100, 200, 200],
]

MUSCLE_NAMES = {
    2: "ANC", 3: "APL", 4: "ECRB", 5: "ECRL", 6: "ECU",
    7: "ED", 8: "EDM", 9: "EPL", 10: "FCR", 11: "FCU",
    12: "FDP", 14: "FDS", 15: "FPL", 16: "PL", 17: "PQ",
    18: "PT", 19: "SUP", 1: "Skin", 25: "Skin",
}


def create_overlay(mri_slice, label_slice, alpha=0.4):
    """Create an RGBA overlay of labels on MRI."""
    # Normalize MRI to 0-1
    mri_norm = mri_slice.astype(float)
    if mri_norm.max() > 0:
        mri_norm = mri_norm / np.percentile(mri_norm[mri_norm > 0], 99)
    mri_norm = np.clip(mri_norm, 0, 1)

    # Create RGB image from MRI (grayscale)
    rgb = np.stack([mri_norm] * 3, axis=-1)

    # Overlay labels
    for label_id in np.unique(label_slice):
        if label_id == 0:
            continue
        mask = label_slice == label_id
        color_idx = int(label_id) if int(label_id) < len(LABEL_COLORS) else -1
        color = np.array(LABEL_COLORS[color_idx]) / 255.0
        for c in range(3):
            rgb[:, :, c] = np.where(mask, rgb[:, :, c] * (1 - alpha) + color[c] * alpha, rgb[:, :, c])

    return np.clip(rgb, 0, 1)


def plot_slices(mri_data, label_data, title="", n_slices=6, gt_data=None, output_path=None):
    """Plot evenly spaced axial slices with label overlay."""
    n_total = mri_data.shape[2]

    # Pick slices that have labels
    labeled_slices = [z for z in range(n_total) if np.any(label_data[:, :, z] > 0)]
    if not labeled_slices:
        labeled_slices = list(range(n_total))

    indices = np.linspace(labeled_slices[0], labeled_slices[-1], n_slices, dtype=int)

    n_rows = 2 if gt_data is not None else 1
    fig, axes = plt.subplots(n_rows, n_slices, figsize=(4 * n_slices, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :] if n_slices > 1 else np.array([[axes]])

    for col, z in enumerate(indices):
        overlay = create_overlay(mri_data[:, :, z], label_data[:, :, z])
        axes[0, col].imshow(overlay, origin="lower")
        axes[0, col].set_title(f"Slice {z}")
        axes[0, col].axis("off")

        if gt_data is not None:
            gt_overlay = create_overlay(mri_data[:, :, z], gt_data[:, :, z])
            axes[1, col].imshow(gt_overlay, origin="lower")
            axes[1, col].set_title(f"GT Slice {z}")
            axes[1, col].axis("off")

    if gt_data is not None:
        axes[0, 0].set_ylabel("Prediction", fontsize=14)
        axes[1, 0].set_ylabel("Ground Truth", fontsize=14)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize MRI segmentation")
    parser.add_argument("mri", help="MRI NIfTI path")
    parser.add_argument("labels", help="Label NIfTI path")
    parser.add_argument("--gt", help="Ground truth label NIfTI for comparison")
    parser.add_argument("--output", "-o", help="Output figure path")
    parser.add_argument("--slices", type=int, default=6, help="Number of slices to show")
    parser.add_argument("--title", default="", help="Figure title")

    args = parser.parse_args()

    mri_img = nib.load(args.mri)
    label_img = nib.load(args.labels)
    mri_data = mri_img.get_fdata()
    label_data = label_img.get_fdata()

    gt_data = None
    if args.gt:
        gt_img = nib.load(args.gt)
        gt_data = gt_img.get_fdata()

    title = args.title or Path(args.labels).stem
    plot_slices(mri_data, label_data, title=title, n_slices=args.slices,
                gt_data=gt_data, output_path=args.output)


if __name__ == "__main__":
    main()
