#!/usr/bin/env python3
"""Generate overlay visualizations: MRI + GT vs MRI + Prediction for each completed subject."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ants

from utils import LABELS, COLORS, make_overlay, DATA_DIR, RESULTS_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_subject(subj_name):
    """Generate side-by-side GT vs prediction overlay for one subject."""
    mri_path = DATA_DIR / subj_name / "mri.nii.gz"
    gt_path = DATA_DIR / subj_name / "labels.nii.gz"
    pred_path = RESULTS_DIR / f"single_{subj_name}.nii.gz"

    if not pred_path.exists():
        print(f"  No prediction for {subj_name}, skipping")
        return False

    mri = ants.image_read(str(mri_path)).numpy()
    gt = ants.image_read(str(gt_path)).numpy()
    pred = ants.image_read(str(pred_path)).numpy()

    # Find slices with GT labels
    label_slices = np.where(np.any(gt > 0, axis=(0, 1)))[0]
    if len(label_slices) == 0:
        print(f"  No labeled slices for {subj_name}")
        return False

    # Pick 8 evenly-spaced slices from labeled region
    n_show = min(8, len(label_slices))
    indices = np.linspace(0, len(label_slices) - 1, n_show, dtype=int)
    show_slices = label_slices[indices]

    fig, axes = plt.subplots(2, n_show, figsize=(3.5 * n_show, 7))
    fig.suptitle(f"{subj_name.replace('_', ' ').title()}", fontsize=16, y=1.02)

    for col, sl in enumerate(show_slices):
        mri_sl = mri[:, :, sl]
        gt_sl = gt[:, :, sl]
        pred_sl = pred[:, :, sl]

        gt_overlay = make_overlay(mri_sl, gt_sl)
        pred_overlay = make_overlay(mri_sl, pred_sl)

        axes[0, col].imshow(gt_overlay)
        axes[0, col].set_title(f"Slice {sl}", fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(pred_overlay)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Ground Truth", fontsize=13, rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Prediction", fontsize=13, rotation=90, labelpad=10)

    # Add legend
    legend_elements = []
    for lid in sorted(LABELS.keys()):
        name = LABELS[lid]
        color = COLORS[lid]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=name))

    fig.legend(handles=legend_elements, loc="lower center", ncol=9,
               fontsize=8, bbox_to_anchor=(0.5, -0.05), frameon=False)

    plt.tight_layout()
    out_path = FIGURES_DIR / f"overlay_{subj_name}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    return True


def plot_combined_summary():
    """Single figure with one representative slice per subject: GT vs Pred."""
    subjects = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and d.name.endswith("_ext"):
            pred_path = RESULTS_DIR / f"single_{d.name}.nii.gz"
            if pred_path.exists():
                subjects.append(d.name)

    if not subjects:
        print("No completed subjects found")
        return

    n_subj = len(subjects)
    fig, axes = plt.subplots(2, n_subj, figsize=(4 * n_subj, 8))
    fig.suptitle("Automatic Forearm MRI Segmentation: GT (top) vs Prediction (bottom)",
                 fontsize=16, y=1.02)

    for col, subj in enumerate(subjects):
        mri = ants.image_read(str(DATA_DIR / subj / "mri.nii.gz")).numpy()
        gt = ants.image_read(str(DATA_DIR / subj / "labels.nii.gz")).numpy()
        pred = ants.image_read(str(RESULTS_DIR / f"single_{subj}.nii.gz")).numpy()

        # Find middle slice of labeled region
        label_slices = np.where(np.any(gt > 0, axis=(0, 1)))[0]
        if len(label_slices) == 0:
            continue
        mid_sl = label_slices[len(label_slices) // 2]

        gt_overlay = make_overlay(mri[:, :, mid_sl], gt[:, :, mid_sl])
        pred_overlay = make_overlay(mri[:, :, mid_sl], pred[:, :, mid_sl])

        if n_subj == 1:
            ax_gt = axes[0]
            ax_pred = axes[1]
        else:
            ax_gt = axes[0, col]
            ax_pred = axes[1, col]

        ax_gt.imshow(gt_overlay)
        ax_gt.set_title(subj.replace("_ext", "").replace("_", "-").upper(), fontsize=11)
        ax_gt.axis("off")

        ax_pred.imshow(pred_overlay)
        ax_pred.axis("off")

    if n_subj > 1:
        axes[0, 0].set_ylabel("Ground\nTruth", fontsize=13, rotation=0, labelpad=50)
        axes[1, 0].set_ylabel("Auto\nPrediction", fontsize=13, rotation=0, labelpad=50)
    else:
        axes[0].set_ylabel("Ground\nTruth", fontsize=13, rotation=0, labelpad=50)
        axes[1].set_ylabel("Auto\nPrediction", fontsize=13, rotation=0, labelpad=50)

    # Legend
    legend_elements = []
    for lid in sorted(LABELS.keys()):
        name = LABELS[lid]
        color = COLORS[lid]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=name))

    fig.legend(handles=legend_elements, loc="lower center", ncol=9,
               fontsize=8, bbox_to_anchor=(0.5, -0.05), frameon=False)

    plt.tight_layout()
    out_path = FIGURES_DIR / "summary_all_subjects.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    print("Generating overlay figures...")

    # Per-subject detailed overlays
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and d.name.endswith("_ext"):
            plot_subject(d.name)

    # Combined summary
    print("\nGenerating combined summary...")
    plot_combined_summary()
    print("Done!")
