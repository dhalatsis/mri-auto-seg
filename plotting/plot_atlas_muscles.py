#!/usr/bin/env python3
"""Comprehensive plot showing muscle locations across all atlas subjects and
how the registration warps atlas labels onto a target.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ants

from utils import LABELS, COLORS, make_overlay, DATA_DIR, RESULTS_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_mid_slice(label_vol):
    """Find the middle labeled slice."""
    labeled = np.where(np.any(label_vol > 0, axis=(0, 1)))[0]
    if len(labeled) == 0:
        return label_vol.shape[2] // 2
    return labeled[len(labeled) // 2]


def main():
    # Load all extension subjects
    subjects = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and d.name.endswith("_ext"):
            if (d / "mri.nii.gz").exists() and (d / "labels.nii.gz").exists():
                subjects.append(d.name)

    print(f"Found {len(subjects)} subjects")

    # Figure 1: All atlas subjects - middle slice overlays (4 columns x 2 rows)
    n_subj = len(subjects)
    n_cols = 4
    n_rows = (n_subj + n_cols - 1) // n_cols

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    fig1.suptitle("All Atlas Subjects: Muscle Segmentation (Middle Slice)", fontsize=18, y=1.02)

    for idx, subj in enumerate(subjects):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes1[row, col] if n_rows > 1 else axes1[col]

        mri = ants.image_read(str(DATA_DIR / subj / "mri.nii.gz")).numpy()
        labels = ants.image_read(str(DATA_DIR / subj / "labels.nii.gz")).numpy()

        mid = get_mid_slice(labels)
        overlay = make_overlay(mri[:, :, mid], labels[:, :, mid])

        ax.imshow(overlay)
        ax.set_title(subj.replace("_ext", "").replace("_", "-").upper(), fontsize=12)
        ax.axis("off")

        # Count muscles present
        unique = set(np.unique(labels[:, :, mid]).astype(int)) - {0}
        ax.text(5, mri.shape[0] - 10, f"{len(unique)} muscles, slice {mid}",
                fontsize=9, color="white", va="bottom",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.6))

    # Hide unused axes
    for idx in range(n_subj, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes1[row, col] if n_rows > 1 else axes1[col]
        ax.axis("off")

    # Legend
    legend_elements = [mpatches.Patch(facecolor=COLORS[lid], label=LABELS[lid])
                       for lid in sorted(LABELS.keys())]
    fig1.legend(handles=legend_elements, loc="lower center", ncol=9,
                fontsize=9, bbox_to_anchor=(0.5, -0.02), frameon=False)

    plt.tight_layout()
    fig1.savefig(str(FIGURES_DIR / "all_atlas_muscles.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("Saved: figures/all_atlas_muscles.png")

    # Figure 2: Registration flow - Atlas -> Warped -> Target GT
    target_name = "sub-07_ext"
    atlas_name = "sub-01_ext"
    pred_name = f"single_{target_name}.nii.gz"

    target_mri = ants.image_read(str(DATA_DIR / target_name / "mri.nii.gz")).numpy()
    target_gt = ants.image_read(str(DATA_DIR / target_name / "labels.nii.gz")).numpy()
    atlas_mri = ants.image_read(str(DATA_DIR / atlas_name / "mri.nii.gz")).numpy()
    atlas_labels = ants.image_read(str(DATA_DIR / atlas_name / "labels.nii.gz")).numpy()
    pred = ants.image_read(str(RESULTS_DIR / pred_name)).numpy()

    mid_target = get_mid_slice(target_gt)
    mid_atlas = get_mid_slice(atlas_labels)

    fig2, axes2 = plt.subplots(2, 4, figsize=(22, 11))
    fig2.suptitle("Atlas-Based Registration Pipeline", fontsize=20, y=1.02)

    # Row 1: Pipeline flow
    atlas_mri_norm = (atlas_mri[:, :, mid_atlas] - atlas_mri[:, :, mid_atlas].min()) / \
                     (atlas_mri[:, :, mid_atlas].max() - atlas_mri[:, :, mid_atlas].min() + 1e-8)
    axes2[0, 0].imshow(atlas_mri_norm, cmap="gray")
    axes2[0, 0].set_title(f"1. Atlas MRI\n({atlas_name})", fontsize=13)
    axes2[0, 0].axis("off")

    atlas_overlay = make_overlay(atlas_mri[:, :, mid_atlas], atlas_labels[:, :, mid_atlas])
    axes2[0, 1].imshow(atlas_overlay)
    axes2[0, 1].set_title("2. Atlas Labels\n(known segmentation)", fontsize=13)
    axes2[0, 1].axis("off")

    target_mri_norm = (target_mri[:, :, mid_target] - target_mri[:, :, mid_target].min()) / \
                      (target_mri[:, :, mid_target].max() - target_mri[:, :, mid_target].min() + 1e-8)
    axes2[0, 2].imshow(target_mri_norm, cmap="gray")
    axes2[0, 2].set_title(f"3. Target MRI\n({target_name})", fontsize=13)
    axes2[0, 2].axis("off")

    axes2[0, 3].text(0.5, 0.5,
                     "SyN Registration\n\nAtlas MRI \u2192 Target MRI\n\n"
                     "Finds deformation field\nthat warps atlas anatomy\n"
                     "onto target anatomy.\n\n"
                     "Same transforms applied\nto atlas labels \u2192\n"
                     "predicted segmentation",
                     ha="center", va="center", fontsize=12,
                     bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.9),
                     transform=axes2[0, 3].transAxes)
    axes2[0, 3].set_title("4. Registration", fontsize=13)
    axes2[0, 3].axis("off")

    # Row 2: Results
    pred_overlay = make_overlay(target_mri[:, :, mid_target], pred[:, :, mid_target])
    axes2[1, 0].imshow(pred_overlay)
    axes2[1, 0].set_title("5. Predicted Labels\n(warped atlas \u2192 target)", fontsize=13)
    axes2[1, 0].axis("off")

    gt_overlay = make_overlay(target_mri[:, :, mid_target], target_gt[:, :, mid_target])
    axes2[1, 1].imshow(gt_overlay)
    axes2[1, 1].set_title("6. Ground Truth\n(manual segmentation)", fontsize=13)
    axes2[1, 1].axis("off")

    # Difference map
    diff = np.zeros((*target_mri.shape[:2], 3))
    pred_sl = pred[:, :, mid_target]
    gt_sl = target_gt[:, :, mid_target]
    correct = (pred_sl > 0) & (pred_sl == gt_sl)
    false_pos = (pred_sl > 0) & (pred_sl != gt_sl)
    false_neg = (pred_sl == 0) & (gt_sl > 0)

    diff[correct] = [0, 0.8, 0]
    diff[false_pos] = [0.8, 0, 0]
    diff[false_neg] = [0, 0, 0.8]

    bg = target_mri_norm * 0.3
    for c in range(3):
        diff[:, :, c] = np.where(diff[:, :, c] > 0, diff[:, :, c], bg)

    axes2[1, 2].imshow(diff)
    axes2[1, 2].set_title("7. Error Map", fontsize=13)
    axes2[1, 2].axis("off")

    error_legend = [
        mpatches.Patch(facecolor=[0, 0.8, 0], label="Correct"),
        mpatches.Patch(facecolor=[0.8, 0, 0], label="Wrong label"),
        mpatches.Patch(facecolor=[0, 0, 0.8], label="Missed"),
    ]
    axes2[1, 2].legend(handles=error_legend, loc="lower right", fontsize=9)

    # Per-slice Dice scores
    n_slices = target_gt.shape[2]
    slice_dices = []
    for sl in range(n_slices):
        p = pred[:, :, sl]
        g = target_gt[:, :, sl]
        inter = np.sum((p > 0) & (p == g))
        total = np.sum(p > 0) + np.sum(g > 0)
        if total > 0:
            slice_dices.append((sl, 2 * inter / total))

    if slice_dices:
        slices, dices = zip(*slice_dices)
        axes2[1, 3].plot(slices, dices, 'b-', linewidth=1.5)
        axes2[1, 3].fill_between(slices, dices, alpha=0.2)
        axes2[1, 3].axvline(x=mid_target, color='r', linestyle='--', alpha=0.5, label=f"Shown slice ({mid_target})")
        axes2[1, 3].set_xlabel("Slice Index", fontsize=11)
        axes2[1, 3].set_ylabel("Dice Score", fontsize=11)
        axes2[1, 3].set_title("8. Per-Slice Accuracy", fontsize=13)
        axes2[1, 3].set_ylim(0, 1)
        axes2[1, 3].legend(fontsize=9)
        axes2[1, 3].grid(alpha=0.3)

    # Muscle legend at bottom
    fig2.legend(handles=legend_elements, loc="lower center", ncol=9,
                fontsize=9, bbox_to_anchor=(0.5, -0.03), frameon=False)

    plt.tight_layout()
    fig2.savefig(str(FIGURES_DIR / "registration_pipeline.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved: figures/registration_pipeline.png")

    # Figure 3: Multiple slices across all subjects showing how muscles move
    fig3, axes3 = plt.subplots(len(subjects), 3, figsize=(12, 4 * len(subjects)))
    fig3.suptitle("Muscle Cross-Sections: Proximal to Distal (All Subjects)", fontsize=18, y=1.01)

    for row, subj in enumerate(subjects):
        mri = ants.image_read(str(DATA_DIR / subj / "mri.nii.gz")).numpy()
        labels = ants.image_read(str(DATA_DIR / subj / "labels.nii.gz")).numpy()

        labeled = np.where(np.any(labels > 0, axis=(0, 1)))[0]
        if len(labeled) < 3:
            continue

        positions = [labeled[len(labeled)//5], labeled[len(labeled)//2], labeled[4*len(labeled)//5]]
        pos_names = ["Proximal", "Middle", "Distal"]

        for col, (sl, pname) in enumerate(zip(positions, pos_names)):
            overlay = make_overlay(mri[:, :, sl], labels[:, :, sl])
            axes3[row, col].imshow(overlay)
            if col == 0:
                axes3[row, col].set_ylabel(
                    subj.replace("_ext", "").replace("_", "-").upper(),
                    fontsize=11, rotation=0, labelpad=80)
            if row == 0:
                axes3[row, col].set_title(pname, fontsize=13)
            axes3[row, col].axis("off")

    fig3.legend(handles=legend_elements, loc="lower center", ncol=9,
                fontsize=9, bbox_to_anchor=(0.5, -0.01), frameon=False)
    plt.tight_layout()
    fig3.savefig(str(FIGURES_DIR / "all_subjects_slices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("Saved: figures/all_subjects_slices.png")


if __name__ == "__main__":
    main()
