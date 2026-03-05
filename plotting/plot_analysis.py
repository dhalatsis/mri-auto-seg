#!/usr/bin/env python3
"""Detailed analysis plots: good vs bad muscles, error patterns, improvement opportunities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ants

from utils import LABELS, NAME_TO_LID, COLORS, make_overlay, DATA_DIR, RESULTS_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    with open(RESULTS_DIR / "results_single.json") as f:
        return json.load(f)


def plot1_muscle_tiers():
    """Categorize muscles into tiers and show with box plots."""
    results = load_results()

    muscle_vals = {}
    for subj, data in results.items():
        for muscle, dice in data["dices"].items():
            muscle_vals.setdefault(muscle, []).append(dice)

    # Sort by mean dice
    sorted_muscles = sorted(muscle_vals.keys(), key=lambda m: np.mean(muscle_vals[m]), reverse=True)
    means = [np.mean(muscle_vals[m]) for m in sorted_muscles]

    # Tier colors
    tier_colors = []
    for m in means:
        if m >= 0.70:
            tier_colors.append("#2ca02c")
        elif m >= 0.55:
            tier_colors.append("#ff7f0e")
        else:
            tier_colors.append("#d62728")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [2, 1]})

    bp = ax1.boxplot([muscle_vals[m] for m in sorted_muscles],
                     labels=sorted_muscles, patch_artist=True, vert=True,
                     medianprops=dict(color="black", linewidth=2))

    for patch, color in zip(bp["boxes"], tier_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, m in enumerate(sorted_muscles):
        vals = muscle_vals[m]
        x = np.random.normal(i + 1, 0.06, len(vals))
        ax1.scatter(x, vals, color="black", s=20, alpha=0.6, zorder=3)

    ax1.axhline(y=0.70, color="green", linestyle="--", alpha=0.4, linewidth=2)
    ax1.axhline(y=0.55, color="orange", linestyle="--", alpha=0.4, linewidth=2)
    ax1.axhline(y=0.90, color="blue", linestyle="--", alpha=0.3, linewidth=2, label="Target (0.90)")

    ax1.set_ylabel("Dice Score", fontsize=14)
    ax1.set_title("Per-Muscle Segmentation Quality (7-Subject LOOCV)", fontsize=15)
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.2)

    tier_patches = [
        mpatches.Patch(facecolor="#2ca02c", alpha=0.7, label="Good (Dice > 0.70)"),
        mpatches.Patch(facecolor="#ff7f0e", alpha=0.7, label="Moderate (0.55-0.70)"),
        mpatches.Patch(facecolor="#d62728", alpha=0.7, label="Needs Work (< 0.55)"),
    ]
    ax1.legend(handles=tier_patches, fontsize=11, loc="lower left")

    ax2.axis("off")
    table_data = []
    for m in sorted_muscles:
        vals = muscle_vals[m]
        mean = np.mean(vals)
        std = np.std(vals)
        worst = np.min(vals)
        best = np.max(vals)
        tier = "Good" if mean >= 0.70 else ("Moderate" if mean >= 0.55 else "Needs Work")
        table_data.append([m, f"{mean:.3f}", f"{std:.3f}", f"{worst:.3f}", f"{best:.3f}", tier])

    table = ax2.table(
        cellText=table_data,
        colLabels=["Muscle", "Mean", "Std", "Worst", "Best", "Tier"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for i, m in enumerate(sorted_muscles):
        mean = np.mean(muscle_vals[m])
        if mean >= 0.70:
            color = "#d4edda"
        elif mean >= 0.55:
            color = "#fff3cd"
        else:
            color = "#f8d7da"
        for j in range(6):
            table[i + 1, j].set_facecolor(color)

    ax2.set_title("Muscle Segmentation Summary", fontsize=14, pad=20)

    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "muscle_tiers.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/muscle_tiers.png")


def plot2_good_vs_bad_examples():
    """Side-by-side comparison of best and worst segmented muscles."""
    results = load_results()

    best_muscles = ["FDP", "ED", "ECU"]
    worst_muscles = ["PL", "PT", "ECRB"]

    good_subj = "sub-03_ext"
    bad_subj = "sub-06_ext"

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.suptitle("Best vs Worst Muscle Segmentation Examples", fontsize=18, y=1.02)

    for row, (subj, label) in enumerate([(good_subj, "Best Subject (S3, Dice=0.71)"),
                                          (bad_subj, "Worst Subject (S9, Dice=0.55)")]):
        mri = ants.image_read(str(DATA_DIR / subj / "mri.nii.gz")).numpy()
        gt = ants.image_read(str(DATA_DIR / subj / "labels.nii.gz")).numpy()
        pred = ants.image_read(str(RESULTS_DIR / f"single_{subj}.nii.gz")).numpy()

        labeled_slices = np.where(np.any(gt > 0, axis=(0, 1)))[0]
        mid = labeled_slices[len(labeled_slices) // 2]

        mri_sl = mri[:, :, mid]
        gt_sl = gt[:, :, mid]
        pred_sl = pred[:, :, mid]

        gt_overlay = make_overlay(mri_sl, gt_sl)
        axes[row, 0].imshow(gt_overlay)
        axes[row, 0].set_title(f"{label}\nGround Truth (slice {mid})", fontsize=12)
        axes[row, 0].axis("off")

        pred_overlay = make_overlay(mri_sl, pred_sl)
        axes[row, 1].imshow(pred_overlay)
        axes[row, 1].set_title("Prediction", fontsize=12)
        axes[row, 1].axis("off")

        mri_norm = (mri_sl - mri_sl.min()) / (mri_sl.max() - mri_sl.min() + 1e-8)
        good_img = np.stack([mri_norm * 0.5] * 3, axis=-1)
        for muscle in best_muscles:
            lid = NAME_TO_LID[muscle]
            gt_mask = gt_sl == lid
            pred_mask = pred_sl == lid
            overlap = gt_mask & pred_mask
            color = COLORS[lid]
            for c in range(3):
                good_img[:, :, c][overlap] = color[c] * 0.8
            fn = gt_mask & ~pred_mask
            good_img[:, :, 2][fn] = 0.7
            fp = pred_mask & ~gt_mask
            good_img[:, :, 0][fp] = 0.7

        axes[row, 2].imshow(good_img)
        axes[row, 2].set_title(f"Good muscles: {', '.join(best_muscles)}\n(color=overlap, blue=missed, red=extra)",
                               fontsize=10)
        axes[row, 2].axis("off")

        bad_img = np.stack([mri_norm * 0.5] * 3, axis=-1)
        for muscle in worst_muscles:
            lid = NAME_TO_LID[muscle]
            gt_mask = gt_sl == lid
            pred_mask = pred_sl == lid
            overlap = gt_mask & pred_mask
            color = COLORS[lid]
            for c in range(3):
                bad_img[:, :, c][overlap] = color[c] * 0.8
            fn = gt_mask & ~pred_mask
            bad_img[:, :, 2][fn] = 0.7
            fp = pred_mask & ~gt_mask
            bad_img[:, :, 0][fp] = 0.7

        axes[row, 3].imshow(bad_img)
        axes[row, 3].set_title(f"Hard muscles: {', '.join(worst_muscles)}\n(color=overlap, blue=missed, red=extra)",
                               fontsize=10)
        axes[row, 3].axis("off")

    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "good_vs_bad_muscles.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/good_vs_bad_muscles.png")


def plot3_muscle_size_vs_dice():
    """Scatter: muscle volume (voxels) vs Dice score."""
    results = load_results()

    muscle_sizes = {}
    muscle_dices = {}

    for d in sorted(DATA_DIR.iterdir()):
        if not (d.is_dir() and d.name.endswith("_ext") and (d / "labels.nii.gz").exists()):
            continue
        gt = ants.image_read(str(d / "labels.nii.gz")).numpy()
        for lid, name in LABELS.items():
            count = np.sum(gt == lid)
            if count > 0:
                muscle_sizes.setdefault(name, []).append(count)

    for subj, data in results.items():
        for muscle, dice in data["dices"].items():
            muscle_dices.setdefault(muscle, []).append(dice)

    fig, ax = plt.subplots(figsize=(12, 8))

    for muscle in sorted(LABELS.values()):
        if muscle not in muscle_sizes or muscle not in muscle_dices:
            continue
        mean_size = np.mean(muscle_sizes[muscle])
        mean_dice = np.mean(muscle_dices[muscle])
        std_dice = np.std(muscle_dices[muscle])
        lid = NAME_TO_LID[muscle]

        ax.errorbar(mean_size, mean_dice, yerr=std_dice,
                    fmt="o", markersize=12, color=COLORS[lid],
                    capsize=5, capthick=2, elinewidth=1.5,
                    markeredgecolor="black", markeredgewidth=1)
        ax.annotate(muscle, (mean_size, mean_dice),
                   textcoords="offset points", xytext=(8, 5), fontsize=11, fontweight="bold")

    ax.set_xlabel("Mean Muscle Volume (voxels)", fontsize=14)
    ax.set_ylabel("Mean Dice Score", fontsize=14)
    ax.set_title("Muscle Size vs Segmentation Accuracy", fontsize=16)
    ax.set_xscale("log")
    ax.axhline(y=0.70, color="green", linestyle="--", alpha=0.3, label="Good threshold")
    ax.axhline(y=0.90, color="blue", linestyle="--", alpha=0.3, label="Target (0.90)")
    ax.set_ylim(0.2, 1.0)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "size_vs_dice.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/size_vs_dice.png")


def plot4_subject_variability():
    """Heatmap: Dice per muscle per subject."""
    results = load_results()

    muscles = sorted(LABELS.values())
    subjects = sorted(results.keys())

    matrix = np.zeros((len(muscles), len(subjects)))
    for j, subj in enumerate(subjects):
        for i, muscle in enumerate(muscles):
            matrix[i, j] = results[subj]["dices"].get(muscle, 0)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([s.replace("_ext", "").replace("_", " ").upper() for s in subjects],
                       rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(len(muscles)))
    ax.set_yticklabels(muscles, fontsize=11)

    for i in range(len(muscles)):
        for j in range(len(subjects)):
            val = matrix[i, j]
            color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    for i, muscle in enumerate(muscles):
        mean = np.mean(matrix[i, :])
        ax.text(len(subjects) + 0.3, i, f"{mean:.3f}", ha="left", va="center",
                fontsize=10, fontweight="bold")
    ax.text(len(subjects) + 0.3, -1, "Mean", ha="left", va="center", fontsize=10, fontweight="bold")

    for j, subj in enumerate(subjects):
        mean = np.mean(matrix[:, j])
        ax.text(j, len(muscles) + 0.3, f"{mean:.3f}", ha="center", va="top",
                fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Dice Score", shrink=0.8)
    ax.set_title("Dice Score Heatmap: Every Muscle \u00d7 Every Subject", fontsize=16)

    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "dice_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/dice_heatmap.png")


def plot5_boundary_errors():
    """Show where errors occur: boundary vs interior for a representative subject."""
    from scipy.ndimage import binary_erosion

    subj = "sub-03_ext"
    gt = ants.image_read(str(DATA_DIR / subj / "labels.nii.gz")).numpy()
    pred = ants.image_read(str(RESULTS_DIR / f"single_{subj}.nii.gz")).numpy()
    mri = ants.image_read(str(DATA_DIR / subj / "mri.nii.gz")).numpy()

    labeled_slices = np.where(np.any(gt > 0, axis=(0, 1)))[0]
    mid = labeled_slices[len(labeled_slices) // 2]

    gt_sl = gt[:, :, mid]
    pred_sl = pred[:, :, mid]
    mri_sl = mri[:, :, mid]
    mri_norm = (mri_sl - mri_sl.min()) / (mri_sl.max() - mri_sl.min() + 1e-8)

    boundary_errors = np.zeros((*gt_sl.shape, 3))
    bg = np.stack([mri_norm * 0.4] * 3, axis=-1)
    boundary_errors = bg.copy()

    for lid in np.unique(gt_sl):
        if lid == 0:
            continue
        gt_mask = gt_sl == lid
        pred_mask = pred_sl == lid

        interior = binary_erosion(gt_mask, iterations=3)
        boundary = gt_mask & ~interior

        correct_interior = interior & pred_mask
        correct_boundary = boundary & pred_mask

        boundary_fn = boundary & ~pred_mask
        interior_fn = interior & ~pred_mask

        for c in range(3):
            boundary_errors[:, :, c][correct_interior] = [0, 0.7, 0][c]
            boundary_errors[:, :, c][correct_boundary] = [0, 0.5, 0.3][c]
            boundary_errors[:, :, c][boundary_fn] = [1.0, 0.8, 0][c]
            boundary_errors[:, :, c][interior_fn] = [0.9, 0, 0][c]

    fp = (pred_sl > 0) & (gt_sl == 0)
    boundary_errors[:, :, 0][fp] = 0.7
    boundary_errors[:, :, 1][fp] = 0.0
    boundary_errors[:, :, 2][fp] = 0.7

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Error Analysis: {subj} (slice {mid})", fontsize=16)

    gt_overlay = make_overlay(mri_sl, gt_sl)
    axes[0].imshow(gt_overlay)
    axes[0].set_title("Ground Truth", fontsize=13)
    axes[0].axis("off")

    pred_overlay = make_overlay(mri_sl, pred_sl)
    axes[1].imshow(pred_overlay)
    axes[1].set_title("Prediction", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(boundary_errors)
    axes[2].set_title("Error Map", fontsize=13)
    axes[2].axis("off")

    error_legend = [
        mpatches.Patch(facecolor=[0, 0.7, 0], label="Correct (interior)"),
        mpatches.Patch(facecolor=[0, 0.5, 0.3], label="Correct (boundary)"),
        mpatches.Patch(facecolor=[1.0, 0.8, 0], label="Boundary miss"),
        mpatches.Patch(facecolor=[0.9, 0, 0], label="Interior miss"),
        mpatches.Patch(facecolor=[0.7, 0, 0.7], label="False positive"),
    ]
    axes[2].legend(handles=error_legend, loc="lower right", fontsize=10)

    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "error_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/error_analysis.png")


def plot6_gap_to_target():
    """Show gap from current performance to 0.9 target per muscle."""
    results = load_results()

    muscle_vals = {}
    for subj, data in results.items():
        for muscle, dice in data["dices"].items():
            muscle_vals.setdefault(muscle, []).append(dice)

    muscles = sorted(muscle_vals.keys(), key=lambda m: np.mean(muscle_vals[m]), reverse=True)
    means = [np.mean(muscle_vals[m]) for m in muscles]
    gaps = [0.9 - m for m in means]

    fig, ax = plt.subplots(figsize=(14, 7))

    bars_current = ax.barh(range(len(muscles)), means, color="#2ca02c", alpha=0.8, label="Current Dice")
    bars_gap = ax.barh(range(len(muscles)), gaps, left=means, color="#d62728", alpha=0.4, label="Gap to 0.90")

    ax.set_yticks(range(len(muscles)))
    ax.set_yticklabels(muscles, fontsize=12)
    ax.set_xlabel("Dice Score", fontsize=14)
    ax.set_title("Gap Analysis: Current Performance vs 0.90 Target", fontsize=16)
    ax.axvline(x=0.9, color="blue", linestyle="--", linewidth=2, alpha=0.5, label="Target (0.90)")
    ax.axvline(x=0.658, color="gray", linestyle=":", linewidth=1.5, alpha=0.5, label="Current mean (0.658)")
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="x", alpha=0.2)

    for i, (m, gap) in enumerate(zip(muscles, gaps)):
        ax.text(0.92, i, f"+{gap:.2f}", va="center", fontsize=10, color="#d62728", fontweight="bold")

    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "gap_to_target.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: figures/gap_to_target.png")


if __name__ == "__main__":
    print("Generating analysis plots...")
    plot1_muscle_tiers()
    plot2_good_vs_bad_examples()
    plot3_muscle_size_vs_dice()
    plot4_subject_variability()
    plot5_boundary_errors()
    plot6_gap_to_target()
    print("\nAll analysis plots generated!")
