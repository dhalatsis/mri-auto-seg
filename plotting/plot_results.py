#!/usr/bin/env python3
"""Generate figures from evaluation results (works with partial results too)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import LABELS, RESULTS_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Old results (before slice fix) for comparison
OLD_SMART = {
    "ANC": 0.2309, "APL": 0.5016, "ECRB": 0.2286, "ECRL": 0.4687,
    "ECU": 0.4353, "ED": 0.5361, "EDM": 0.3896, "EPL": 0.4182,
    "FCR": 0.3626, "FCU": 0.4209, "FDP": 0.6502, "FDS": 0.5431,
    "FPL": 0.5023, "PL": 0.1652, "PQ": 0.3562, "PT": 0.1453,
    "SUP": 0.3479,
}
OLD_OVERALL = 0.3943


def load_results():
    """Load whatever results are available."""
    single = {}
    multi = {}

    single_path = RESULTS_DIR / "results_single.json"
    multi_path = RESULTS_DIR / "results_multi3.json"

    if single_path.exists():
        with open(single_path) as f:
            single = json.load(f)
    if multi_path.exists():
        with open(multi_path) as f:
            multi = json.load(f)

    return single, multi


def plot_per_muscle_comparison(single_results, multi_results):
    """Bar chart: per-muscle Dice for single vs multi vs old."""
    muscle_names = sorted(LABELS.values())

    # Aggregate per-muscle
    single_means = {}
    multi_means = {}
    single_stds = {}
    multi_stds = {}

    for muscle in muscle_names:
        s_vals = [d["dices"].get(muscle, 0) for d in single_results.values()]
        m_vals = [d["dices"].get(muscle, 0) for d in multi_results.values()]
        single_means[muscle] = np.mean(s_vals) if s_vals else 0
        multi_means[muscle] = np.mean(m_vals) if m_vals else 0
        single_stds[muscle] = np.std(s_vals) if s_vals else 0
        multi_stds[muscle] = np.std(m_vals) if m_vals else 0

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(muscle_names))
    width = 0.25

    old_vals = [OLD_SMART.get(m, 0) for m in muscle_names]
    single_vals = [single_means[m] for m in muscle_names]
    multi_vals = [multi_means[m] for m in muscle_names]
    single_err = [single_stds[m] for m in muscle_names]
    multi_err = [multi_stds[m] for m in muscle_names]

    bars1 = ax.bar(x - width, old_vals, width, label=f"Before Fix (mean={np.mean(old_vals):.3f})",
                   color="#d62728", alpha=0.8)
    bars2 = ax.bar(x, single_vals, width, yerr=single_err,
                   label=f"Fixed Single-Atlas (mean={np.mean(single_vals):.3f})",
                   color="#1f77b4", alpha=0.8, capsize=3)
    bars3 = ax.bar(x + width, multi_vals, width, yerr=multi_err,
                   label=f"Fixed Multi-Atlas Top-3 (mean={np.mean(multi_vals):.3f})",
                   color="#2ca02c", alpha=0.8, capsize=3)

    ax.set_xlabel("Muscle", fontsize=14)
    ax.set_ylabel("Dice Score", fontsize=14)
    ax.set_title("Per-Muscle Segmentation Accuracy (LOOCV, Extension)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(muscle_names, rotation=45, ha="right", fontsize=11)
    ax.legend(fontsize=12, loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.4, label="")

    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "per_muscle_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'per_muscle_comparison.png'}")


def plot_per_subject(single_results, multi_results):
    """Per-subject mean Dice comparison."""
    subjects = sorted(single_results.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(subjects))
    width = 0.35

    s_vals = [single_results[s]["mean_dice"] for s in subjects]
    m_vals = [multi_results[s]["mean_dice"] for s in subjects]

    bars1 = ax.bar(x - width/2, s_vals, width, label="Smart Single-Atlas",
                   color="#1f77b4", alpha=0.8)
    bars2 = ax.bar(x + width/2, m_vals, width, label="Smart Multi-Atlas (Top-3)",
                   color="#2ca02c", alpha=0.8)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Subject", fontsize=14)
    ax.set_ylabel("Mean Dice Score", fontsize=14)
    ax.set_title("Per-Subject Segmentation Accuracy", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_ext", "") for s in subjects],
                       rotation=30, ha="right", fontsize=11)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "per_subject_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'per_subject_comparison.png'}")


def plot_improvement_waterfall():
    """Show improvement from old to new results per muscle."""
    muscle_names = sorted(LABELS.values())

    fig, ax = plt.subplots(figsize=(14, 6))

    old_vals = [OLD_SMART.get(m, 0) for m in muscle_names]
    improvements = []
    for m in muscle_names:
        old = OLD_SMART.get(m, 0)
        improvements.append(0)

    colors = ["#2ca02c" if imp >= 0 else "#d62728" for imp in improvements]

    ax.bar(range(len(muscle_names)), old_vals, color="#d62728", alpha=0.5,
           label="Before Fix")
    ax.set_xticks(range(len(muscle_names)))
    ax.set_xticklabels(muscle_names, rotation=45, ha="right")
    ax.set_ylabel("Dice Score")
    ax.set_title("Pre-Fix Baseline Accuracy (to be compared with post-fix)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "baseline_before_fix.png"), dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'baseline_before_fix.png'}")


def plot_mri_overlay_examples():
    """Plot MRI slices with predicted vs GT label overlays."""
    try:
        import ants
    except ImportError:
        print("ANTsPy not available, skipping overlay plots")
        return

    from utils import DATA_DIR

    subjects = sorted([d.name for d in DATA_DIR.iterdir()
                       if d.is_dir() and d.name.endswith("_ext")
                       and (d / "labels.nii.gz").exists()])

    # Use first available subject with prediction
    for subj in subjects:
        pred_path = RESULTS_DIR / f"multi3_{subj}.nii.gz"
        if not pred_path.exists():
            pred_path = RESULTS_DIR / f"single_{subj}.nii.gz"
        if not pred_path.exists():
            continue

        mri = ants.image_read(str(DATA_DIR / subj / "mri.nii.gz")).numpy()
        gt = ants.image_read(str(DATA_DIR / subj / "labels.nii.gz")).numpy()
        pred = ants.image_read(str(pred_path)).numpy()

        # Find slices with labels
        label_slices = np.where(np.any(gt > 0, axis=(0, 1)))[0]
        if len(label_slices) == 0:
            continue

        # Pick 6 evenly-spaced slices
        n_show = min(6, len(label_slices))
        indices = np.linspace(0, len(label_slices)-1, n_show, dtype=int)
        show_slices = label_slices[indices]

        # Color map for labels
        n_labels = 20
        cmap = plt.cm.tab20(np.linspace(0, 1, n_labels))

        fig, axes = plt.subplots(3, n_show, figsize=(4*n_show, 12))
        fig.suptitle(f"Segmentation Results: {subj}", fontsize=16, y=0.98)

        for col, sl in enumerate(show_slices):
            # MRI
            mri_slice = mri[:, :, sl]
            mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)

            # Ground truth overlay
            gt_slice = gt[:, :, sl]
            gt_rgb = np.stack([mri_norm]*3, axis=-1)
            for lid in np.unique(gt_slice):
                if lid == 0:
                    continue
                mask = gt_slice == lid
                color = cmap[int(lid) % n_labels][:3]
                for c in range(3):
                    gt_rgb[:, :, c][mask] = mri_norm[mask] * 0.4 + color[c] * 0.6

            # Prediction overlay
            pred_slice = pred[:, :, sl]
            pred_rgb = np.stack([mri_norm]*3, axis=-1)
            for lid in np.unique(pred_slice):
                if lid == 0:
                    continue
                mask = pred_slice == lid
                color = cmap[int(lid) % n_labels][:3]
                for c in range(3):
                    pred_rgb[:, :, c][mask] = mri_norm[mask] * 0.4 + color[c] * 0.6

            # Plot
            axes[0, col].imshow(mri_norm, cmap="gray")
            axes[0, col].set_title(f"Slice {sl}", fontsize=11)
            axes[0, col].axis("off")

            axes[1, col].imshow(gt_rgb)
            axes[1, col].axis("off")

            axes[2, col].imshow(pred_rgb)
            axes[2, col].axis("off")

        axes[0, 0].set_ylabel("MRI", fontsize=14, rotation=0, labelpad=50)
        axes[1, 0].set_ylabel("Ground Truth", fontsize=14, rotation=0, labelpad=80)
        axes[2, 0].set_ylabel("Prediction", fontsize=14, rotation=0, labelpad=80)

        plt.tight_layout()
        fig.savefig(str(FIGURES_DIR / f"overlay_{subj}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {FIGURES_DIR / f'overlay_{subj}.png'}")
        break  # Just do first available subject


def plot_partial_results():
    """Generate a quick summary with whatever results are available from the output file."""
    output_file = Path("/tmp/claude-1000/-home-dc23-projects-mri-auto-seg/tasks/b217bjkjf.output")
    if not output_file.exists():
        print("No output file found")
        return

    text = output_file.read_text()

    # Parse completed subjects
    subjects = {}
    pattern = r"Target: (\S+).*?Single-atlas \(best\): mean_dice=(\d+\.\d+).*?Multi-atlas \(top-3\): mean_dice=(\d+\.\d+)"
    for match in re.finditer(pattern, text, re.DOTALL):
        name = match.group(1)
        single = float(match.group(2))
        multi = float(match.group(3))
        subjects[name] = {"single": single, "multi": multi}

    # Parse per-muscle results
    muscle_pattern = r"Target: (\S+).*?Muscle\s+Single\s+Multi3\s*\n\s*-+\n((?:\s+\w+\s+\d+\.\d+\s+\d+\.\d+\n)+)"
    muscle_data = {}
    for match in re.finditer(muscle_pattern, text, re.DOTALL):
        name = match.group(1)
        muscle_block = match.group(2)
        muscles = {}
        for line in muscle_block.strip().split("\n"):
            parts = line.split()
            if len(parts) == 3:
                muscles[parts[0]] = {"single": float(parts[1]), "multi": float(parts[2])}
        muscle_data[name] = muscles

    if not subjects:
        print("No completed subjects found yet")
        return

    print(f"\nPartial results ({len(subjects)} subjects completed):")
    for name, vals in subjects.items():
        print(f"  {name}: single={vals['single']:.4f}, multi={vals['multi']:.4f}")

    # Generate per-muscle figure from partial results
    muscle_names = sorted(LABELS.values())
    single_means = {m: [] for m in muscle_names}
    multi_means = {m: [] for m in muscle_names}

    for subj, muscles in muscle_data.items():
        for muscle, vals in muscles.items():
            if muscle in single_means:
                single_means[muscle].append(vals["single"])
                multi_means[muscle].append(vals["multi"])

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(muscle_names))
    width = 0.25

    old_vals = [OLD_SMART.get(m, 0) for m in muscle_names]
    s_vals = [np.mean(single_means[m]) if single_means[m] else 0 for m in muscle_names]
    m_vals = [np.mean(multi_means[m]) if multi_means[m] else 0 for m in muscle_names]

    ax.bar(x - width, old_vals, width,
           label=f"Before Fix (mean={np.mean(old_vals):.3f})",
           color="#d62728", alpha=0.8)
    ax.bar(x, s_vals, width,
           label=f"Fixed Single-Atlas (mean={np.mean(s_vals):.3f}, n={len(subjects)})",
           color="#1f77b4", alpha=0.8)
    ax.bar(x + width, m_vals, width,
           label=f"Fixed Multi-Atlas Top-3 (mean={np.mean(m_vals):.3f}, n={len(subjects)})",
           color="#2ca02c", alpha=0.8)

    ax.set_xlabel("Muscle", fontsize=14)
    ax.set_ylabel("Dice Score", fontsize=14)
    ax.set_title(f"Per-Muscle Segmentation: Before vs After Slice Fix ({len(subjects)}/{7} subjects)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(muscle_names, rotation=45, ha="right", fontsize=11)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "partial_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'partial_comparison.png'}")

    return subjects, muscle_data


if __name__ == "__main__":
    single, multi = load_results()

    if single and multi:
        print("Full results available, generating final figures...")
        plot_per_muscle_comparison(single, multi)
        plot_per_subject(single, multi)
        plot_mri_overlay_examples()
    else:
        print("Full results not yet available, using partial results from output...")
        result = plot_partial_results()
        plot_mri_overlay_examples()
