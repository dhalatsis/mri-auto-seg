#!/usr/bin/env python3
"""Generate figures comparing all experiments: atlas baseline, JLF, and nnU-Net."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import LABELS, FIGURES_DIR, DATA_DIR

REPO_ROOT_U = Path(__file__).resolve().parent.parent
ATLAS_RESULTS_DIR = REPO_ROOT_U / "results" / "atlas_baseline"
JLF_RESULTS_DIR = REPO_ROOT_U / "results" / "jlf"
NNUNET_RESULTS_DIR = REPO_ROOT_U / "results" / "nnunet"

# ---------- Load all results ----------

def load_atlas_results():
    """Load atlas-based LOOCV results (single + multi-atlas top-3)."""
    single_path = ATLAS_RESULTS_DIR / "results_single.json"
    multi_path = ATLAS_RESULTS_DIR / "results_multi3.json"
    single = json.load(open(single_path)) if single_path.exists() else {}
    multi = json.load(open(multi_path)) if multi_path.exists() else {}
    return single, multi


REPO_ROOT = REPO_ROOT_U


def load_jlf_results():
    """Load JLF LOOCV results."""
    path = JLF_RESULTS_DIR / "results_all.json"
    if not path.exists():
        return {}
    raw = json.load(open(path))
    # Structure: {method: {subject: {dices: {...}, mean_dice: ...}}}
    # Normalize: some entries have nested dices, others flat
    result = {}
    for method, subjects in raw.items():
        result[method] = {}
        for subj, info in subjects.items():
            if "dices" in info:
                result[method][subj] = info
            else:
                # flat dict of muscle: dice
                dices = {k: v for k, v in info.items() if k in LABELS.values()}
                result[method][subj] = {"dices": dices, "mean_dice": float(np.mean(list(dices.values())))}
    return result


def load_nnunet_results():
    """Load nnU-Net fold 0 results (or compute from prediction)."""
    # Try saved results first
    results_path = NNUNET_RESULTS_DIR / "results_nnunet.json"
    if results_path.exists():
        return json.load(open(results_path))

    # Compute from fold 0 prediction
    pred_path = NNUNET_RESULTS_DIR / "fold_0_pred" / "forearm_001.nii.gz"
    if not pred_path.exists():
        return {}

    remap_path = REPO_ROOT / "nnUNet_data" / "nnUNet_raw" / "Dataset001_ForearmMuscles" / "label_remap.json"
    if not remap_path.exists():
        return {}

    from utils import compute_dices
    with open(remap_path) as f:
        label_remap = json.load(f)
    nn_to_orig = {int(k): v["original_id"] for k, v in label_remap.items()}

    pred = nib.load(str(pred_path)).get_fdata().astype(np.int16)
    remapped = np.zeros_like(pred)
    for nn_id, orig_id in nn_to_orig.items():
        remapped[pred == nn_id] = orig_id

    gt = nib.load(str(DATA_DIR / "sub-07_ext" / "labels.nii.gz")).get_fdata().astype(np.int16)
    dices = compute_dices(remapped, gt)

    return {"sub-07_ext": {"dices": dices, "mean_dice": float(np.mean(list(dices.values())))}}


def aggregate_per_muscle(results_dict, dice_key="dices"):
    """From {subject: {dices: {muscle: val}}} compute per-muscle mean/std."""
    muscle_vals = {}
    for subj, info in results_dict.items():
        dices = info[dice_key] if isinstance(info.get(dice_key), dict) else info.get("dices", {})
        for muscle, val in dices.items():
            muscle_vals.setdefault(muscle, []).append(val)
    means = {m: np.mean(v) for m, v in muscle_vals.items()}
    stds = {m: np.std(v) for m, v in muscle_vals.items()}
    return means, stds


# ---------- Figures ----------

def plot_method_comparison_bars(atlas_single, jlf_data, nnunet_data):
    """Per-muscle grouped bar chart across all methods."""
    muscles = sorted(LABELS.values())

    # Atlas single
    a_means, a_stds = aggregate_per_muscle(atlas_single)
    # JLF top-3
    jlf3 = jlf_data.get("jlf_3", {})
    j_means, j_stds = aggregate_per_muscle(jlf3)
    # nnU-Net
    n_means, n_stds = aggregate_per_muscle(nnunet_data)

    fig, ax = plt.subplots(figsize=(18, 8))
    x = np.arange(len(muscles))
    width = 0.25

    a_vals = [a_means.get(m, 0) for m in muscles]
    j_vals = [j_means.get(m, 0) for m in muscles]
    n_vals = [n_means.get(m, 0) for m in muscles]

    ax.bar(x - width, a_vals, width,
           label=f"Atlas Single (mean={np.mean(a_vals):.3f})",
           color="#1f77b4", alpha=0.85)
    ax.bar(x, j_vals, width,
           label=f"JLF Top-3 (mean={np.mean(j_vals):.3f})",
           color="#ff7f0e", alpha=0.85)
    ax.bar(x + width, n_vals, width,
           label=f"nnU-Net fold 0 (mean={np.mean(n_vals):.3f})",
           color="#2ca02c", alpha=0.85)

    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(len(muscles) - 0.5, 0.91, "Target: 0.90", color="red", fontsize=10, ha="right")

    ax.set_xlabel("Muscle", fontsize=13)
    ax.set_ylabel("Dice Score", fontsize=13)
    ax.set_title("Per-Muscle Dice: Atlas vs JLF vs nnU-Net", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(muscles, rotation=45, ha="right", fontsize=11)
    ax.legend(fontsize=12, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "comparison" / "per_muscle_all_methods.png"
    fig.savefig(str(out), dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_method_summary(atlas_single, atlas_multi, jlf_data, nnunet_data):
    """Overall mean Dice comparison across methods (horizontal bar chart)."""
    methods = {}

    # Atlas methods
    if atlas_single:
        vals = [v["mean_dice"] for v in atlas_single.values()]
        methods["Atlas Single"] = (np.mean(vals), np.std(vals), len(vals))
    if atlas_multi:
        vals = [v["mean_dice"] for v in atlas_multi.values()]
        methods["Atlas Multi-3"] = (np.mean(vals), np.std(vals), len(vals))

    # JLF methods
    for key, label in [("single", "JLF Single"), ("majority_3", "Majority Top-3"),
                       ("majority_5", "Majority Top-5"), ("jlf_3", "JLF Top-3"),
                       ("jlf_5", "JLF Top-5")]:
        if key in jlf_data:
            vals = [v["mean_dice"] for v in jlf_data[key].values()]
            methods[label] = (np.mean(vals), np.std(vals), len(vals))

    # nnU-Net
    if nnunet_data:
        vals = [v["mean_dice"] for v in nnunet_data.values()]
        methods["nnU-Net (fold 0, 42%)"] = (np.mean(vals), np.std(vals), len(vals))

    if not methods:
        return

    # Sort by mean
    sorted_methods = sorted(methods.items(), key=lambda x: x[1][0])
    names = [m[0] for m in sorted_methods]
    means = [m[1][0] for m in sorted_methods]
    stds = [m[1][1] for m in sorted_methods]
    ns = [m[1][2] for m in sorted_methods]

    # Color by category
    colors = []
    for name in names:
        if "nnU-Net" in name:
            colors.append("#2ca02c")
        elif "JLF" in name or "Majority" in name:
            colors.append("#ff7f0e")
        else:
            colors.append("#1f77b4")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), means, xerr=stds, height=0.6,
                   color=colors, alpha=0.85, capsize=4)

    for i, (mean, n) in enumerate(zip(means, ns)):
        ax.text(mean + 0.01, i, f"{mean:.3f} (n={n})", va="center", fontsize=11)

    ax.axvline(x=0.9, color="red", linestyle="--", alpha=0.5)
    ax.text(0.905, len(names) - 0.5, "Target", color="red", fontsize=10)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlabel("Mean Dice Score", fontsize=13)
    ax.set_title("Method Comparison — Mean Dice (LOOCV)", fontsize=16)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "comparison" / "method_summary.png"
    fig.savefig(str(out), dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_nnunet_detail(nnunet_data):
    """nnU-Net per-muscle Dice with gap-to-target analysis."""
    if not nnunet_data:
        return

    muscles = sorted(LABELS.values())
    n_means, _ = aggregate_per_muscle(nnunet_data)
    vals = [n_means.get(m, 0) for m in muscles]
    gaps = [0.9 - v for v in vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left: per-muscle Dice
    colors = ["#2ca02c" if v >= 0.85 else "#ff7f0e" if v >= 0.7 else "#d62728" for v in vals]
    bars = ax1.bar(range(len(muscles)), vals, color=colors, alpha=0.85)
    ax1.axhline(y=0.9, color="red", linestyle="--", alpha=0.5)
    ax1.axhline(y=0.8, color="orange", linestyle="--", alpha=0.3)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, rotation=45)

    ax1.set_xticks(range(len(muscles)))
    ax1.set_xticklabels(muscles, rotation=45, ha="right", fontsize=11)
    ax1.set_ylabel("Dice Score", fontsize=13)
    ax1.set_title("nnU-Net Per-Muscle Dice (fold 0, epoch ~416)", fontsize=14)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    # Right: gap to target
    gap_colors = ["#d62728" if g > 0.1 else "#ff7f0e" if g > 0 else "#2ca02c" for g in gaps]
    ax2.barh(range(len(muscles)), gaps, color=gap_colors, alpha=0.85)
    ax2.set_yticks(range(len(muscles)))
    ax2.set_yticklabels(muscles, fontsize=11)
    ax2.set_xlabel("Gap to 0.90 Target", fontsize=13)
    ax2.set_title("Gap Analysis — Distance to Target Dice", fontsize=14)
    ax2.axvline(x=0, color="green", linewidth=2)
    ax2.grid(axis="x", alpha=0.3)

    for i, g in enumerate(gaps):
        ax2.text(g + 0.005, i, f"{g:+.3f}", va="center", fontsize=10)

    plt.tight_layout()
    out = FIGURES_DIR / "nnunet" / "nnunet_fold0_detail.png"
    fig.savefig(str(out), dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_nnunet_vs_atlas_subject(atlas_single, nnunet_data):
    """Side-by-side comparison on sub-07_ext (both have this subject)."""
    subj = "sub-07_ext"
    if subj not in atlas_single or subj not in nnunet_data:
        return

    muscles = sorted(LABELS.values())
    atlas_dices = atlas_single[subj]["dices"]
    nn_dices = nnunet_data[subj]["dices"]

    a_vals = [atlas_dices.get(m, 0) for m in muscles]
    n_vals = [nn_dices.get(m, 0) for m in muscles]
    improvements = [n - a for n, a in zip(n_vals, a_vals)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [2, 1]})

    x = np.arange(len(muscles))
    width = 0.35

    ax1.bar(x - width/2, a_vals, width, label=f"Atlas (mean={np.mean(a_vals):.3f})",
            color="#1f77b4", alpha=0.85)
    ax1.bar(x + width/2, n_vals, width, label=f"nnU-Net (mean={np.mean(n_vals):.3f})",
            color="#2ca02c", alpha=0.85)
    ax1.axhline(y=0.9, color="red", linestyle="--", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(muscles, rotation=45, ha="right", fontsize=11)
    ax1.set_ylabel("Dice Score", fontsize=13)
    ax1.set_title(f"Atlas vs nnU-Net — {subj}", fontsize=16)
    ax1.legend(fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    # Improvement bars
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in improvements]
    ax2.bar(x, improvements, color=colors, alpha=0.8)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(muscles, rotation=45, ha="right", fontsize=11)
    ax2.set_ylabel("Improvement", fontsize=13)
    ax2.set_title("nnU-Net Improvement over Atlas", fontsize=14)
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(improvements):
        ax2.text(i, v + 0.005 if v >= 0 else v - 0.02, f"{v:+.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "comparison" / "atlas_vs_nnunet_subject10.png"
    fig.savefig(str(out), dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_nnunet_overlay(nnunet_data):
    """Plot MRI slices with nnU-Net prediction overlay vs GT."""
    pred_path = NNUNET_RESULTS_DIR / "fold_0_pred" / "forearm_001.nii.gz"
    remap_path = REPO_ROOT / "nnUNet_data" / "nnUNet_raw" / "Dataset001_ForearmMuscles" / "label_remap.json"
    if not pred_path.exists() or not remap_path.exists():
        return

    with open(remap_path) as f:
        label_remap = json.load(f)
    nn_to_orig = {int(k): v["original_id"] for k, v in label_remap.items()}

    pred_raw = nib.load(str(pred_path)).get_fdata().astype(np.int16)
    pred = np.zeros_like(pred_raw)
    for nn_id, orig_id in nn_to_orig.items():
        pred[pred_raw == nn_id] = orig_id

    mri = nib.load(str(DATA_DIR / "sub-07_ext" / "mri.nii.gz")).get_fdata()
    gt = nib.load(str(DATA_DIR / "sub-07_ext" / "labels.nii.gz")).get_fdata().astype(np.int16)

    # Find slices with labels
    label_slices = np.where(np.any(gt > 0, axis=(0, 1)))[0]
    n_show = min(6, len(label_slices))
    indices = np.linspace(0, len(label_slices) - 1, n_show, dtype=int)
    show_slices = label_slices[indices]

    cmap = plt.cm.tab20(np.linspace(0, 1, 20))

    fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 12))
    fig.suptitle("nnU-Net Segmentation — sub-07_ext (fold 0, epoch ~416)", fontsize=16, y=0.98)

    for col, sl in enumerate(show_slices):
        mri_slice = mri[:, :, sl]
        mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)

        gt_slice = gt[:, :, sl]
        pred_slice = pred[:, :, sl]

        # MRI
        axes[0, col].imshow(mri_norm, cmap="gray")
        axes[0, col].set_title(f"Slice {sl}", fontsize=11)
        axes[0, col].axis("off")

        # GT overlay
        gt_rgb = np.stack([mri_norm] * 3, axis=-1)
        for lid in np.unique(gt_slice):
            if lid == 0:
                continue
            mask = gt_slice == lid
            color = cmap[int(lid) % 20][:3]
            for c in range(3):
                gt_rgb[:, :, c][mask] = mri_norm[mask] * 0.4 + color[c] * 0.6
        axes[1, col].imshow(gt_rgb)
        axes[1, col].axis("off")

        # Prediction overlay
        pred_rgb = np.stack([mri_norm] * 3, axis=-1)
        for lid in np.unique(pred_slice):
            if lid == 0:
                continue
            mask = pred_slice == lid
            color = cmap[int(lid) % 20][:3]
            for c in range(3):
                pred_rgb[:, :, c][mask] = mri_norm[mask] * 0.4 + color[c] * 0.6
        axes[2, col].imshow(pred_rgb)
        axes[2, col].axis("off")

    axes[0, 0].set_ylabel("MRI", fontsize=14, rotation=0, labelpad=50)
    axes[1, 0].set_ylabel("Ground Truth", fontsize=14, rotation=0, labelpad=80)
    axes[2, 0].set_ylabel("nnU-Net Pred", fontsize=14, rotation=0, labelpad=80)

    plt.tight_layout()
    out = FIGURES_DIR / "nnunet" / "nnunet_overlay_subject10.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_roadmap_progress():
    """Visual summary of progress toward 0.90 target."""
    stages = [
        ("Baseline\n(before fix)", 0.394, "#d62728"),
        ("Atlas Single\n(after fix)", 0.658, "#1f77b4"),
        ("JLF Top-3", 0.517, "#ff7f0e"),
        ("nnU-Net\n(fold 0, 42%)", 0.804, "#2ca02c"),
        ("Target", 0.90, "#9467bd"),
    ]

    fig, ax = plt.subplots(figsize=(14, 5))

    names = [s[0] for s in stages]
    vals = [s[1] for s in stages]
    colors = [s[2] for s in stages]

    bars = ax.bar(range(len(stages)), vals, color=colors, alpha=0.85, width=0.6,
                  edgecolor="black", linewidth=0.5)

    for i, (name, val) in enumerate(zip(names, vals)):
        ax.text(i, val + 0.015, f"{val:.3f}", ha="center", fontsize=13, fontweight="bold")

    ax.axhline(y=0.9, color="#9467bd", linestyle="--", alpha=0.7, linewidth=2)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Mean Dice Score", fontsize=14)
    ax.set_title("Segmentation Accuracy — Roadmap Progress", fontsize=18)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Arrow showing progress
    ax.annotate("", xy=(3, 0.804), xytext=(1, 0.658),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.text(2, 0.72, "+0.146", fontsize=12, color="green", ha="center", fontweight="bold")

    plt.tight_layout()
    out = FIGURES_DIR / "comparison" / "roadmap_progress.png"
    fig.savefig(str(out), dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Loading results...")
    atlas_single, atlas_multi = load_atlas_results()
    jlf_data = load_jlf_results()
    nnunet_data = load_nnunet_results()

    print(f"  Atlas single: {len(atlas_single)} subjects")
    print(f"  Atlas multi:  {len(atlas_multi)} subjects")
    print(f"  JLF methods:  {list(jlf_data.keys())}")
    print(f"  nnU-Net:      {len(nnunet_data)} subjects")

    print("\nGenerating figures...")
    plot_method_comparison_bars(atlas_single, jlf_data, nnunet_data)
    plot_method_summary(atlas_single, atlas_multi, jlf_data, nnunet_data)
    plot_nnunet_detail(nnunet_data)
    plot_nnunet_vs_atlas_subject(atlas_single, nnunet_data)
    plot_nnunet_overlay(nnunet_data)
    plot_roadmap_progress()

    print("\nDone! Figures saved to:")
    print(f"  {FIGURES_DIR / 'comparison/'}")
    print(f"  {FIGURES_DIR / 'nnunet/'}")
