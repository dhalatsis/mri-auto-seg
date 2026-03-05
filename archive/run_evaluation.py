#!/usr/bin/env python3
"""Run comprehensive evaluation of the atlas-based segmentation pipeline.

Tests single-atlas and multi-atlas approaches with various parameters.
Uses the extension subjects for evaluation.
"""

import ants
import time
import json
import numpy as np
from pathlib import Path

LABELS = {
    2: "ANC", 3: "APL", 4: "ECRB", 5: "ECRL", 6: "ECU",
    7: "ED", 8: "EDM", 9: "EPL", 10: "FCR", 11: "FCU",
    12: "FDP", 14: "FDS", 15: "FPL", 16: "PL", 17: "PQ",
    18: "PT", 19: "SUP",
}

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Working resolution for registration (original is 0.27mm)
WORKING_SPACING = (0.5, 0.5, 3.0)


def compute_dices(pred, gt):
    """Compute per-label Dice scores."""
    dices = {}
    for lid, name in sorted(LABELS.items()):
        p = pred == lid
        g = gt == lid
        inter = np.sum(p & g)
        total = np.sum(p) + np.sum(g)
        if total > 0:
            dices[name] = float(2.0 * inter / total)
    return dices


def register_pair(atlas_mri, atlas_labels, target_mri, working_spacing=WORKING_SPACING):
    """Register atlas to target and warp labels."""
    atlas_w = ants.resample_image(atlas_mri, working_spacing, False, 0)
    target_w = ants.resample_image(target_mri, working_spacing, False, 0)

    reg = ants.registration(
        fixed=target_w, moving=atlas_w,
        type_of_transform="SyN", verbose=False,
    )

    warped_labels = ants.apply_transforms(
        fixed=target_mri, moving=atlas_labels,
        transformlist=reg["fwdtransforms"],
        interpolator="genericLabel",
    )
    warped_mri = ants.apply_transforms(
        fixed=target_mri, moving=atlas_mri,
        transformlist=reg["fwdtransforms"],
        interpolator="linear",
    )
    return warped_labels, warped_mri


def majority_vote(label_arrays):
    """Majority voting across label arrays."""
    stacked = np.stack(label_arrays, axis=0)
    all_labels = np.unique(stacked)
    all_labels = all_labels[all_labels > 0]

    result = np.zeros(label_arrays[0].shape, dtype=np.float32)
    if len(all_labels) == 0:
        return result

    vote_counts = np.zeros((len(all_labels),) + label_arrays[0].shape, dtype=np.int32)
    for i, lab in enumerate(all_labels):
        vote_counts[i] = np.sum(stacked == lab, axis=0)

    max_votes = np.max(vote_counts, axis=0)
    best_idx = np.argmax(vote_counts, axis=0)
    mask = max_votes > 0
    result[mask] = all_labels[best_idx[mask]]
    return result


def get_extension_subjects():
    """Get all extension subject directories."""
    subjects = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and "extension" in d.name:
            mri = d / "mri.nii.gz"
            lab = d / "labels.nii.gz"
            if mri.exists() and lab.exists():
                subjects.append(d)
    return subjects


def run_single_atlas_loocv():
    """Leave-one-out with single best atlas (nearest neighbor by first atlas)."""
    subjects = get_extension_subjects()
    print(f"\n{'='*60}")
    print(f"Single-Atlas LOOCV ({len(subjects)} subjects)")
    print(f"{'='*60}")

    all_dices = {}

    for i, target_dir in enumerate(subjects):
        target_name = target_dir.name
        target_mri = ants.image_read(str(target_dir / "mri.nii.gz"))
        gt = ants.image_read(str(target_dir / "labels.nii.gz")).numpy()

        # Use first available atlas (not self)
        atlas_dir = [d for d in subjects if d != target_dir][0]
        atlas_name = atlas_dir.name

        print(f"\n[{i+1}/{len(subjects)}] {atlas_name} -> {target_name}")

        atlas_mri = ants.image_read(str(atlas_dir / "mri.nii.gz"))
        atlas_labels = ants.image_read(str(atlas_dir / "labels.nii.gz"))

        t0 = time.time()
        warped_labels, _ = register_pair(atlas_mri, atlas_labels, target_mri)
        elapsed = time.time() - t0

        pred = warped_labels.numpy()
        dices = compute_dices(pred, gt)
        valid = [v for v in dices.values()]
        mean_dice = np.mean(valid) if valid else 0

        print(f"  Time: {elapsed:.0f}s, Mean Dice: {mean_dice:.4f}")
        all_dices[target_name] = dices

        ants.image_write(warped_labels,
                         str(RESULTS_DIR / f"single_{target_name}.nii.gz"))

    return all_dices


def run_multi_atlas_loocv():
    """Leave-one-out with all remaining atlases + majority voting."""
    subjects = get_extension_subjects()
    print(f"\n{'='*60}")
    print(f"Multi-Atlas LOOCV ({len(subjects)} subjects, majority voting)")
    print(f"{'='*60}")

    all_dices = {}

    for i, target_dir in enumerate(subjects):
        target_name = target_dir.name
        target_mri = ants.image_read(str(target_dir / "mri.nii.gz"))
        gt = ants.image_read(str(target_dir / "labels.nii.gz")).numpy()

        atlas_dirs = [d for d in subjects if d != target_dir]
        print(f"\n[{i+1}/{len(subjects)}] Target: {target_name} ({len(atlas_dirs)} atlases)")

        warped_label_arrays = []
        t0 = time.time()

        for j, atlas_dir in enumerate(atlas_dirs):
            atlas_mri = ants.image_read(str(atlas_dir / "mri.nii.gz"))
            atlas_labels = ants.image_read(str(atlas_dir / "labels.nii.gz"))

            warped, _ = register_pair(atlas_mri, atlas_labels, target_mri)
            warped_label_arrays.append(warped.numpy())
            print(f"  Atlas {j+1}/{len(atlas_dirs)}: {atlas_dir.name} done")

        # Majority vote
        fused = majority_vote(warped_label_arrays)
        elapsed = time.time() - t0

        dices = compute_dices(fused, gt)
        valid = [v for v in dices.values()]
        mean_dice = np.mean(valid) if valid else 0

        print(f"  Time: {elapsed:.0f}s, Mean Dice: {mean_dice:.4f}")
        all_dices[target_name] = dices

        # Save
        fused_img = target_mri.new_image_like(fused)
        ants.image_write(fused_img,
                         str(RESULTS_DIR / f"multi_{target_name}.nii.gz"))

    return all_dices


def print_summary(results, title):
    """Print a summary table of results."""
    print(f"\n{'='*60}")
    print(f"Summary: {title}")
    print(f"{'='*60}")

    # Aggregate per muscle
    muscle_dices = {}
    for subj, dices in results.items():
        for muscle, dice in dices.items():
            muscle_dices.setdefault(muscle, []).append(dice)

    print(f"\n{'Muscle':8s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print("-" * 42)
    overall = []
    for muscle in sorted(muscle_dices.keys()):
        vals = muscle_dices[muscle]
        overall.extend(vals)
        print(f"{muscle:8s} {np.mean(vals):8.4f} {np.std(vals):8.4f} "
              f"{np.min(vals):8.4f} {np.max(vals):8.4f}")
    print("-" * 42)
    print(f"{'OVERALL':8s} {np.mean(overall):8.4f} {np.std(overall):8.4f}")

    return np.mean(overall)


if __name__ == "__main__":
    import sys

    subjects = get_extension_subjects()
    print(f"Found {len(subjects)} extension subjects: {[s.name for s in subjects]}")

    if "--multi" in sys.argv:
        results = run_multi_atlas_loocv()
        mean = print_summary(results, "Multi-Atlas LOOCV (Majority Voting)")
    else:
        results = run_single_atlas_loocv()
        mean = print_summary(results, "Single-Atlas LOOCV")

    with open(RESULTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'evaluation_results.json'}")
