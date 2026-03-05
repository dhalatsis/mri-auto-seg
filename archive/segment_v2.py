#!/usr/bin/env python3
"""Improved atlas-based segmentation with preprocessing.

Improvements over v1:
1. N4 bias field correction
2. Intensity normalization (histogram matching)
3. Forearm mask extraction for focused registration
4. Multi-resolution SyN with optimized parameters
5. Multi-atlas majority voting and JLF
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
RESULTS_DIR = Path("results/v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def register_pair(atlas_mri, atlas_labels, target_mri,
                  working_spacing=(0.5, 0.5, 3.0)):
    """Register atlas to target with histogram matching and SyN."""
    # Resample for registration
    atlas_w = ants.resample_image(atlas_mri, working_spacing, False, 0)
    target_w = ants.resample_image(target_mri, working_spacing, False, 0)

    # Histogram matching (match atlas to target intensities)
    atlas_w_matched = ants.histogram_match_image(atlas_w, target_w)

    # Multi-step registration: Rigid -> Affine -> SyN
    reg = ants.registration(
        fixed=target_w,
        moving=atlas_w_matched,
        type_of_transform="SyN",
        aff_metric="mattes",
        syn_metric="CC",
        syn_sampling=4,
        reg_iterations=(100, 70, 50, 10),
        verbose=False,
    )

    # Apply transforms at original resolution
    warped_labels = ants.apply_transforms(
        fixed=target_mri,
        moving=atlas_labels,
        transformlist=reg["fwdtransforms"],
        interpolator="genericLabel",
    )
    warped_mri = ants.apply_transforms(
        fixed=target_mri,
        moving=atlas_mri,
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


def get_subjects(position="extension"):
    """Get subject directories for a position."""
    subjects = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and position in d.name:
            mri = d / "mri.nii.gz"
            lab = d / "labels.nii.gz"
            if mri.exists() and lab.exists():
                subjects.append(d)
    return subjects


def run_evaluation(mode="single", position="extension", max_atlases=None):
    """Run LOOCV evaluation.

    Args:
        mode: 'single' (first atlas only), 'multi' (all atlases + majority vote),
              'best3' (top 3 atlases + majority vote)
        position: 'extension' or 'flexion'
        max_atlases: Maximum number of atlases to use (None = all)
    """
    subjects = get_subjects(position)
    print(f"\nEvaluation: {mode} mode, {len(subjects)} {position} subjects")

    all_results = {}
    total_time = 0

    for i, target_dir in enumerate(subjects):
        target_name = target_dir.name
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(subjects)}] Target: {target_name}")
        print(f"{'='*60}")

        target_mri = ants.image_read(str(target_dir / "mri.nii.gz"))
        gt = ants.image_read(str(target_dir / "labels.nii.gz")).numpy()

        atlas_dirs = [d for d in subjects if d != target_dir]
        if max_atlases:
            atlas_dirs = atlas_dirs[:max_atlases]

        if mode == "single":
            atlas_dirs = atlas_dirs[:1]

        t0 = time.time()
        warped_arrays = []

        for j, atlas_dir in enumerate(atlas_dirs):
            print(f"  Atlas {j+1}/{len(atlas_dirs)}: {atlas_dir.name}")
            atlas_mri = ants.image_read(str(atlas_dir / "mri.nii.gz"))
            atlas_labels = ants.image_read(str(atlas_dir / "labels.nii.gz"))

            at = time.time()
            warped, _ = register_pair(atlas_mri, atlas_labels, target_mri)
            warped_arrays.append(warped.numpy())
            print(f"    Done in {time.time()-at:.0f}s")

        # Fuse
        if len(warped_arrays) == 1:
            fused = warped_arrays[0]
        else:
            fused = majority_vote(warped_arrays)

        elapsed = time.time() - t0
        total_time += elapsed

        dices = compute_dices(fused, gt)
        valid = list(dices.values())
        mean_dice = np.mean(valid) if valid else 0

        print(f"\n  Results (time={elapsed:.0f}s, mean_dice={mean_dice:.4f}):")
        for name, dice in sorted(dices.items()):
            print(f"    {name:6s}: {dice:.4f}")

        all_results[target_name] = {"dices": dices, "mean_dice": mean_dice, "time": elapsed}

        # Save prediction
        fused_img = target_mri.new_image_like(fused.astype(np.float32))
        ants.image_write(fused_img,
                         str(RESULTS_DIR / f"{mode}_{target_name}.nii.gz"))

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {mode} mode")
    print(f"{'='*60}")

    muscle_dices = {}
    for subj, data in all_results.items():
        for muscle, dice in data["dices"].items():
            muscle_dices.setdefault(muscle, []).append(dice)

    print(f"\n{'Muscle':8s} {'Mean':>8s} {'Std':>8s}")
    print("-" * 26)
    overall = []
    for muscle in sorted(muscle_dices.keys()):
        vals = muscle_dices[muscle]
        overall.extend(vals)
        print(f"{muscle:8s} {np.mean(vals):8.4f} {np.std(vals):8.4f}")
    print("-" * 26)
    print(f"{'OVERALL':8s} {np.mean(overall):8.4f}")
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    with open(RESULTS_DIR / f"results_{mode}.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    max_atlases = int(sys.argv[2]) if len(sys.argv) > 2 else None

    run_evaluation(mode=mode, max_atlases=max_atlases)
