#!/usr/bin/env python3
"""Evaluate on fixed (corrected slice mapping) data.

Runs two modes:
1. Smart single-atlas: rank by affine NCC, use best atlas
2. Smart multi-atlas (top-3): rank by affine NCC, use top 3, majority vote
"""

import sys
from pathlib import Path

# Add repo root to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import ants
import time
import json
import numpy as np

from utils import (
    LABELS, compute_dices, majority_vote, rank_atlases_fast,
    get_subjects, RESULTS_DIR,
)
from utils.registration import register_pair

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    subjects = get_subjects()
    print(f"Found {len(subjects)} extension subjects")

    ranking_spacing = (1.0, 1.0, 3.0)
    working_spacing = (0.5, 0.5, 3.0)

    # Preload
    mris_ds = {}
    mris_full = {}
    labels_full = {}
    for s in subjects:
        mris_full[s] = ants.image_read(str(s / "mri.nii.gz"))
        mris_ds[s] = ants.resample_image(mris_full[s], ranking_spacing, False, 0)
        labels_full[s] = ants.image_read(str(s / "labels.nii.gz"))
        print(f"  Loaded {s.name}")

    results_single = {}
    results_multi3 = {}
    total_time = 0

    for i, target_dir in enumerate(subjects):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(subjects)}] Target: {target_dir.name}")
        print(f"{'='*60}")

        atlas_dirs = [d for d in subjects if d != target_dir]
        atlas_ds_list = [mris_ds[d] for d in atlas_dirs]

        t0 = time.time()

        # Rank atlases
        print("  Ranking atlases...")
        ranked = rank_atlases_fast(mris_ds[target_dir], atlas_ds_list, atlas_dirs)
        for ncc, d in ranked:
            print(f"    {d.name}: NCC={ncc:.4f}")

        # Register top 3 atlases (we need all 3 for multi; single uses just #1)
        top3 = ranked[:3]
        warped_arrays = []

        for j, (ncc, atlas_dir) in enumerate(top3):
            print(f"  Registering atlas {j+1}/3: {atlas_dir.name} (NCC={ncc:.4f})...")
            at = time.time()
            warped = register_pair(
                mris_full[atlas_dir], labels_full[atlas_dir],
                mris_full[target_dir], working_spacing
            )
            warped_arrays.append(warped.numpy())
            print(f"    Done in {time.time()-at:.0f}s")

        elapsed = time.time() - t0
        total_time += elapsed

        gt = labels_full[target_dir].numpy()

        # Single-atlas result (best atlas only)
        pred_single = warped_arrays[0]
        dices_single = compute_dices(pred_single, gt)
        mean_single = np.mean(list(dices_single.values()))

        # Multi-atlas result (top 3 majority vote)
        pred_multi = majority_vote(warped_arrays)
        dices_multi = compute_dices(pred_multi, gt)
        mean_multi = np.mean(list(dices_multi.values()))

        print(f"\n  Single-atlas (best): mean_dice={mean_single:.4f}")
        print(f"  Multi-atlas (top-3): mean_dice={mean_multi:.4f}")
        print(f"  Time: {elapsed:.0f}s")

        print(f"\n  {'Muscle':6s} {'Single':>8s} {'Multi3':>8s}")
        print(f"  {'-'*24}")
        for name in sorted(set(list(dices_single.keys()) + list(dices_multi.keys()))):
            s = dices_single.get(name, 0)
            m = dices_multi.get(name, 0)
            print(f"  {name:6s} {s:8.4f} {m:8.4f}")

        results_single[target_dir.name] = {
            "best_atlas": top3[0][1].name,
            "best_ncc": top3[0][0],
            "dices": dices_single,
            "mean_dice": mean_single,
            "time": elapsed,
        }
        results_multi3[target_dir.name] = {
            "atlases": [d.name for _, d in top3],
            "dices": dices_multi,
            "mean_dice": mean_multi,
            "time": elapsed,
        }

        # Save predictions
        target_img = mris_full[target_dir]
        single_img = target_img.new_image_like(pred_single.astype(np.float32))
        multi_img = target_img.new_image_like(pred_multi.astype(np.float32))
        ants.image_write(single_img, str(RESULTS_DIR / f"single_{target_dir.name}.nii.gz"))
        ants.image_write(multi_img, str(RESULTS_DIR / f"multi3_{target_dir.name}.nii.gz"))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for mode_name, results in [("Smart Single-Atlas", results_single),
                                ("Smart Multi-Atlas (top-3)", results_multi3)]:
        muscle_dices = {}
        for subj, data in results.items():
            for muscle, dice in data["dices"].items():
                muscle_dices.setdefault(muscle, []).append(dice)

        print(f"\n--- {mode_name} ---")
        print(f"{'Muscle':8s} {'Mean':>8s} {'Std':>8s}")
        print("-" * 26)
        overall = []
        for muscle in sorted(muscle_dices.keys()):
            vals = muscle_dices[muscle]
            overall.extend(vals)
            print(f"{muscle:8s} {np.mean(vals):8.4f} {np.std(vals):8.4f}")
        print("-" * 26)
        print(f"{'OVERALL':8s} {np.mean(overall):8.4f}")

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    with open(RESULTS_DIR / "results_single.json", "w") as f:
        json.dump(results_single, f, indent=2)
    with open(RESULTS_DIR / "results_multi3.json", "w") as f:
        json.dump(results_multi3, f, indent=2)


if __name__ == "__main__":
    main()
