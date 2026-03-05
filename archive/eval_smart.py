#!/usr/bin/env python3
"""Evaluate smart atlas selection: for each target, pick the best-matching atlas
using fast affine NCC ranking, then do full SyN registration."""

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
RESULTS_DIR = Path("results/smart")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_dices(pred, gt):
    dices = {}
    for lid, name in sorted(LABELS.items()):
        p = pred == lid; g = gt == lid
        inter = np.sum(p & g); total = np.sum(p) + np.sum(g)
        if total > 0:
            dices[name] = float(2 * inter / total)
    return dices


def get_subjects():
    subjects = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and "extension" in d.name:
            if (d / "mri.nii.gz").exists() and (d / "labels.nii.gz").exists():
                subjects.append(d)
    return subjects


def rank_atlases_fast(target_mri_ds, atlas_mris_ds, atlas_dirs):
    """Rank atlases by NCC after affine registration (fast, ~7s each)."""
    scores = []
    for atlas_ds, atlas_dir in zip(atlas_mris_ds, atlas_dirs):
        reg = ants.registration(
            fixed=target_mri_ds, moving=atlas_ds,
            type_of_transform="Affine", verbose=False,
        )
        ncc = float(np.corrcoef(
            reg["warpedmovout"].numpy().flatten(),
            target_mri_ds.numpy().flatten()
        )[0, 1])
        scores.append((ncc, atlas_dir))

    scores.sort(reverse=True)
    return scores


def main():
    subjects = get_subjects()
    print(f"Found {len(subjects)} extension subjects")

    # Preload all MRIs at low res for fast ranking
    ranking_spacing = (1.0, 1.0, 3.0)
    working_spacing = (0.5, 0.5, 3.0)

    mris_ds = {}
    mris_full = {}
    labels_full = {}
    for s in subjects:
        mris_full[s] = ants.image_read(str(s / "mri.nii.gz"))
        mris_ds[s] = ants.resample_image(mris_full[s], ranking_spacing, False, 0)
        labels_full[s] = ants.image_read(str(s / "labels.nii.gz"))
        print(f"  Loaded {s.name}")

    all_results = {}
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
        best_ncc, best_dir = ranked[0]
        print(f"  Best atlas: {best_dir.name} (NCC={best_ncc:.4f})")

        # Full SyN registration with best atlas
        print("  Running SyN registration...")
        atlas_w = ants.resample_image(mris_full[best_dir], working_spacing, False, 0)
        target_w = ants.resample_image(mris_full[target_dir], working_spacing, False, 0)
        atlas_matched = ants.histogram_match_image(atlas_w, target_w)

        reg = ants.registration(
            fixed=target_w, moving=atlas_matched,
            type_of_transform="SyN",
            aff_metric="mattes", syn_metric="CC", syn_sampling=4,
            reg_iterations=(100, 70, 50, 10),
            verbose=False,
        )

        warped = ants.apply_transforms(
            fixed=mris_full[target_dir],
            moving=labels_full[best_dir],
            transformlist=reg["fwdtransforms"],
            interpolator="genericLabel",
        )

        elapsed = time.time() - t0
        total_time += elapsed

        pred = warped.numpy()
        gt = labels_full[target_dir].numpy()
        dices = compute_dices(pred, gt)
        mean_dice = np.mean(list(dices.values()))

        print(f"\n  Results ({elapsed:.0f}s, mean_dice={mean_dice:.4f}):")
        for name, dice in sorted(dices.items()):
            print(f"    {name:6s}: {dice:.4f}")

        all_results[target_dir.name] = {
            "best_atlas": best_dir.name,
            "best_ncc": best_ncc,
            "dices": dices,
            "mean_dice": mean_dice,
            "time": elapsed,
        }

        ants.image_write(warped, str(RESULTS_DIR / f"smart_{target_dir.name}.nii.gz"))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Smart Single-Atlas Selection")
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

    # Atlas selections
    print("\nAtlas selections:")
    for subj, data in sorted(all_results.items()):
        print(f"  {subj}: {data['best_atlas']} (NCC={data['best_ncc']:.3f}, Dice={data['mean_dice']:.3f})")

    with open(RESULTS_DIR / "smart_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
