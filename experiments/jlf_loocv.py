#!/usr/bin/env python3
"""JLF (Joint Label Fusion) LOOCV experiment.

Compares three fusion strategies using the same top-K registered atlases:
1. Single-atlas (best match only) — current baseline
2. Majority voting (top-K) — current multi-atlas baseline
3. Joint Label Fusion (top-K) — the new method

Also tests different values of K (number of atlases) for JLF.

Usage:
    python3 -u experiments/jlf_loocv.py 2>&1 | tee results/jlf/jlf_loocv.log
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import ants
import time
import json
import numpy as np

from utils import (
    LABELS, compute_dices, majority_vote, rank_atlases_fast,
    get_subjects, RESULTS_DIR, REPO_ROOT,
)
from utils.registration import register_pair

# Use all available cores for JLF C++ backend
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "6")

JLF_RESULTS_DIR = REPO_ROOT / "results" / "jlf"
JLF_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_mask(target_mri):
    """Create a foreground mask for JLF (nonzero voxels with some margin)."""
    arr = target_mri.numpy()
    # Use Otsu-like threshold: anything above 10% of max
    threshold = arr.max() * 0.05
    mask_arr = (arr > threshold).astype(np.float32)
    return target_mri.new_image_like(mask_arr)


def run_jlf(target_mri, warped_mri_list, warped_label_list, beta=4, rad=2):
    """Run Joint Label Fusion on pre-registered atlases."""
    mask = create_mask(target_mri)

    result = ants.joint_label_fusion(
        target_image=target_mri,
        target_image_mask=mask,
        atlas_list=warped_mri_list,
        label_list=warped_label_list,
        beta=beta,
        rad=rad,
        usecor=False,
        verbose=False,
    )
    return result["segmentation"]


def register_pair_full(atlas_mri, atlas_labels, target_mri, working_spacing):
    """Register atlas to target and return warped MRI + warped labels.

    Unlike utils.register_pair which only returns warped labels,
    JLF also needs the warped intensity image.
    """
    atlas_w = ants.resample_image(atlas_mri, working_spacing, False, 0)
    target_w = ants.resample_image(target_mri, working_spacing, False, 0)
    atlas_matched = ants.histogram_match_image(atlas_w, target_w)

    reg = ants.registration(
        fixed=target_w, moving=atlas_matched,
        type_of_transform="SyN",
        aff_metric="mattes", syn_metric="CC", syn_sampling=4,
        reg_iterations=(100, 70, 50, 10),
        verbose=False,
    )

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

    return warped_mri, warped_labels


def main():
    subjects = get_subjects()
    print(f"Found {len(subjects)} extension subjects")

    ranking_spacing = (1.0, 1.0, 3.0)
    working_spacing = (0.5, 0.5, 3.0)
    TOP_K = 5  # Register top 5 atlases (test JLF with 3, 4, 5)

    # Preload all images
    print("\nPreloading images...")
    mris_ds = {}
    mris_full = {}
    labels_full = {}
    for s in subjects:
        mris_full[s] = ants.image_read(str(s / "mri.nii.gz"))
        mris_ds[s] = ants.resample_image(mris_full[s], ranking_spacing, False, 0)
        labels_full[s] = ants.image_read(str(s / "labels.nii.gz"))
        print(f"  Loaded {s.name}")

    # Results containers
    results = {
        "single": {},
        "majority_3": {},
        "majority_5": {},
        "jlf_3": {},
        "jlf_5": {},
    }

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

        # Register top K atlases (need both warped MRI and labels for JLF)
        top_k = ranked[:TOP_K]
        warped_mris = []
        warped_labels_arrays = []
        warped_labels_ants = []
        warped_mris_ants = []

        for j, (ncc, atlas_dir) in enumerate(top_k):
            print(f"  Registering atlas {j+1}/{TOP_K}: {atlas_dir.name} (NCC={ncc:.4f})...")
            at = time.time()

            w_mri, w_labels = register_pair_full(
                mris_full[atlas_dir], labels_full[atlas_dir],
                mris_full[target_dir], working_spacing
            )
            warped_mris_ants.append(w_mri)
            warped_labels_ants.append(w_labels)
            warped_mris.append(w_mri)
            warped_labels_arrays.append(w_labels.numpy())
            print(f"    Done in {time.time()-at:.0f}s")

        elapsed_reg = time.time() - t0

        gt = labels_full[target_dir].numpy()

        # === Method 1: Single-atlas (best match) ===
        pred_single = warped_labels_arrays[0]
        dices_single = compute_dices(pred_single, gt)
        mean_single = np.mean(list(dices_single.values()))

        # === Method 2: Majority voting top-3 ===
        pred_mv3 = majority_vote(warped_labels_arrays[:3])
        dices_mv3 = compute_dices(pred_mv3, gt)
        mean_mv3 = np.mean(list(dices_mv3.values()))

        # === Method 3: Majority voting top-5 ===
        pred_mv5 = majority_vote(warped_labels_arrays[:5])
        dices_mv5 = compute_dices(pred_mv5, gt)
        mean_mv5 = np.mean(list(dices_mv5.values()))

        # === Method 4: JLF top-3 ===
        print(f"  Running JLF (top-3)...")
        jt = time.time()
        try:
            pred_jlf3_img = run_jlf(
                mris_full[target_dir],
                warped_mris_ants[:3],
                warped_labels_ants[:3],
            )
            pred_jlf3 = pred_jlf3_img.numpy()
            dices_jlf3 = compute_dices(pred_jlf3, gt)
            mean_jlf3 = np.mean(list(dices_jlf3.values()))
            print(f"    JLF top-3 done in {time.time()-jt:.0f}s")
        except Exception as e:
            print(f"    JLF top-3 FAILED: {e}")
            pred_jlf3 = pred_mv3  # fallback
            dices_jlf3 = dices_mv3
            mean_jlf3 = mean_mv3

        # === Method 5: JLF top-5 ===
        print(f"  Running JLF (top-5)...")
        jt = time.time()
        try:
            pred_jlf5_img = run_jlf(
                mris_full[target_dir],
                warped_mris_ants[:5],
                warped_labels_ants[:5],
            )
            pred_jlf5 = pred_jlf5_img.numpy()
            dices_jlf5 = compute_dices(pred_jlf5, gt)
            mean_jlf5 = np.mean(list(dices_jlf5.values()))
            print(f"    JLF top-5 done in {time.time()-jt:.0f}s")
        except Exception as e:
            print(f"    JLF top-5 FAILED: {e}")
            pred_jlf5 = pred_mv5
            dices_jlf5 = dices_mv5
            mean_jlf5 = mean_mv5

        elapsed_total = time.time() - t0
        total_time += elapsed_total

        # Print comparison
        print(f"\n  Results for {target_dir.name}:")
        print(f"  {'Method':<20s} {'Mean Dice':>10s}")
        print(f"  {'-'*32}")
        print(f"  {'Single (best)':20s} {mean_single:10.4f}")
        print(f"  {'Majority top-3':20s} {mean_mv3:10.4f}")
        print(f"  {'Majority top-5':20s} {mean_mv5:10.4f}")
        print(f"  {'JLF top-3':20s} {mean_jlf3:10.4f}")
        print(f"  {'JLF top-5':20s} {mean_jlf5:10.4f}")
        print(f"  Time: {elapsed_total:.0f}s (registration: {elapsed_reg:.0f}s)")

        # Per-muscle detail
        print(f"\n  {'Muscle':6s} {'Single':>8s} {'MV-3':>8s} {'MV-5':>8s} {'JLF-3':>8s} {'JLF-5':>8s}")
        print(f"  {'-'*46}")
        all_muscles = sorted(set(
            list(dices_single.keys()) + list(dices_mv3.keys()) +
            list(dices_jlf3.keys()) + list(dices_jlf5.keys())
        ))
        for name in all_muscles:
            s = dices_single.get(name, 0)
            m3 = dices_mv3.get(name, 0)
            m5 = dices_mv5.get(name, 0)
            j3 = dices_jlf3.get(name, 0)
            j5 = dices_jlf5.get(name, 0)
            # Mark best with asterisk
            best = max(s, m3, m5, j3, j5)
            print(f"  {name:6s} {s:8.4f} {m3:8.4f} {m5:8.4f} {j3:8.4f} {j5:8.4f}")

        # Store results
        def make_result(dices, mean_dice, extra=None):
            r = {"dices": dices, "mean_dice": mean_dice, "time": elapsed_total}
            if extra:
                r.update(extra)
            return r

        results["single"][target_dir.name] = make_result(
            dices_single, mean_single,
            {"best_atlas": top_k[0][1].name, "best_ncc": top_k[0][0]}
        )
        results["majority_3"][target_dir.name] = make_result(dices_mv3, mean_mv3)
        results["majority_5"][target_dir.name] = make_result(dices_mv5, mean_mv5)
        results["jlf_3"][target_dir.name] = make_result(dices_jlf3, mean_jlf3)
        results["jlf_5"][target_dir.name] = make_result(dices_jlf5, mean_jlf5)

        # Save JLF predictions
        target_img = mris_full[target_dir]
        for method, pred_arr in [("jlf3", pred_jlf3), ("jlf5", pred_jlf5)]:
            pred_img = target_img.new_image_like(pred_arr.astype(np.float32))
            ants.image_write(pred_img, str(JLF_RESULTS_DIR / f"{method}_{target_dir.name}.nii.gz"))

        # Save incremental results
        with open(JLF_RESULTS_DIR / "results_all.json", "w") as f:
            json.dump(results, f, indent=2)

    # === Final Summary ===
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    for method_name, method_key in [
        ("Single-Atlas (best)", "single"),
        ("Majority Vote (top-3)", "majority_3"),
        ("Majority Vote (top-5)", "majority_5"),
        ("JLF (top-3)", "jlf_3"),
        ("JLF (top-5)", "jlf_5"),
    ]:
        method_results = results[method_key]
        muscle_dices = {}
        for subj, data in method_results.items():
            for muscle, dice in data["dices"].items():
                muscle_dices.setdefault(muscle, []).append(dice)

        print(f"\n--- {method_name} ---")
        print(f"{'Muscle':8s} {'Mean':>8s} {'Std':>8s}")
        print("-" * 26)
        overall = []
        for muscle in sorted(muscle_dices.keys()):
            vals = muscle_dices[muscle]
            overall.extend(vals)
            print(f"{muscle:8s} {np.mean(vals):8.4f} {np.std(vals):8.4f}")
        print("-" * 26)
        print(f"{'OVERALL':8s} {np.mean(overall):8.4f} {np.std(overall):8.4f}")

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save final results
    with open(JLF_RESULTS_DIR / "results_all.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {JLF_RESULTS_DIR / 'results_all.json'}")


if __name__ == "__main__":
    main()
