#!/usr/bin/env python3
"""Atlas-based forearm MRI segmentation pipeline.

Supports:
1. Single-atlas registration + label propagation
2. Multi-atlas registration + Joint Label Fusion
3. Leave-one-out cross-validation for evaluation

Usage:
    # Single atlas segmentation
    python segment.py --target data/subject_1_extension/mri.nii.gz \
                      --atlas data/subject_2_extension \
                      --output results/subject_1_from_2.nii.gz

    # Multi-atlas segmentation (uses all atlas dirs)
    python segment.py --target data/subject_1_extension/mri.nii.gz \
                      --atlas data/subject_2_extension data/subject_3_extension \
                      --output results/subject_1_multi.nii.gz

    # Leave-one-out cross-validation
    python segment.py --loocv --data-dir data/ --output-dir results/loocv/
"""

import os
import sys
import time
import argparse
import numpy as np
import nibabel as nib
import ants
from pathlib import Path


MUSCLE_LABELS = {
    2: "ANC",    3: "APL",   4: "ECRB",  5: "ECRL",  6: "ECU",
    7: "ED",     8: "EDM",   9: "EPL",  10: "FCR",  11: "FCU",
    12: "FDP",  14: "FDS",  15: "FPL",  16: "PL",   17: "PQ",
    18: "PT",   19: "SUP",
}


def load_atlas(atlas_dir):
    """Load an atlas (MRI + labels) from a directory."""
    atlas_dir = Path(atlas_dir)
    mri_path = atlas_dir / "mri.nii.gz"

    # Try different label file names
    label_candidates = ["labels.nii.gz", "full.nii.gz", "bones_muscles.nii.gz"]
    label_path = None
    for name in label_candidates:
        p = atlas_dir / name
        if p.exists():
            label_path = p
            break

    if not mri_path.exists():
        raise FileNotFoundError(f"MRI not found: {mri_path}")
    if label_path is None:
        raise FileNotFoundError(f"Labels not found in {atlas_dir}")

    mri = ants.image_read(str(mri_path))
    labels = ants.image_read(str(label_path))

    print(f"  Atlas: {atlas_dir.name} (MRI: {mri.shape}, Labels: {label_path.name})")
    return mri, labels


def resample_to_working_res(image, spacing, is_label=False):
    """Resample image to a working resolution for faster registration."""
    interp_type = 1 if is_label else 0  # 1=NN for labels, 0=linear for images
    return ants.resample_image(image, spacing, use_voxel_space=False, interp_type=interp_type)


def register_and_propagate(atlas_mri, atlas_labels, target_mri,
                           transform_type="SyN", working_spacing=None):
    """Register atlas to target and propagate labels.

    Args:
        atlas_mri: Atlas MRI (ANTs image)
        atlas_labels: Atlas label image (ANTs image)
        target_mri: Target MRI to segment (ANTs image)
        transform_type: ANTs registration type
        working_spacing: If set, downsample to this spacing for registration,
                        then apply transforms at original resolution.

    Returns:
        warped_labels: Label image in target space
        warped_mri: Atlas MRI warped to target space
        reg: Registration result dict
    """
    t0 = time.time()

    if working_spacing is not None:
        # Downsample for registration
        atlas_mri_work = resample_to_working_res(atlas_mri, working_spacing)
        target_mri_work = resample_to_working_res(target_mri, working_spacing)
        print(f"    Working resolution: {atlas_mri_work.shape} (from {atlas_mri.shape})")
    else:
        atlas_mri_work = atlas_mri
        target_mri_work = target_mri

    reg = ants.registration(
        fixed=target_mri_work,
        moving=atlas_mri_work,
        type_of_transform=transform_type,
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

    elapsed = time.time() - t0
    print(f"    Registration completed in {elapsed:.1f}s")

    return warped_labels, warped_mri, reg


def single_atlas_segment(target_mri, atlas_dir, output_path,
                         transform_type="SyN", working_spacing=None):
    """Segment target using a single atlas."""
    print(f"\nSingle-atlas segmentation")
    atlas_mri, atlas_labels = load_atlas(atlas_dir)

    warped_labels, warped_mri, _ = register_and_propagate(
        atlas_mri, atlas_labels, target_mri,
        transform_type=transform_type, working_spacing=working_spacing,
    )

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ants.image_write(warped_labels, str(output_path))
    print(f"  Saved: {output_path}")

    return warped_labels


def multi_atlas_segment(target_mri, atlas_dirs, output_path, use_jlf=True,
                        transform_type="SyN", working_spacing=None):
    """Segment target using multiple atlases with Joint Label Fusion.

    Args:
        target_mri: ANTs image of the target MRI
        atlas_dirs: List of atlas directory paths
        output_path: Where to save the result
        use_jlf: If True, use Joint Label Fusion. If False, use majority voting.
        transform_type: ANTs registration type
        working_spacing: Resolution for registration
    """
    n_atlases = len(atlas_dirs)
    print(f"\nMulti-atlas segmentation ({n_atlases} atlases, {'JLF' if use_jlf else 'majority voting'})")

    warped_images = []
    warped_labels_list = []

    for i, atlas_dir in enumerate(atlas_dirs):
        print(f"  Atlas {i+1}/{n_atlases}:")
        atlas_mri, atlas_labels = load_atlas(atlas_dir)
        warped_labels, warped_mri, _ = register_and_propagate(
            atlas_mri, atlas_labels, target_mri,
            transform_type=transform_type, working_spacing=working_spacing,
        )
        warped_images.append(warped_mri)
        warped_labels_list.append(warped_labels)

    if use_jlf and n_atlases >= 2:
        print("  Running Joint Label Fusion...")
        t0 = time.time()

        # Create a mask from the target
        target_mask = ants.get_mask(target_mri)

        result = ants.joint_label_fusion(
            target_image=target_mri,
            target_image_mask=target_mask,
            atlas_list=warped_images,
            label_list=warped_labels_list,
            beta=4,
            rad=2,
            rho=0.01,
            verbose=False,
        )
        fused_labels = result["segmentation"]
        print(f"    JLF completed in {time.time() - t0:.1f}s")
    else:
        # Majority voting fallback
        print("  Running majority voting...")
        fused_labels = majority_vote(warped_labels_list, target_mri)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    ants.image_write(fused_labels, str(output_path))
    print(f"  Saved: {output_path}")

    return fused_labels


def majority_vote(label_images, reference):
    """Simple majority voting across label images."""
    arrays = [img.numpy() for img in label_images]
    stacked = np.stack(arrays, axis=0)

    # Get all unique labels
    all_labels = np.unique(stacked)
    all_labels = all_labels[all_labels > 0]

    result = np.zeros(arrays[0].shape, dtype=np.float32)

    if len(all_labels) > 0:
        # For each voxel, find the most common non-zero label
        # If no atlas predicts non-zero, keep zero
        vote_counts = np.zeros((len(all_labels),) + arrays[0].shape, dtype=np.int32)
        for i, lab in enumerate(all_labels):
            vote_counts[i] = np.sum(stacked == lab, axis=0)

        max_votes = np.max(vote_counts, axis=0)
        best_label_idx = np.argmax(vote_counts, axis=0)

        mask = max_votes > 0
        result[mask] = all_labels[best_label_idx[mask]]

    return reference.new_image_like(result)


def compute_dice(pred, gt, label_id):
    """Compute Dice score for a single label."""
    pred_mask = (pred == label_id)
    gt_mask = (gt == label_id)

    intersection = np.sum(pred_mask & gt_mask)
    total = np.sum(pred_mask) + np.sum(gt_mask)

    if total == 0:
        return float("nan")  # Label not present in either
    return 2.0 * intersection / total


def evaluate_segmentation(pred_path, gt_path):
    """Evaluate predicted segmentation against ground truth."""
    pred_img = nib.load(str(pred_path))
    gt_img = nib.load(str(gt_path))

    pred = pred_img.get_fdata()
    gt = gt_img.get_fdata()

    # Handle shape mismatch (different resolutions)
    if pred.shape != gt.shape:
        print(f"  Warning: shape mismatch pred={pred.shape} gt={gt.shape}")
        return {}

    results = {}
    all_labels = set(np.unique(gt)) | set(np.unique(pred))
    all_labels.discard(0)

    for label_id in sorted(all_labels):
        dice = compute_dice(pred, gt, label_id)
        name = MUSCLE_LABELS.get(int(label_id), f"Label_{int(label_id)}")
        results[name] = dice

    return results


def run_loocv(data_dir, output_dir, position_filter=None,
              transform_type="SyN", working_spacing=None, use_jlf=True):
    """Leave-one-out cross-validation across all subjects."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all subjects with both MRI and labels
    subjects = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        mri = d / "mri.nii.gz"
        labels = None
        for name in ["labels.nii.gz", "full.nii.gz"]:
            if (d / name).exists():
                labels = d / name
                break
        if mri.exists() and labels:
            if position_filter and position_filter not in d.name:
                continue
            subjects.append(d)

    print(f"Found {len(subjects)} subjects for LOOCV")
    if len(subjects) < 2:
        print("Need at least 2 subjects for LOOCV")
        return

    all_results = {}

    for i, target_dir in enumerate(subjects):
        target_name = target_dir.name
        print(f"\n{'='*60}")
        print(f"LOOCV {i+1}/{len(subjects)}: Target = {target_name}")
        print(f"{'='*60}")

        target_mri = ants.image_read(str(target_dir / "mri.nii.gz"))

        # All other subjects are atlases
        atlas_dirs = [d for d in subjects if d != target_dir]

        out_path = output_dir / f"{target_name}_predicted.nii.gz"

        if len(atlas_dirs) == 1:
            pred_labels = single_atlas_segment(
                target_mri, atlas_dirs[0], out_path,
                transform_type=transform_type, working_spacing=working_spacing,
            )
        else:
            pred_labels = multi_atlas_segment(
                target_mri, atlas_dirs, out_path, use_jlf=use_jlf,
                transform_type=transform_type, working_spacing=working_spacing,
            )

        # Evaluate
        gt_path = None
        for name in ["labels.nii.gz", "full.nii.gz"]:
            if (target_dir / name).exists():
                gt_path = target_dir / name
                break

        if gt_path:
            dice_scores = evaluate_segmentation(out_path, gt_path)
            all_results[target_name] = dice_scores

            print(f"\n  Dice scores for {target_name}:")
            valid_scores = {k: v for k, v in dice_scores.items() if not np.isnan(v)}
            for name, dice in sorted(valid_scores.items()):
                print(f"    {name:6s}: {dice:.4f}")
            if valid_scores:
                mean_dice = np.mean(list(valid_scores.values()))
                print(f"    {'MEAN':6s}: {mean_dice:.4f}")

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print("LOOCV Summary")
        print(f"{'='*60}")

        # Aggregate per-muscle
        muscle_dices = {}
        for subj, scores in all_results.items():
            for muscle, dice in scores.items():
                if not np.isnan(dice):
                    muscle_dices.setdefault(muscle, []).append(dice)

        print(f"\n{'Muscle':8s} {'Mean Dice':>10s} {'Std':>8s} {'N':>4s}")
        print("-" * 34)
        overall_means = []
        for muscle in sorted(muscle_dices.keys()):
            dices = muscle_dices[muscle]
            mean = np.mean(dices)
            std = np.std(dices)
            overall_means.append(mean)
            print(f"{muscle:8s} {mean:10.4f} {std:8.4f} {len(dices):4d}")

        print("-" * 34)
        print(f"{'OVERALL':8s} {np.mean(overall_means):10.4f}")

        # Save results
        import json
        results_file = output_dir / "loocv_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=lambda x: None if np.isnan(x) else x)
        print(f"\nResults saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Atlas-based forearm MRI segmentation")
    parser.add_argument("--target", type=str, help="Path to target MRI NIfTI")
    parser.add_argument("--atlas", type=str, nargs="+", help="Path(s) to atlas directories")
    parser.add_argument("--output", type=str, default="predicted_labels.nii.gz")
    parser.add_argument("--loocv", action="store_true", help="Run leave-one-out cross-validation")
    parser.add_argument("--data-dir", type=str, default="data/", help="Data directory for LOOCV")
    parser.add_argument("--output-dir", type=str, default="results/loocv/")
    parser.add_argument("--position", type=str, choices=["extension", "flexion"],
                        help="Filter by position for LOOCV")
    parser.add_argument("--no-jlf", action="store_true", help="Disable JLF, use majority voting")
    parser.add_argument("--transform", type=str, default="SyN",
                        choices=["SyN", "SyNCC", "SyNRA", "ElasticSyN", "Affine"],
                        help="Registration transform type")
    parser.add_argument("--working-spacing", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Working resolution in mm (e.g., 1.0 1.0 3.0)")

    args = parser.parse_args()

    working_spacing = tuple(args.working_spacing) if args.working_spacing else None

    if args.loocv:
        run_loocv(args.data_dir, args.output_dir, args.position,
                  transform_type=args.transform, working_spacing=working_spacing,
                  use_jlf=not args.no_jlf)
        return

    if not args.target:
        parser.error("--target is required (unless using --loocv)")

    target_mri = ants.image_read(args.target)
    print(f"Target: {args.target} (shape={target_mri.shape})")

    if not args.atlas:
        parser.error("--atlas is required (unless using --loocv)")

    if len(args.atlas) == 1:
        single_atlas_segment(target_mri, args.atlas[0], args.output,
                             transform_type=args.transform,
                             working_spacing=working_spacing)
    else:
        multi_atlas_segment(
            target_mri, args.atlas, args.output, use_jlf=not args.no_jlf,
            transform_type=args.transform, working_spacing=working_spacing,
        )


if __name__ == "__main__":
    main()
