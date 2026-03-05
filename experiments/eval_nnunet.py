#!/usr/bin/env python3
"""Evaluate nnU-Net predictions after training completes.

For each fold, the held-out case is predicted using the best checkpoint.
Predictions are remapped from nnU-Net contiguous labels back to original IDs
and Dice scores are computed against ground truth.

Usage:
    python3 experiments/eval_nnunet.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import subprocess
import numpy as np
import nibabel as nib

from utils import LABELS, compute_dices, get_subjects, REPO_ROOT

# nnU-Net paths
NNUNET_BASE = REPO_ROOT / "nnUNet_data"
DATASET_DIR = NNUNET_BASE / "nnUNet_raw" / "Dataset001_ForearmMuscles"
MODEL_DIR = (NNUNET_BASE / "nnUNet_results" / "Dataset001_ForearmMuscles" /
             "nnUNetTrainer__nnUNetPlans__3d_fullres")

RESULTS_DIR = REPO_ROOT / "results" / "nnunet"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["nnUNet_raw"] = str(NNUNET_BASE / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(NNUNET_BASE / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(NNUNET_BASE / "nnUNet_results")

# Load label remapping
with open(DATASET_DIR / "label_remap.json") as f:
    label_remap = json.load(f)

# Build nnunet_id -> original_id mapping
NNUNET_TO_ORIG = {}
for nnunet_str, info in label_remap.items():
    NNUNET_TO_ORIG[int(nnunet_str)] = info["original_id"]

# Load case mapping
with open(DATASET_DIR / "case_mapping.json") as f:
    case_mapping = json.load(f)


def remap_prediction(pred_path, output_path):
    """Remap nnU-Net contiguous labels back to original sparse IDs."""
    img = nib.load(str(pred_path))
    data = img.get_fdata().astype(np.int16)

    remapped = np.zeros_like(data)
    for nnunet_id, orig_id in NNUNET_TO_ORIG.items():
        remapped[data == nnunet_id] = orig_id

    out_img = nib.Nifti1Image(remapped, img.affine, img.header)
    nib.save(out_img, str(output_path))
    return remapped


def predict_fold(fold):
    """Run nnU-Net prediction for one fold's validation case."""
    case_id = f"forearm_{fold+1:03d}"
    input_dir = DATASET_DIR / "imagesTr"
    output_dir = RESULTS_DIR / f"fold_{fold}_pred"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if checkpoint exists
    checkpoint = MODEL_DIR / f"fold_{fold}" / "checkpoint_best.pth"
    if not checkpoint.exists():
        print(f"  Fold {fold}: no checkpoint found, skipping")
        return None

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", "1",
        "-c", "3d_fullres",
        "-f", str(fold),
        "--save_probabilities",
    ]
    print(f"  Running prediction for fold {fold} ({case_id})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return None

    # The prediction file
    pred_path = output_dir / f"{case_id}.nii.gz"
    if not pred_path.exists():
        print(f"  WARNING: prediction file not found at {pred_path}")
        return None

    return pred_path


def main():
    subjects = get_subjects()
    print(f"Found {len(subjects)} subjects")
    print(f"Model dir: {MODEL_DIR}")

    # Check which folds are trained
    trained_folds = []
    for fold in range(7):
        checkpoint = MODEL_DIR / f"fold_{fold}" / "checkpoint_best.pth"
        if checkpoint.exists():
            trained_folds.append(fold)
            print(f"  Fold {fold}: checkpoint found")
        else:
            print(f"  Fold {fold}: NOT trained yet")

    if not trained_folds:
        print("\nNo trained folds found. Run training first.")
        return

    results = {}

    for fold in trained_folds:
        case_id = f"forearm_{fold+1:03d}"
        subj_name = case_mapping[case_id]

        print(f"\n{'='*50}")
        print(f"Fold {fold}: evaluating {subj_name} ({case_id})")
        print(f"{'='*50}")

        # Predict
        pred_path = predict_fold(fold)
        if pred_path is None:
            continue

        # Remap labels
        remapped_path = RESULTS_DIR / f"nnunet_{subj_name}.nii.gz"
        pred_remapped = remap_prediction(pred_path, remapped_path)
        print(f"  Saved remapped prediction to {remapped_path}")

        # Load ground truth
        gt_path = next(s for s in subjects if s.name == subj_name) / "labels.nii.gz"
        gt = nib.load(str(gt_path)).get_fdata().astype(np.int16)

        # Compute Dice
        dices = compute_dices(pred_remapped, gt)
        mean_dice = np.mean(list(dices.values()))

        print(f"  Mean Dice: {mean_dice:.4f}")
        print(f"  {'Muscle':8s} {'Dice':>8s}")
        print(f"  {'-'*18}")
        for name in sorted(dices.keys()):
            print(f"  {name:8s} {dices[name]:8.4f}")

        results[subj_name] = {
            "fold": fold,
            "case_id": case_id,
            "dices": dices,
            "mean_dice": mean_dice,
        }

    # Summary
    if results:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")

        muscle_dices = {}
        for subj, data in results.items():
            for muscle, dice in data["dices"].items():
                muscle_dices.setdefault(muscle, []).append(dice)

        print(f"\n{'Muscle':8s} {'Mean':>8s} {'Std':>8s} {'N':>4s}")
        print("-" * 30)
        overall = []
        for muscle in sorted(muscle_dices.keys()):
            vals = muscle_dices[muscle]
            overall.extend(vals)
            print(f"{muscle:8s} {np.mean(vals):8.4f} {np.std(vals):8.4f} {len(vals):4d}")
        print("-" * 30)
        print(f"{'OVERALL':8s} {np.mean(overall):8.4f} {np.std(overall):8.4f}")

        # Save results
        with open(RESULTS_DIR / "results_nnunet.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {RESULTS_DIR / 'results_nnunet.json'}")


if __name__ == "__main__":
    main()
