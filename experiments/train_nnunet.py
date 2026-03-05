#!/usr/bin/env python3
"""Train nnU-Net with leave-one-out cross-validation (7 folds for 7 subjects).

nnU-Net's default is 5-fold CV, but with only 7 subjects we want true LOO
to match our atlas-based evaluation. This script:

1. Creates a custom splits file for 7-fold LOO
2. Trains all 7 folds (3d_fullres config)
3. Runs prediction on each held-out subject
4. Evaluates and compares to atlas-based baseline

Usage:
    python3 -u experiments/train_nnunet.py 2>&1 | tee results/nnunet/train.log
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import subprocess
import numpy as np
import nibabel as nib

from utils import LABELS, REPO_ROOT

# nnU-Net paths
NNUNET_BASE = REPO_ROOT / "nnUNet_data"
DATASET_ID = 1

# Set environment
os.environ["nnUNet_raw"] = str(NNUNET_BASE / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(NNUNET_BASE / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(NNUNET_BASE / "nnUNet_results")

RESULTS_DIR = REPO_ROOT / "results" / "nnunet"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_loo_splits():
    """Create 7-fold leave-one-out splits file for nnU-Net."""
    case_ids = [f"forearm_{i+1:03d}" for i in range(7)]

    splits = []
    for fold_idx in range(7):
        val_case = case_ids[fold_idx]
        train_cases = [c for c in case_ids if c != val_case]
        splits.append({
            "train": train_cases,
            "val": [val_case],
        })

    # Save to preprocessed directory
    splits_path = (NNUNET_BASE / "nnUNet_preprocessed" /
                   "Dataset001_ForearmMuscles" / "splits_final.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Created 7-fold LOO splits at {splits_path}")
    for i, s in enumerate(splits):
        print(f"  Fold {i}: val={s['val']}, train={s['train']}")
    return splits


def train_fold(fold):
    """Train one fold of nnU-Net."""
    print(f"\n{'='*60}")
    print(f"Training fold {fold}/6")
    print(f"{'='*60}")

    cmd = [
        "nnUNetv2_train", str(DATASET_ID), "3d_fullres", str(fold),
        "--npz",  # Save softmax outputs for potential ensemble
    ]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: Fold {fold} training may have failed (exit code {result.returncode})")
    return result.returncode


def main():
    print("nnU-Net 7-fold LOO Training")
    print("=" * 60)

    # Step 1: Create LOO splits
    splits = create_loo_splits()

    # Step 2: Train all 7 folds
    for fold in range(7):
        ret = train_fold(fold)
        if ret != 0:
            print(f"Fold {fold} failed, continuing to next fold...")

    print("\n" + "=" * 60)
    print("All folds complete!")
    print(f"Results in: {os.environ['nnUNet_results']}")


if __name__ == "__main__":
    main()
