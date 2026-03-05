#!/usr/bin/env python3
"""Prepare forearm MRI data in nnU-Net v2 format.

nnU-Net requires:
  nnUNet_raw/Dataset001_ForearmMuscles/
    imagesTr/  - {case_id}_0000.nii.gz (channel 0 = T2 MRI)
    labelsTr/  - {case_id}.nii.gz (integer labels)
    dataset.json

We use all 7 extension subjects. nnU-Net handles cross-validation internally.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import shutil
import numpy as np
import nibabel as nib

from utils import LABELS, get_subjects, REPO_ROOT

# nnU-Net directory structure
NNUNET_BASE = REPO_ROOT / "nnUNet_data"
DATASET_ID = 1
DATASET_NAME = f"Dataset{DATASET_ID:03d}_ForearmMuscles"
DATASET_DIR = NNUNET_BASE / "nnUNet_raw" / DATASET_NAME

IMAGES_TR = DATASET_DIR / "imagesTr"
LABELS_TR = DATASET_DIR / "labelsTr"

# Remap sparse labels (2,3,...,19 with gaps at 1,13) to contiguous 0-17
# nnU-Net requires consecutive integer labels
ORIGINAL_LIDS = sorted(LABELS.keys())  # [2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19]
ORIG_TO_NNUNET = {orig: new for new, orig in enumerate(ORIGINAL_LIDS, start=1)}
ORIG_TO_NNUNET[0] = 0  # background stays 0
NNUNET_TO_ORIG = {v: k for k, v in ORIG_TO_NNUNET.items()}


def remap_labels(label_path, output_path):
    """Remap sparse label IDs to contiguous 0-17 for nnU-Net."""
    img = nib.load(str(label_path))
    data = img.get_fdata().astype(np.int16)

    remapped = np.zeros_like(data)
    for orig, new in ORIG_TO_NNUNET.items():
        remapped[data == orig] = new

    out_img = nib.Nifti1Image(remapped, img.affine, img.header)
    nib.save(out_img, str(output_path))


def main():
    subjects = get_subjects()
    print(f"Found {len(subjects)} subjects")

    # Create directories
    IMAGES_TR.mkdir(parents=True, exist_ok=True)
    LABELS_TR.mkdir(parents=True, exist_ok=True)

    # Copy data
    case_ids = []
    for i, subj_dir in enumerate(subjects):
        # Use simple case IDs: forearm_001, forearm_002, etc.
        case_id = f"forearm_{i+1:03d}"
        case_ids.append(case_id)

        src_mri = subj_dir / "mri.nii.gz"
        src_labels = subj_dir / "labels.nii.gz"

        dst_mri = IMAGES_TR / f"{case_id}_0000.nii.gz"
        dst_labels = LABELS_TR / f"{case_id}.nii.gz"

        print(f"  {subj_dir.name} -> {case_id}")

        # Copy MRI (channel 0)
        shutil.copy2(str(src_mri), str(dst_mri))

        # Copy labels (keeping original sparse IDs)
        remap_labels(src_labels, dst_labels)

        # Verify
        mri = nib.load(str(dst_mri))
        lab = nib.load(str(dst_labels))
        print(f"    MRI shape={mri.shape}, Labels shape={lab.shape}")
        print(f"    Labels: {sorted(set(np.unique(lab.get_fdata().astype(int))) - {0})}")

    # Generate dataset.json with remapped contiguous labels
    label_dict = {"background": 0}
    for orig_lid in ORIGINAL_LIDS:
        name = LABELS[orig_lid]
        label_dict[name] = ORIG_TO_NNUNET[orig_lid]

    dataset_json = {
        "channel_names": {
            "0": "T2"
        },
        "labels": label_dict,
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
    }

    json_path = DATASET_DIR / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"\nSaved dataset.json to {json_path}")

    # Print summary
    print(f"\nnnU-Net dataset ready at: {DATASET_DIR}")
    print(f"  Images: {IMAGES_TR} ({len(case_ids)} cases)")
    print(f"  Labels: {LABELS_TR} ({len(case_ids)} cases)")
    print(f"\nTo train:")
    print(f"  export nnUNet_raw={NNUNET_BASE / 'nnUNet_raw'}")
    print(f"  export nnUNet_preprocessed={NNUNET_BASE / 'nnUNet_preprocessed'}")
    print(f"  export nnUNet_results={NNUNET_BASE / 'nnUNet_results'}")
    print(f"  nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")
    print(f"  nnUNetv2_train {DATASET_ID} 3d_fullres 0  # fold 0")
    print(f"  # For full 5-fold CV:")
    print(f"  for fold in 0 1 2 3 4; do nnUNetv2_train {DATASET_ID} 3d_fullres $fold; done")

    # Save subject mapping and label mapping for later inverse remapping
    mapping = {}
    for i, subj_dir in enumerate(subjects):
        case_id = f"forearm_{i+1:03d}"
        mapping[case_id] = subj_dir.name

    with open(DATASET_DIR / "case_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nCase mapping saved to {DATASET_DIR / 'case_mapping.json'}")

    # Save label remapping (nnunet_id -> original_id) for prediction postprocessing
    label_remap = {str(v): {"original_id": k, "name": LABELS.get(k, "background")}
                   for k, v in ORIG_TO_NNUNET.items()}
    with open(DATASET_DIR / "label_remap.json", "w") as f:
        json.dump(label_remap, f, indent=2)
    print(f"Label remapping saved to {DATASET_DIR / 'label_remap.json'}")
    print(f"\nLabel mapping (original -> nnU-Net):")
    for orig_lid in ORIGINAL_LIDS:
        print(f"  {LABELS[orig_lid]:6s} (ID {orig_lid:2d}) -> nnU-Net ID {ORIG_TO_NNUNET[orig_lid]:2d}")


if __name__ == "__main__":
    main()
