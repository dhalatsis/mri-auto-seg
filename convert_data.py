#!/usr/bin/env python3
"""Convert all MRI data to a uniform NIfTI format.

Handles:
1. Dataset 1: DICOM .IMA files -> NIfTI volume (already has label NIfTI)
2. Dataset 2: DICOM files -> NIfTI volumes, ImageJ ROI files -> NIfTI label volumes
"""

import os
import struct
import zipfile
import tempfile
import numpy as np
import nibabel as nib
import pydicom
from pathlib import Path
from scipy.ndimage import binary_fill_holes
from skimage.draw import polygon as draw_polygon

from utils.labels import MUSCLE_LABELS

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "data"


def parse_imagej_roi(filepath):
    """Parse an ImageJ ROI file and return polygon coordinates (x, y arrays)."""
    with open(filepath, "rb") as f:
        data = f.read()

    magic = data[0:4]
    if magic != b"Iout":
        raise ValueError(f"Not an ImageJ ROI file: {filepath}")

    roi_type = data[6]
    top = struct.unpack(">h", data[8:10])[0]
    left = struct.unpack(">h", data[10:12])[0]
    n_coords = struct.unpack(">h", data[16:18])[0]

    # Coordinates start at offset 64
    offset = 64
    x_coords = []
    y_coords = []
    for i in range(n_coords):
        x = struct.unpack(">h", data[offset + i * 2 : offset + i * 2 + 2])[0]
        x_coords.append(x + left)
    offset += n_coords * 2
    for i in range(n_coords):
        y = struct.unpack(">h", data[offset + i * 2 : offset + i * 2 + 2])[0]
        y_coords.append(y + top)

    return np.array(x_coords), np.array(y_coords)


def dicom_to_nifti(dicom_dir, output_path):
    """Convert a directory of DICOM files to a NIfTI volume.

    Returns:
        img: nibabel NIfTI image
        instance_to_slice: dict mapping DICOM InstanceNumber -> NIfTI slice index
    """
    dicom_dir = Path(dicom_dir)
    files = sorted(dicom_dir.iterdir())

    # Read all DICOM slices
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f))
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        raise ValueError(f"No valid DICOM files in {dicom_dir}")

    # Sort by ImagePositionPatient Z coordinate
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    # Build mapping: DICOM InstanceNumber -> NIfTI slice index
    instance_to_slice = {}
    for nifti_idx, s in enumerate(slices):
        inst_num = int(getattr(s, "InstanceNumber", nifti_idx + 1))
        instance_to_slice[inst_num] = nifti_idx

    # Build 3D volume
    rows = slices[0].Rows
    cols = slices[0].Columns
    n_slices = len(slices)

    volume = np.zeros((rows, cols, n_slices), dtype=np.float32)
    for i, s in enumerate(slices):
        pixel_array = s.pixel_array.astype(np.float32)
        # Apply rescale if available
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", 0)
        volume[:, :, i] = pixel_array * slope + intercept

    # Build affine from DICOM metadata
    pixel_spacing = [float(x) for x in slices[0].PixelSpacing]
    pos_first = [float(x) for x in slices[0].ImagePositionPatient]
    pos_last = [float(x) for x in slices[-1].ImagePositionPatient]

    orientation = [float(x) for x in slices[0].ImageOrientationPatient]
    row_cosine = np.array(orientation[:3])
    col_cosine = np.array(orientation[3:])

    # Slice direction
    if n_slices > 1:
        slice_cosine = (np.array(pos_last) - np.array(pos_first)) / (n_slices - 1)
        slice_spacing = np.linalg.norm(slice_cosine)
        slice_cosine = slice_cosine / slice_spacing
    else:
        slice_cosine = np.cross(row_cosine, col_cosine)
        slice_spacing = float(getattr(slices[0], "SliceThickness", 1.0))

    # Build affine matrix
    affine = np.eye(4)
    affine[:3, 0] = row_cosine * pixel_spacing[1]
    affine[:3, 1] = col_cosine * pixel_spacing[0]
    affine[:3, 2] = slice_cosine * slice_spacing
    affine[:3, 3] = pos_first

    img = nib.Nifti1Image(volume, affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(img, str(output_path))
    print(f"  Saved NIfTI: {output_path} (shape={volume.shape})")
    return img, instance_to_slice


def rois_to_label_volume(roi_dir, reference_img, muscle_labels, position="extension"):
    """Convert ImageJ ROI zip files to a 3D label volume.

    Args:
        roi_dir: Directory containing per-slice subdirectories with ROI zip files
        reference_img: Reference NIfTI image for shape/affine
        muscle_labels: Dict mapping muscle name -> label ID
        position: 'extension' or 'flexion'
    """
    ref_shape = reference_img.shape
    label_vol = np.zeros(ref_shape, dtype=np.int16)

    # Get sorted subject slice directories
    slice_dirs = sorted(
        [d for d in Path(roi_dir).iterdir() if d.is_dir()],
        key=lambda x: int(x.name),
    )

    for slice_dir in slice_dirs:
        subject_id = int(slice_dir.name)
        # Find corresponding MRI slice directory to get DICOM count
        mri_slice_dir = (
            BASE_DIR
            / "MRI_Segmentation_2"
            / "mri_files"
            / position
            / str(subject_id)
        )

        if not mri_slice_dir.exists():
            continue

        # Get DICOM files for this subject to know total slices
        dcm_files = sorted(mri_slice_dir.iterdir())
        n_total_slices = len(dcm_files)

        # Process each muscle ROI zip
        for roi_zip_path in slice_dir.glob("*.zip"):
            muscle_name = roi_zip_path.stem
            if muscle_name not in muscle_labels:
                print(f"  Warning: Unknown muscle {muscle_name}, skipping")
                continue

            label_id = muscle_labels[muscle_name]

            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(roi_zip_path, "r") as zf:
                    zf.extractall(tmpdir)

                # Process each per-slice ROI file
                for roi_file in Path(tmpdir).glob("*.roi"):
                    # Filename format: <slice_number>-<muscle>.roi
                    parts = roi_file.stem.split("-")
                    try:
                        slice_idx = int(parts[0]) - 1  # Convert to 0-indexed
                    except ValueError:
                        continue

                    if slice_idx < 0 or slice_idx >= ref_shape[2]:
                        continue

                    try:
                        x_coords, y_coords = parse_imagej_roi(str(roi_file))
                    except Exception as e:
                        print(f"  Warning: Failed to parse {roi_file}: {e}")
                        continue

                    # Clip coordinates to image bounds
                    x_coords = np.clip(x_coords, 0, ref_shape[1] - 1)
                    y_coords = np.clip(y_coords, 0, ref_shape[0] - 1)

                    if len(x_coords) < 3:
                        continue

                    # Draw filled polygon on the slice
                    rr, cc = draw_polygon(y_coords, x_coords, shape=ref_shape[:2])
                    label_vol[rr, cc, slice_idx] = label_id

    return label_vol


def convert_dataset1():
    """Convert Dataset 1 (PD_PROPELLER) - DICOMs to NIfTI."""
    print("=== Converting Dataset 1 ===")
    src_dir = BASE_DIR / "PD_PROPELLER_5MM_FATS_FLX_0012"
    out_dir = OUTPUT_DIR / "template-01"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert DICOMs to NIfTI volume
    dicom_files = sorted(src_dir.glob("*.IMA"))
    if dicom_files:
        print(f"  Found {len(dicom_files)} DICOM files")
        img, _ = dicom_to_nifti(src_dir, out_dir / "mri.nii.gz")
    else:
        print("  No DICOM .IMA files found, skipping MRI conversion")
        img = None

    # Copy existing label files
    import shutil

    for label_file in ["full.nii.gz", "bones_muscles.nii.gz"]:
        src = src_dir / label_file
        dst = out_dir / label_file
        if src.exists():
            shutil.copy2(str(src), str(dst))
            print(f"  Copied {label_file}")

    return img


def convert_dataset2():
    """Convert Dataset 2 (MRI_Segmentation_2) - multiple subjects."""
    print("\n=== Converting Dataset 2 ===")
    src_dir = BASE_DIR / "MRI_Segmentation_2"

    if not src_dir.exists():
        print("  Dataset 2 not found, skipping")
        return

    subjects = sorted(
        [d.name for d in (src_dir / "mri_files" / "extension").iterdir() if d.is_dir()],
        key=int,
    )

    pos_abbrev = {"extension": "ext", "flexion": "flx"}
    for position in ["extension", "flexion"]:
        print(f"\n  --- Position: {position} ---")

        for idx, subject_id in enumerate(subjects, start=1):
            subj_name = f"sub-{idx:02d}_{pos_abbrev[position]}"
            out_dir = OUTPUT_DIR / subj_name
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  Processing {subj_name}...")

            # Convert DICOMs
            dicom_dir = src_dir / "mri_files" / position / subject_id
            if dicom_dir.exists() and any(dicom_dir.iterdir()):
                ref_img, instance_to_slice = dicom_to_nifti(
                    dicom_dir, out_dir / "mri.nii.gz"
                )
            else:
                print(f"    No DICOM files for {subj_name}")
                continue

            # Convert ROIs to labels
            roi_base = src_dir / "roi_files" / position / subject_id
            if roi_base.exists():
                print(f"    Converting ROIs (using DICOM instance mapping)...")
                # For dataset 2, each subject folder directly contains the ROI zips
                # We need to process them into a label volume

                label_vol = np.zeros(ref_img.shape, dtype=np.int16)

                for roi_zip_path in roi_base.glob("*.zip"):
                    muscle_name = roi_zip_path.stem
                    if muscle_name not in MUSCLE_LABELS:
                        print(f"    Warning: Unknown muscle {muscle_name}")
                        continue

                    label_id = MUSCLE_LABELS[muscle_name]

                    with tempfile.TemporaryDirectory() as tmpdir:
                        with zipfile.ZipFile(roi_zip_path, "r") as zf:
                            zf.extractall(tmpdir)

                        for roi_file in sorted(Path(tmpdir).glob("*.roi")):
                            parts = roi_file.stem.split("-")
                            try:
                                # ROI filename number is DICOM InstanceNumber
                                instance_num = int(parts[0])
                            except ValueError:
                                continue

                            # Map DICOM instance number to NIfTI slice index
                            slice_idx = instance_to_slice.get(instance_num)
                            if slice_idx is None:
                                continue

                            if slice_idx < 0 or slice_idx >= ref_img.shape[2]:
                                continue

                            try:
                                x_coords, y_coords = parse_imagej_roi(str(roi_file))
                            except Exception as e:
                                print(f"    Warning: {roi_file.name}: {e}")
                                continue

                            x_coords = np.clip(x_coords, 0, ref_img.shape[1] - 1)
                            y_coords = np.clip(y_coords, 0, ref_img.shape[0] - 1)

                            if len(x_coords) < 3:
                                continue

                            rr, cc = draw_polygon(
                                y_coords, x_coords, shape=ref_img.shape[:2]
                            )
                            label_vol[rr, cc, slice_idx] = label_id

                n_labeled = np.sum(label_vol > 0)
                n_muscles = len(np.unique(label_vol)) - 1
                print(
                    f"    Labels: {n_muscles} muscles, {n_labeled} labeled voxels"
                )

                label_img = nib.Nifti1Image(label_vol, ref_img.affine, ref_img.header)
                nib.save(label_img, str(out_dir / "labels.nii.gz"))
                print(f"    Saved {out_dir / 'labels.nii.gz'}")
            else:
                print(f"    No ROI files for {subj_name}")


if __name__ == "__main__":
    print("MRI Data Conversion Pipeline")
    print("=" * 50)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    convert_dataset1()
    convert_dataset2()

    print("\n" + "=" * 50)
    print("Conversion complete!")
    print(f"Output directory: {OUTPUT_DIR}")

    # Summary
    print("\nConverted datasets:")
    for d in sorted(OUTPUT_DIR.iterdir()):
        if d.is_dir():
            files = list(d.glob("*.nii.gz"))
            print(f"  {d.name}: {[f.name for f in files]}")
