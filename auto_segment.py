#!/usr/bin/env python3
"""Auto-segment a new forearm MRI using atlas-based registration.

This is the main user-facing script. Given a new MRI (DICOM directory or NIfTI),
it registers multiple atlas templates and fuses their labels to produce a
segmentation.

Usage:
    # Segment a NIfTI file using all available atlases
    python auto_segment.py input_mri.nii.gz -o output_labels.nii.gz

    # Segment a DICOM directory
    python auto_segment.py /path/to/dicom_dir/ -o output_labels.nii.gz

    # Use specific atlases
    python auto_segment.py input.nii.gz -o output.nii.gz \
        --atlases data/sub-01_ext data/sub-02_ext

    # Quick mode (2 atlases, faster)
    python auto_segment.py input.nii.gz -o output.nii.gz --quick
"""

import os
import sys
import time
import argparse
import numpy as np
import ants
from pathlib import Path
from scipy import ndimage

from utils import LABELS, find_atlases, load_atlas, majority_vote, AFFINE_SPACING
from utils.registration import DEFAULT_WORKING_SPACING


def rank_atlases(target_mri, atlas_dirs, top_k=None):
    """Rank atlases by similarity to target using fast affine registration.

    Returns atlas_dirs sorted by descending similarity.
    """
    target_ds = ants.resample_image(target_mri, AFFINE_SPACING, False, 0)

    scores = []
    for atlas_dir in atlas_dirs:
        atlas_mri = ants.image_read(str(Path(atlas_dir) / "mri.nii.gz"))
        atlas_ds = ants.resample_image(atlas_mri, AFFINE_SPACING, False, 0)

        reg = ants.registration(
            fixed=target_ds, moving=atlas_ds,
            type_of_transform="Affine", verbose=False,
        )
        warped = reg["warpedmovout"]
        ncc = float(np.corrcoef(
            warped.numpy().flatten(), target_ds.numpy().flatten()
        )[0, 1])
        scores.append((ncc, atlas_dir))
        print(f"    {Path(atlas_dir).name}: NCC={ncc:.4f}")

    scores.sort(reverse=True)
    ranked = [s[1] for s in scores]

    if top_k:
        ranked = ranked[:top_k]

    return ranked


def load_input(input_path):
    """Load input MRI from NIfTI or DICOM directory."""
    input_path = Path(input_path)

    if input_path.is_dir():
        # DICOM directory - convert first
        from convert_data import dicom_to_nifti
        import tempfile
        tmp = tempfile.mktemp(suffix=".nii.gz")
        dicom_to_nifti(input_path, tmp)
        return ants.image_read(tmp), tmp
    else:
        return ants.image_read(str(input_path)), None


def register_atlas_to_target(atlas_mri, atlas_labels, target_mri,
                              working_spacing=DEFAULT_WORKING_SPACING):
    """Register one atlas to the target and propagate labels."""
    # Resample for registration
    atlas_w = ants.resample_image(atlas_mri, working_spacing, False, 0)
    target_w = ants.resample_image(target_mri, working_spacing, False, 0)

    # Histogram matching
    atlas_matched = ants.histogram_match_image(atlas_w, target_w)

    # SyN registration
    reg = ants.registration(
        fixed=target_w,
        moving=atlas_matched,
        type_of_transform="SyN",
        aff_metric="mattes",
        syn_metric="CC",
        syn_sampling=4,
        reg_iterations=(100, 70, 50, 10),
        verbose=False,
    )

    # Apply at original resolution
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

    # Compute similarity score (NCC between warped atlas and target)
    similarity = np.corrcoef(
        warped_mri.numpy().flatten(),
        target_mri.numpy().flatten()
    )[0, 1]

    return warped_labels, warped_mri, similarity


def postprocess_labels(label_array, min_component_size=50):
    """Clean up segmentation labels.

    - Remove small disconnected components per label
    - Fill small holes within each label
    """
    cleaned = np.zeros_like(label_array)

    for label_id in np.unique(label_array):
        if label_id == 0:
            continue

        mask = (label_array == label_id).astype(np.int32)

        # Remove small components
        labeled, n_components = ndimage.label(mask)
        if n_components > 1:
            for comp_id in range(1, n_components + 1):
                comp_size = np.sum(labeled == comp_id)
                if comp_size < min_component_size:
                    mask[labeled == comp_id] = 0

        # Fill small holes (2D per slice to preserve anatomy)
        for z in range(mask.shape[2]):
            mask[:, :, z] = ndimage.binary_fill_holes(mask[:, :, z]).astype(np.int32)

        cleaned[mask > 0] = label_id

    return cleaned


def auto_segment(input_path, output_path, atlas_dirs=None, n_atlases=None,
                 position="extension", quick=False, postprocess=True,
                 smart_select=True):
    """Main segmentation function."""
    total_t0 = time.time()

    # Load input
    print(f"Loading input: {input_path}")
    target_mri, tmp_path = load_input(input_path)
    print(f"  Shape: {target_mri.shape}, Spacing: {target_mri.spacing}")

    # Find atlases
    if atlas_dirs is None:
        atlas_dirs = find_atlases(position=position)

    if quick:
        n_atlases = min(2, len(atlas_dirs))

    # Smart atlas selection: rank by similarity, pick top N
    if smart_select and len(atlas_dirs) > 1:
        print(f"\nRanking {len(atlas_dirs)} atlases by similarity...")
        atlas_dirs = rank_atlases(target_mri, atlas_dirs, top_k=n_atlases)
        print(f"Selected top {len(atlas_dirs)}: {[Path(d).name for d in atlas_dirs]}")
    elif n_atlases:
        atlas_dirs = atlas_dirs[:n_atlases]

    print(f"\nUsing {len(atlas_dirs)} atlases")

    # Register each atlas
    warped_labels_list = []
    similarities = []

    for i, atlas_dir in enumerate(atlas_dirs):
        print(f"\n  Atlas {i+1}/{len(atlas_dirs)}: {Path(atlas_dir).name}")
        atlas_mri, atlas_labels = load_atlas(atlas_dir)

        t0 = time.time()
        warped_labels, warped_mri, similarity = register_atlas_to_target(
            atlas_mri, atlas_labels, target_mri
        )
        elapsed = time.time() - t0

        warped_labels_list.append(warped_labels.numpy())
        similarities.append(similarity)
        print(f"    Done in {elapsed:.0f}s (similarity={similarity:.4f})")

    # Fuse labels
    if len(warped_labels_list) == 1:
        fused = warped_labels_list[0]
    else:
        print(f"\nFusing {len(warped_labels_list)} atlases (majority voting)...")
        fused = majority_vote(warped_labels_list)

    # Post-process
    if postprocess:
        print("Post-processing (removing small components, filling holes)...")
        fused = postprocess_labels(fused)

    # Save
    output_img = target_mri.new_image_like(fused.astype(np.float32))
    ants.image_write(output_img, str(output_path))

    total_elapsed = time.time() - total_t0
    print(f"\nSegmentation complete!")
    print(f"  Output: {output_path}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Summary
    n_labeled = np.sum(fused > 0)
    unique_labels = set(np.unique(fused)) - {0}
    print(f"  Labels found: {len(unique_labels)}")
    print(f"  Labeled voxels: {n_labeled} ({100*n_labeled/fused.size:.1f}%)")

    for lid in sorted(unique_labels):
        name = LABELS.get(int(lid), f"Label_{int(lid)}")
        count = np.sum(fused == lid)
        print(f"    {name:6s} (label {int(lid):2d}): {count:>8d} voxels")

    # Clean up temp file
    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)

    return output_img


def main():
    parser = argparse.ArgumentParser(
        description="Auto-segment a forearm MRI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python auto_segment.py input.nii.gz -o segmentation.nii.gz
    python auto_segment.py /path/to/dicoms/ -o segmentation.nii.gz --quick
    python auto_segment.py input.nii.gz -o seg.nii.gz --atlases data/sub-01_ext data/sub-02_ext
        """,
    )
    parser.add_argument("input", help="Input MRI (NIfTI file or DICOM directory)")
    parser.add_argument("-o", "--output", required=True, help="Output labels NIfTI path")
    parser.add_argument("--atlases", nargs="+", help="Atlas directories to use")
    parser.add_argument("-n", "--n-atlases", type=int, help="Max number of atlases")
    parser.add_argument("--position", default="extension",
                        choices=["extension", "flexion"],
                        help="Wrist position to match atlases")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (2 atlases only)")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Skip post-processing")
    parser.add_argument("--no-smart", action="store_true",
                        help="Disable smart atlas selection (use all in order)")

    args = parser.parse_args()

    atlas_dirs = [Path(a) for a in args.atlases] if args.atlases else None

    auto_segment(
        args.input,
        args.output,
        atlas_dirs=atlas_dirs,
        n_atlases=args.n_atlases,
        position=args.position,
        quick=args.quick,
        postprocess=not args.no_postprocess,
        smart_select=not args.no_smart,
    )


if __name__ == "__main__":
    main()
