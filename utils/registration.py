"""Registration and label fusion utilities."""

import numpy as np
import ants

DEFAULT_WORKING_SPACING = (0.5, 0.5, 3.0)
AFFINE_SPACING = (1.0, 1.0, 3.0)


def rank_atlases_fast(target_mri_ds, atlas_mris_ds, atlas_dirs):
    """Rank atlases by NCC after fast affine registration.

    Args:
        target_mri_ds: Downsampled target ANTs image.
        atlas_mris_ds: List of downsampled atlas ANTs images.
        atlas_dirs: List of atlas directory paths (parallel to atlas_mris_ds).

    Returns:
        List of (ncc_score, atlas_dir) sorted descending.
    """
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


def register_pair(atlas_mri, atlas_labels, target_mri,
                  working_spacing=DEFAULT_WORKING_SPACING):
    """Register one atlas to target using SyN and propagate labels.

    Returns:
        warped_labels: ANTs image of warped labels in target space.
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

    warped = ants.apply_transforms(
        fixed=target_mri,
        moving=atlas_labels,
        transformlist=reg["fwdtransforms"],
        interpolator="genericLabel",
    )
    return warped


def majority_vote(label_arrays):
    """Majority voting across multiple label arrays."""
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
