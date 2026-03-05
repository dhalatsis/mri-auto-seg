"""Shared utilities for MRI forearm auto-segmentation."""

from .labels import LABELS, MUSCLE_LABELS, NAME_TO_LID
from .metrics import compute_dices
from .visualization import COLORS, make_overlay
from .data import DATA_DIR, RESULTS_DIR, FIGURES_DIR, get_subjects, find_atlases, load_atlas, REPO_ROOT
from .registration import (
    register_pair, majority_vote, rank_atlases_fast,
    DEFAULT_WORKING_SPACING, AFFINE_SPACING,
)
