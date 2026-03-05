"""Data loading and discovery utilities."""

import ants
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results" / "atlas_baseline"
FIGURES_DIR = REPO_ROOT / "figures"


_POSITION_MAP = {"extension": "ext", "flexion": "flx"}


def get_subjects(position="extension"):
    """Find all subject directories with MRI and labels for a given position."""
    suffix = _POSITION_MAP.get(position, position)
    subjects = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and d.name.endswith(f"_{suffix}"):
            if (d / "mri.nii.gz").exists() and (d / "labels.nii.gz").exists():
                subjects.append(d)
    return subjects


def find_atlases(data_dir=None, position="extension"):
    """Find all available atlas directories."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    suffix = _POSITION_MAP.get(position, position) if position else None
    atlases = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        if suffix and not d.name.endswith(f"_{suffix}"):
            continue
        mri = d / "mri.nii.gz"
        has_labels = (d / "labels.nii.gz").exists() or (d / "full.nii.gz").exists()
        if mri.exists() and has_labels:
            atlases.append(d)
    return atlases


def load_atlas(atlas_dir):
    """Load atlas MRI and labels."""
    atlas_dir = Path(atlas_dir)
    mri = ants.image_read(str(atlas_dir / "mri.nii.gz"))
    for name in ["labels.nii.gz", "full.nii.gz"]:
        p = atlas_dir / name
        if p.exists():
            labels = ants.image_read(str(p))
            return mri, labels
    raise FileNotFoundError(f"No label file in {atlas_dir}")
