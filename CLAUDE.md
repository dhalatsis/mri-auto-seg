# MRI Forearm Auto-Segmentation

## Goal
Automate manual MRI segmentation of forearm anatomy (17 muscles) from ~10-15 hours per MRI to ~20 minutes. Target: **0.90 mean Dice** (currently 0.658). Uses atlas-based registration with smart atlas selection.

## Data
- **Dataset 2:** `MRI_Segmentation_2/` — 7 subjects x 2 positions (extension/flexion) = 14 scans. T2-weighted, 448x448x94, 0.27mm res, 3mm slice. 17 muscle labels from ImageJ ROI contours.
- **Template 1:** `PD_PROPELLER_5MM_FATS_FLX_0012/` — 57 DICOM slices, 320x320, PD-weighted, 0.44mm res, 6mm slice. 24 labels (muscles+skin+bone) in `full.nii.gz`
- **Converted data** in `data/` — all NIfTI format (mri.nii.gz + labels.nii.gz per subject). Naming: `sub-NN_ext`/`sub-NN_flx` (contiguous IDs), `template-01`. See `data/subject_mapping.json` for original study ID mapping.
- **Future:** ~10-13 annotated MRIs will be available as templates

### Muscle labels (shared ID scheme)
ANC=2, APL=3, ECRB=4, ECRL=5, ECU=6, ED=7, EDM=8, EPL=9, FCR=10, FCU=11, FDP=12, FDS=14, FPL=15, PL=16, PQ=17, PT=18, SUP=19

## Repository Structure

```
mri-auto-seg/
├── CLAUDE.md                    # This file
├── METHOD.md                    # Detailed methodology writeup
├── IMPROVEMENT_ROADMAP.md       # Path from 0.66 to 0.90 Dice
├── requirements.txt
│
├── auto_segment.py              # Main user-facing segmentation script
├── convert_data.py              # DICOM/ImageJ ROI → NIfTI conversion
│
├── utils/                       # Shared code module
│   ├── __init__.py              # Re-exports common symbols
│   ├── labels.py                # LABELS, MUSCLE_LABELS, NAME_TO_LID
│   ├── metrics.py               # compute_dices()
│   ├── visualization.py         # make_overlay(), COLORS
│   ├── data.py                  # get_subjects(), find_atlases(), load_atlas(), DATA_DIR
│   └── registration.py          # register_pair(), majority_vote(), rank_atlases_fast()
│
├── eval/                        # Evaluation scripts
│   └── eval_loocv.py            # LOOCV evaluation (smart single + multi-atlas)
│
├── plotting/                    # Visualization scripts
│   ├── plot_results.py          # Per-muscle/per-subject bar charts
│   ├── plot_analysis.py         # Tiers, error analysis, gap-to-target
│   ├── plot_overlays.py         # GT vs prediction overlays
│   ├── plot_atlas_muscles.py    # Muscle location visualization
│   └── visualize.py             # Simple MRI slice viewer
│
├── experiments/                  # New experiment scripts go here
│   └── README.md                # Conventions
│
├── archive/                     # Deprecated scripts (preserved for reference)
│
├── data/                        # Converted NIfTI data (sub-NN_ext/flx, template-01)
│   └── subject_mapping.json     # Old→new subject name mapping
├── results/                     # Evaluation outputs
│   ├── atlas_baseline/          # Atlas-based LOOCV (best current)
│   ├── jlf/                     # JLF experiment results
│   ├── nnunet/                  # nnU-Net experiment results
│   └── archive/                 # Legacy results (pre-rename)
└── figures/                     # Generated figures
```

## Current Results (LOOCV, 7 extension subjects)
- **Smart single-atlas: 0.658 mean Dice** (best method)
- Smart multi-atlas top-3: 0.632 mean Dice
- Best muscles: FDP (0.83), ED (0.80), ECU (0.78), SUP (0.75), FCU (0.75)
- Weakest: PL (0.42), PT (0.42), ECRB (0.48) — small/variable muscles
- Runtime: ~20 min per new MRI (single atlas)

## Active Roadmap
1. **Stage 1 (next):** Joint Label Fusion (JLF) — replace majority voting with `ants.joint_label_fusion()`. Expected: +0.02-0.05 Dice.
2. **Stage 2:** More atlases — 10-15 annotated subjects. Expected: +0.05-0.10 Dice.
3. **Stage 3:** nnU-Net — train on all manual segmentations. Expected: 0.80-0.85 Dice.
4. **Stage 4:** Foundation model fine-tuning. Expected: 0.88-0.93 Dice.

## Conventions
- **Shared code:** Import from `utils/` — `from utils import LABELS, compute_dices, register_pair`
- **New experiments:** Place scripts in `experiments/`, save results to `results/<experiment_name>/`
- **Paths:** Use `utils.REPO_ROOT` or `utils.DATA_DIR` for repo-relative paths
- **Running scripts from subdirectories:** Scripts use `sys.path.insert(0, ...)` to find utils

## Technical Stack
- Python 3.10 (ANTsPy, nibabel, pydicom, scipy, scikit-image)
- RTX 2080 Ti available (not used by ANTsPy — CPU-only registration)
- Registration at 0.5mm working resolution (~12-22 min/pair on 6-core CPU)
- NIfTI format for all volumes and masks
