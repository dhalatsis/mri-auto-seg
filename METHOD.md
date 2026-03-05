# Automatic Forearm MRI Muscle Segmentation

## Method: Atlas-Based Registration with Smart Atlas Selection

### Overview

This project automates the segmentation of 17 forearm muscles in MRI volumes. Manual segmentation of these muscles typically requires 10-15 hours per subject by an expert anatomist. Our automated pipeline reduces this to ~20 minutes per subject using atlas-based registration.

The core idea is simple: if we have one or more MRIs where the muscles are already manually labeled (the "atlases"), we can use image registration to find a spatial mapping between an atlas and a new (unlabeled) target MRI. Applying this same mapping to the atlas labels produces an automatic segmentation of the target.

### Data

- **7 subjects** with bilateral forearm MRI scans in extension and flexion positions (14 scans total)
- **MRI acquisition**: T2-weighted TSE, 448x448 matrix, 0.268mm in-plane resolution, 3mm slice thickness, 78-94 slices
- **17 muscles** manually segmented as ImageJ ROI polygon contours, converted to NIfTI label volumes
- Muscles: ANC, APL, ECRB, ECRL, ECU, ED, EDM, EPL, FCR, FCU, FDP, FDS, FPL, PL, PQ, PT, SUP

### Pipeline Steps

#### 1. Atlas Selection (Fast Affine Ranking)

Not all atlases are equally good for a given target. A subject with a large forearm will match better with another large forearm. We rank atlases by similarity:

1. Downsample both atlas and target MRIs to 1mm isotropic resolution (fast)
2. Run affine registration (rigid + affine transform, ~7 seconds per pair)
3. Compute normalized cross-correlation (NCC) between the warped atlas and target
4. Select the atlas with highest NCC as the best match

This step takes ~1 minute total (6 affine registrations).

#### 2. SyN Deformable Registration

Using the best atlas, we perform full deformable registration with ANTsPy's Symmetric Normalization (SyN) algorithm:

1. **Resample** both images to working resolution (0.5mm in-plane, 3mm slice)
2. **Histogram matching** - normalize atlas intensity distribution to match the target
3. **SyN registration** - iterative optimization that finds a smooth, invertible deformation field
   - Affine initialization (Mattes mutual information metric)
   - Deformable SyN refinement (cross-correlation metric, 4-voxel neighborhood)
   - Multi-resolution: 100 -> 70 -> 50 -> 10 iterations
4. The result is a deformation field mapping every voxel in the atlas to its corresponding location in the target

This step takes ~12-22 minutes depending on the subject pair.

#### 3. Label Propagation

The deformation field from step 2 is applied to the atlas's label volume:

1. Apply forward transforms (affine + deformation field) to the atlas labels
2. Use **nearest-neighbor interpolation** (`genericLabel` interpolator) to preserve discrete label values
3. Apply at the target's original (full) resolution for maximum accuracy

This step is fast (<10 seconds).

#### 4. Post-Processing (Optional)

- Remove small disconnected components (< 50 voxels) per muscle label
- Fill small holes within each label (2D per slice to preserve anatomy)

### Multi-Atlas Fusion

When using multiple atlases, we register the top-3 ranked atlases independently and fuse their predictions via **majority voting**: at each voxel, the final label is the one predicted by the majority of atlases. This can improve robustness but we found that with our current data, the best single atlas outperforms multi-atlas fusion (see results below).

### Results (Leave-One-Out Cross-Validation)

Each of the 7 extension subjects is held out as a target, and the remaining 6 serve as potential atlases. The best atlas is selected by NCC ranking.

#### Overall Results

| Method | Mean Dice | Runtime per Subject |
|--------|-----------|-------------------|
| Smart Single-Atlas (best) | **0.658** | ~20 min |
| Smart Multi-Atlas (top-3) | 0.632 | ~50 min |
| Before slice fix (invalid) | 0.394 | - |

#### Per-Subject Results (Smart Single-Atlas)

| Subject | Mean Dice | Best Atlas |
|---------|-----------|------------|
| Subject 10 | 0.676 | Subject 1 |
| Subject 1 | 0.652 | Subject 7 |
| Subject 2 | 0.687 | Subject 7 |
| Subject 3 | 0.707 | Subject 10 |
| Subject 6 | 0.686 | Subject 10 |
| Subject 7 | 0.649 | Subject 1 |
| Subject 9 | 0.546 | Subject 7 |

#### Per-Muscle Results (Smart Single-Atlas, Mean +/- Std)

| Muscle | Dice | Notes |
|--------|------|-------|
| FDP | 0.826 +/- 0.023 | Best - large, central muscle |
| ED | 0.798 +/- 0.045 | Large extensor |
| ECU | 0.776 +/- 0.052 | |
| SUP | 0.748 +/- 0.039 | |
| FCU | 0.752 +/- 0.054 | |
| ECRL | 0.742 +/- 0.059 | |
| FDS | 0.741 +/- 0.074 | |
| APL | 0.703 +/- 0.068 | |
| PQ | 0.699 +/- 0.047 | |
| FPL | 0.690 +/- 0.021 | Very consistent |
| ANC | 0.647 +/- 0.108 | |
| EPL | 0.628 +/- 0.085 | |
| FCR | 0.557 +/- 0.186 | High variance |
| EDM | 0.548 +/- 0.096 | Small muscle |
| ECRB | 0.481 +/- 0.179 | High variance |
| PT | 0.421 +/- 0.202 | Often absent/thin |
| PL | 0.421 +/- 0.215 | Small, sometimes absent |

### Key Observations

1. **Atlas selection matters enormously.** The NCC-based ranking consistently identifies the best atlas, and a well-matched single atlas outperforms poorly-matched multi-atlas fusion.

2. **Large muscles segment well** (FDP, ED, ECU, FCU all > 0.75 Dice). These have consistent anatomy and clear boundaries.

3. **Small/variable muscles are harder** (PL, PT, ECRB < 0.50 Dice). These muscles vary in size across individuals, and PL is anatomically absent in ~14% of people.

4. **Subject 9 is an outlier** (0.546 mean Dice). It has notably different anatomy, particularly with very low ECRB (0.13) and FCR (0.16) scores, suggesting unusual muscle geometry.

5. **More atlases should help.** With 10-13 annotated subjects (as planned), we expect better atlas matches for outlier subjects, likely pushing overall Dice above 0.70.

### Technical Stack

- **ANTsPy** (antspyx): Deformable image registration (SyN algorithm)
- **NiBabel**: NIfTI file I/O
- **PyDICOM**: DICOM file parsing
- **NumPy/SciPy**: Numerical operations, morphological post-processing
- **scikit-image**: Polygon rasterization for ROI conversion

### Usage

```bash
# Segment a new MRI (NIfTI)
python auto_segment.py input.nii.gz -o segmentation.nii.gz

# Segment from DICOM directory
python auto_segment.py /path/to/dicoms/ -o segmentation.nii.gz

# Quick mode (best 2 atlases)
python auto_segment.py input.nii.gz -o seg.nii.gz --quick
```

### Limitations and Future Work

1. **CPU-only registration**: ANTsPy SyN runs on CPU (~20 min/subject). GPU-based registration (e.g., VoxelMorph) could reduce this to seconds but requires retraining.

2. **Small atlas pool**: With only 7 subjects, outlier anatomies aren't well represented. Adding 6+ more atlases should substantially improve accuracy.

3. **No deep learning yet**: A U-Net trained on the atlas-registered data could further improve accuracy and speed. The current pipeline provides excellent training data for this.

4. **Extension only**: Current evaluation is on wrist extension position. Flexion data is available but not yet evaluated.

5. **Joint Label Fusion (JLF)**: A more sophisticated fusion method that weights atlas contributions by local similarity. ANTsPy supports this but it's slower. Could improve multi-atlas results.
