# Data Summary

## Overview

| Property | Dataset 2 (subjects) | Template 1 |
|----------|---------------------|------------|
| Source | MRI_Segmentation_2 | PD_PROPELLER_5MM_FATS_FLX_0012 |
| Subjects | 7 x 2 positions = 14 scans | 1 scan |
| Sequence | T2-weighted TSE | PD-weighted PROPELLER |
| Matrix | 448 x 448 x 94 | 320 x 320 x 57 |
| Voxel size | 0.268 x 0.268 x 3.0 mm | 0.438 x 0.438 x 6.0 mm |
| Muscles labeled | 17 | 23 (+ skin, bone) |
| Size on disk | ~25 MB per scan | ~7 MB |
| Total | ~357 MB | |

## Subject Mapping

Contiguous IDs sorted by original study number. See `data/subject_mapping.json`.

| ID | Original study ID | Extension | Flexion | Muscles | Notes |
|----|-------------------|-----------|---------|---------|-------|
| sub-01 | 1 | sub-01_ext | sub-01_flx | 17/17 | |
| sub-02 | 2 | sub-02_ext | sub-02_flx | 17/17 | |
| sub-03 | 3 | sub-03_ext | sub-03_flx | 17/17 | |
| sub-04 | 6 | sub-04_ext | sub-04_flx | 17/17 | |
| sub-05 | 7 | sub-05_ext | sub-05_flx | 17/17 | |
| sub-06 | 9 | sub-06_ext | sub-06_flx | 16/17 | Missing PL; anatomy outlier (lowest Dice) |
| sub-07 | 10 | sub-07_ext | sub-07_flx | 17/17 | |
| template-01 | PD_PROPELLER | — | — | 23 | Different protocol; not used in LOOCV |

## File Structure

Each subject directory contains:
- `mri.nii.gz` — MRI volume (float32)
- `labels.nii.gz` — Integer label volume (17 muscle IDs)

Template additionally has `full.nii.gz` (23 labels including skin/bone).

## Muscle Labels

17 forearm muscles with sparse integer IDs (no labels at 1 or 13):

| ID | Muscle | Full name | Mean volume (voxels) | Size range |
|----|--------|-----------|---------------------|------------|
| 2 | ANC | Anconeus | 31,397 | 19,583 – 49,220 |
| 3 | APL | Abductor pollicis longus | 61,950 | 40,721 – 78,224 |
| 4 | ECRB | Extensor carpi radialis brevis | 29,335 | 5,668 – 42,317 |
| 5 | ECRL | Extensor carpi radialis longus | 138,819 | 85,855 – 173,301 |
| 6 | ECU | Extensor carpi ulnaris | 66,519 | 42,671 – 85,005 |
| 7 | ED | Extensor digitorum | 107,143 | 70,927 – 131,907 |
| 8 | EDM | Extensor digiti minimi | 20,819 | 13,390 – 30,319 |
| 9 | EPL | Extensor pollicis longus | 36,565 | 17,443 – 62,440 |
| 10 | FCR | Flexor carpi radialis | 100,394 | 43,561 – 152,366 |
| 11 | FCU | Flexor carpi ulnaris | 121,146 | 80,825 – 168,064 |
| 12 | FDP | Flexor digitorum profundus | 328,177 | 251,813 – 433,784 |
| 14 | FDS | Flexor digitorum superficialis | 257,832 | 155,862 – 326,998 |
| 15 | FPL | Flexor pollicis longus | 90,049 | 64,031 – 111,709 |
| 16 | PL | Palmaris longus | 34,554 | 0 – 63,783 |
| 17 | PQ | Pronator quadratus | 31,564 | 23,156 – 49,412 |
| 18 | PT | Pronator teres | 48,399 | 23,864 – 79,804 |
| 19 | SUP | Supinator | 95,589 | 76,183 – 121,396 |

**Largest muscles:** FDP, FDS, ECRL, FCU, ED (>100k voxels mean)
**Smallest muscles:** EDM, ECRB, ANC, PQ, PL (<35k voxels mean)
**Most variable:** ECRB (7.5x range), PL (absent in sub-06), FCR (3.5x range)

## Current Results (LOOCV, 7 extension subjects)

| Method | Mean Dice | Notes |
|--------|-----------|-------|
| Atlas single (best match) | 0.658 | ~20 min/subject |
| Atlas multi-3 (majority vote) | 0.632 | ~50 min/subject |
| JLF top-3 | 0.517 | Did not improve over atlas |
| nnU-Net (fold 0 only) | 0.804 | 1 held-out subject |
| **Target** | **0.900** | |

### Per-muscle tiers (atlas single baseline)

- **Good (>0.70):** FDP (0.83), ED (0.80), ECU (0.78), SUP (0.75), FCU (0.75), ECRL (0.73)
- **Moderate (0.55–0.70):** FDS (0.69), FPL (0.66), FCR (0.65), APL (0.63), EPL (0.61), ANC (0.60), PQ (0.58), EDM (0.55)
- **Needs work (<0.55):** ECRB (0.48), PT (0.42), PL (0.42)
