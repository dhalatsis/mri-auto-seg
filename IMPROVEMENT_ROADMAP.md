# Improvement Roadmap: From 0.66 to 0.90 Dice

## Current State

| Metric | Value |
|--------|-------|
| Overall mean Dice | 0.658 |
| Best muscle (FDP) | 0.826 |
| Worst muscle (PL) | 0.421 |
| Best subject (S3) | 0.707 |
| Worst subject (S9) | 0.546 |
| Atlas pool size | 7 subjects |
| Runtime per subject | ~20 min (CPU) |

### Three Tiers of Muscle Performance

| Tier | Muscles | Mean Dice | Character |
|------|---------|-----------|-----------|
| Good (>0.70) | FDP, ED, ECU, FCU, SUP, ECRL, FDS, APL, PQ, FPL | 0.74 | Large, central, consistent anatomy |
| Moderate (0.55-0.70) | ANC, EPL, FCR, EDM | 0.60 | Medium-size, higher variance |
| Needs Work (<0.55) | ECRB, PT, PL | 0.44 | Small, variable, sometimes absent |

---

## Improvement Strategies (Ordered by Effort/Impact)

### 1. More Atlases (Low Effort, High Impact)

**Expected gain: +0.05-0.10 overall Dice**

Currently we have only 7 atlases. The outlier subject (S9, Dice=0.546) has no good anatomical match in our pool. With more atlases:

- Better atlas matches for every target (especially outliers)
- Multi-atlas fusion becomes viable (currently 3 noisy atlases hurt more than help)
- Recommendation: **10-15 annotated subjects** should substantially improve atlas matching

**Why it helps**: S9's ECRB=0.13 and FCR=0.16 are catastrophically bad — a single better-matched atlas would likely bring these to 0.5+. The NCC-based selection already works well; it just needs more options.

**Human effort**: ~10-15 hours per new subject for manual segmentation. Priority: annotate subjects with diverse anatomies (large forearms, small forearms, different muscle proportions).

### 2. Joint Label Fusion (Medium Effort, Medium Impact)

**Expected gain: +0.02-0.05 overall Dice**

Replace majority voting with **Joint Label Fusion (JLF)** or **STAPLE**:

- JLF weights each atlas's contribution by **local** (patch-wise) similarity to the target
- An atlas that matches well in one region but poorly in another gets high weight only where it matches
- ANTsPy has JLF built-in (`ants.joint_label_fusion()`)
- STAPLE estimates per-voxel label confidence from multiple raters

**Why it helps**: With majority voting, a poorly-registered atlas can outvote a good one. JLF would selectively trust each atlas where it's actually accurate.

**Human effort**: Minimal — algorithmic change, requires registering all (or top-5) atlases to each target.

### 3. Boundary Refinement Post-Processing (Medium Effort, Medium Impact)

**Expected gain: +0.02-0.04 overall Dice**

The error analysis shows most errors are at **muscle boundaries**, not interiors. Targeted fixes:

- **Active contour / level set refinement**: Initialize from atlas prediction, refine boundaries using image gradients
- **Graph-cut / CRF smoothing**: Add spatial and intensity constraints to clean noisy boundaries
- **Boundary-aware morphological operations**: Targeted erosion/dilation at inter-muscle boundaries
- **Slice consistency**: Enforce smooth muscle contour transitions between adjacent slices (muscles don't jump around)

**Why it helps**: The bulk of each muscle is correctly identified; only the boundaries between adjacent muscles are blurred. Even simple smoothing could help.

### 4. Registration Improvements (Medium Effort, Medium Impact)

**Expected gain: +0.03-0.06 overall Dice**

Current SyN parameters may not be optimal:

- **Higher working resolution**: Currently using 0.5mm in-plane. Going to 0.3mm would capture finer boundary details (but 4x slower)
- **Label-guided registration**: Use the atlas labels to constrain registration (muscles shouldn't cross boundaries)
- **Multi-channel registration**: Register both MRI intensity and tissue-type probability maps simultaneously
- **Landmark initialization**: Manual or automatic identification of a few anatomical landmarks (e.g., radius, ulna) to initialize registration better
- **Per-muscle local registration**: After global SyN, run smaller local registrations focused on each muscle region

**Human effort**: Landmark identification could be semi-automated (bone detection), but per-muscle refinement requires significant engineering.

### 5. Deep Learning — The Big Jump (High Effort, Highest Impact)

**Expected gain: +0.15-0.25 overall Dice (to 0.80-0.90)**

This is the path to 0.90. Two approaches:

#### 5a. nnU-Net (Recommended First Step)

- **nnU-Net** auto-configures architecture, preprocessing, training for any medical segmentation task
- Train on our 7 manually-segmented subjects using 7-fold cross-validation
- Can be augmented with atlas-propagated predictions as additional training data
- State-of-the-art in many medical segmentation benchmarks
- **Expected Dice with 7 subjects: 0.70-0.80**
- **Expected Dice with 15+ subjects: 0.80-0.90**

#### 5b. Pretrained Foundation Models

- **TotalSegmentator** / **STU-Net** / **UniSeg**: Large models pretrained on thousands of CT/MRI scans
- Fine-tune on our forearm data (transfer learning)
- Even with only 7 subjects, fine-tuning a pretrained model can be very effective
- MedSAM / SAM-Med3D: Segment-anything models adapted for medical imaging; can use prompted segmentation

#### 5c. Semi-Supervised / Self-Training

1. Use atlas-based pipeline to generate pseudo-labels for unlabeled MRIs
2. Have expert correct the worst mistakes (focus on small muscles: PL, PT, ECRB)
3. Train nnU-Net on both manual labels + corrected pseudo-labels
4. Repeat — each iteration improves pseudo-labels

**This human-in-the-loop cycle is the most efficient path to 0.90.**

### 6. GPU-Accelerated Registration (High Effort, Runtime Impact)

**Expected gain: Minimal Dice improvement, but 100x faster**

- **VoxelMorph**: Learning-based registration, GPU inference in seconds
- **TransMorph**: Transformer-based registration
- Train on our atlas pairs, then apply to new subjects in <10 seconds
- Doesn't directly improve accuracy but enables real-time segmentation + more atlas comparisons

---

## Where Human Touch Has the Most Impact

### High-Value Manual Corrections (Best ROI)

| Priority | Action | Impact | Time |
|----------|--------|--------|------|
| 1 | **Annotate 6-8 more subjects** | +0.05-0.10 Dice overall | 60-120 hrs |
| 2 | **Correct PL/PT/ECRB in auto-predictions** | Brings worst muscles from 0.42 to 0.65+ | 1-2 hrs/subject |
| 3 | **Verify/fix boundary regions between adjacent muscles** | +0.03 Dice on boundary errors | 1 hr/subject |
| 4 | **Identify and annotate anatomical variants** (absent PL, thin PT) | Handles outlier subjects | 2-3 hrs |

### Why Manual Correction is Strategic

The current pipeline gets **interior regions correct** (see error analysis). The expert doesn't need to segment from scratch — they just need to:
1. Fix muscle **boundaries** (drag contours a few pixels)
2. Fill in **missed small muscles** (PL, PT, ECRB)
3. Remove **false positives** at muscle ends (proximal/distal slices)

This "correction" workflow is **5-10x faster** than manual segmentation from scratch.

---

## Realistic Improvement Targets

| Stage | Method | Expected Dice | Timeline |
|-------|--------|---------------|----------|
| Current | Smart single-atlas (7 subjects) | **0.658** | Done |
| Stage 1 | + More atlases (15 subjects) + JLF | **0.72-0.75** | 2-3 months |
| Stage 2 | + nnU-Net (15 subjects, 5-fold CV) | **0.80-0.85** | 1-2 months after Stage 1 |
| Stage 3 | + Semi-supervised self-training + corrections | **0.85-0.90** | 2-3 months after Stage 2 |
| Stage 4 | + Pretrained foundation model fine-tuning | **0.88-0.93** | 1-2 months after Stage 3 |

### Key Insight

**The bottleneck is training data, not the algorithm.** With 7 subjects, even perfect algorithms are limited. The single most impactful investment is annotating more subjects. Combined with deep learning (nnU-Net), 15 well-annotated subjects could realistically achieve 0.85+ Dice.

---

## Muscle-Specific Strategies

### Muscles Close to 0.90 (FDP, ED, ECU — need +0.07-0.12)

These just need:
- Better atlas matching (more atlases)
- Boundary refinement
- Possibly reachable with atlas-based methods alone

### Muscles at 0.70-0.80 (FCU, SUP, ECRL, FDS, APL, PQ, FPL)

These need:
- More atlases for consistent matching
- Deep learning to learn local appearance features
- The registration captures overall shape but misses local deformations

### Muscles Below 0.55 (ECRB, PT, PL)

These require fundamentally different approaches:
- **PL**: Sometimes anatomically absent (~14% of people). Need atlas selection that accounts for this, or a classifier that detects PL presence/absence
- **PT**: Thin, only present in proximal slices. Gets confused with surrounding muscles. Need slice-level attention
- **ECRB**: High variance (0.13 to 0.73 across subjects). Boundary with ECRL is ambiguous on MRI. May need higher resolution or multi-contrast imaging

For these muscles, **deep learning + expert corrections** is likely the only path to 0.80+.

---

## Summary: What to Do Next

1. **Immediate** (no extra data needed):
   - Try JLF instead of majority voting
   - Optimize SyN parameters (try higher resolution, different metrics)
   - Add boundary refinement post-processing

2. **Short-term** (needs human time):
   - Annotate 6-8 more subjects (diverse anatomies)
   - Re-run atlas-based pipeline with larger pool

3. **Medium-term** (needs GPU + engineering):
   - Train nnU-Net on all available manual segmentations
   - Set up semi-supervised self-training loop

4. **Long-term** (research):
   - Fine-tune medical foundation models
   - GPU registration for real-time segmentation
   - Multi-contrast MRI for ambiguous boundaries
