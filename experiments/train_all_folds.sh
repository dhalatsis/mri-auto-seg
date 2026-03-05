#!/bin/bash
# Train all 7 folds of nnU-Net sequentially.
# Usage: nohup bash experiments/train_all_folds.sh > results/nnunet/all_folds.log 2>&1 &

set -e

export nnUNet_raw=/home/dc23/projects/mri-auto-seg/nnUNet_data/nnUNet_raw
export nnUNet_preprocessed=/home/dc23/projects/mri-auto-seg/nnUNet_data/nnUNet_preprocessed
export nnUNet_results=/home/dc23/projects/mri-auto-seg/nnUNet_data/nnUNet_results

for fold in 0 1 2 3 4 5 6; do
    echo "=============================================="
    echo "Training fold $fold / 6"
    echo "=============================================="
    # --c will continue if checkpoint exists, start fresh otherwise
    nnUNetv2_train 1 3d_fullres $fold --npz --c
    echo "Fold $fold complete at $(date)"
done

echo ""
echo "All 7 folds complete!"
echo "Run: python3 experiments/eval_nnunet.py"
