#!/bin/bash
# Train folds 1-6 of nnU-Net sequentially (fold 0 already running).
# Usage: nohup bash experiments/train_remaining_folds.sh > results/nnunet/folds_1_6.log 2>&1 &

set -e

export nnUNet_raw=/home/dc23/projects/mri-auto-seg/nnUNet_data/nnUNet_raw
export nnUNet_preprocessed=/home/dc23/projects/mri-auto-seg/nnUNet_data/nnUNet_preprocessed
export nnUNet_results=/home/dc23/projects/mri-auto-seg/nnUNet_data/nnUNet_results

# Wait for fold 0 to finish (check for checkpoint_final.pth)
FOLD0_DIR="$nnUNet_results/Dataset001_ForearmMuscles/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0"
echo "Waiting for fold 0 to complete..."
while [ ! -f "$FOLD0_DIR/checkpoint_final.pth" ]; do
    sleep 300
done
echo "Fold 0 complete! Starting remaining folds at $(date)"

for fold in 1 2 3 4 5 6; do
    echo "=============================================="
    echo "Training fold $fold / 6 at $(date)"
    echo "=============================================="
    nnUNetv2_train 1 3d_fullres $fold --npz
    echo "Fold $fold complete at $(date)"
done

echo ""
echo "All folds complete at $(date)!"
echo "Run: python3 experiments/eval_nnunet.py"
