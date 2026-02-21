#!/bin/bash
# Evaluate smb-vision-v1 via linear probing on CT-RATE.
# Volumes are lazy-loaded; pooled features are cached for fast re-runs.
#
# Single GPU:   bash eval_encoder.sh
# Multi-GPU:    accelerate config && bash eval_encoder.sh
export WANDB_PROJECT="smb-vision"

DATA=/workspace/data/CR-RATE

accelerate launch eval_encoder.py \
    --dataset_path "$DATA" \
    --train_label_csv "$DATA/dataset/multi_abnormality_labels/train_predicted_labels.csv" \
    --val_label_csv "$DATA/dataset/multi_abnormality_labels/valid_predicted_labels.csv" \
    --head_batch_size 1024 \
    --num_workers 16 \
    --lr 1e-2 \
    --weight_decay 1e-5 \
    --epochs 5000 \
    --seed 42
