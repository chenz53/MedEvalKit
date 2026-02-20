#!/bin/bash
# Evaluate smb-vision-v1 via linear probing on CT-RATE.
# Volumes are lazy-loaded; pooled features are cached for fast re-runs.
#
# Single GPU:   bash eval_encoder.sh
# Multi-GPU:    accelerate config && bash eval_encoder.sh

accelerate launch eval_encoder.py \
    --dataset_path ./datas/CT_RATE \
    --batch_size 2 \
    --num_workers 4 \
    --lr 1e-3 \
    --epochs 100 \
    --seed 42
