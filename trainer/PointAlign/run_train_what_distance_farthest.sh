#!/usr/bin/env bash
set -euo pipefail

# Edit values directly if needed.
POINTALIGN_ROOT=/home/wangxingjian/PointQA_Eval/trainer/PointAlign
DATASET_ROOT=/home/wangxingjian/PointQA_Eval/what_distance_farthest
CUDA_VISIBLE_DEVICES=0

cd "$POINTALIGN_ROOT"

echo "[1/2] Prepare PointAlign-format data"
python3 ./scripts/prepare_what_distance_farthest.py \
  --dataset-root "$DATASET_ROOT" \
  --pointalign-root "$POINTALIGN_ROOT" \
  --pointnum 8192

echo "[2/2] Start PointAlign training (test config)"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 train.py --cfg-path ./finetune_what_distance_farthest.yaml
