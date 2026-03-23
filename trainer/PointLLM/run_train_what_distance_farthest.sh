#!/usr/bin/env bash
set -euo pipefail

# ===== Paths =====
# 数据/模型路径：迁移机器或切实验数据集时优先改这里。
POINTLLM_ROOT=/home/wangxingjian/PointQA_Eval/trainer/PointLLM
DATASET_ROOT=/home/wangxingjian/PointQA_Eval/what_distance_farthest

# 基座 LLM 初始化权重目录（Stage-1 起点）。
MODEL_NAME_OR_PATH=/home/wangxingjian/model/PointLLM_7B_v1.1_init
# 点云 backbone 初始化权重（PointBERT）。
POINT_BACKBONE_CKPT=/home/wangxingjian/model/PointLLM_7B_v1.1_init/point_bert_v1.2.pt

# 训练输入与标注。
DATA_PATH=/home/wangxingjian/PointQA_Eval/trainer/PointLLM/data/objaverse_data_what_distance_farthest
ANNO_STAGE1=/home/wangxingjian/PointQA_Eval/trainer/PointLLM/data/anno_data/PointQA_what_distance_farthest_stage1.json
ANNO_STAGE2=/home/wangxingjian/PointQA_Eval/trainer/PointLLM/data/anno_data/PointQA_what_distance_farthest_stage2.json

# 输出目录：不同实验建议改 run_name + 输出目录，避免覆盖。
OUTPUT_STAGE1=/home/wangxingjian/PointQA_Eval/trainer/PointLLM/outputs/PointQA_what_distance_farthest/stage1
OUTPUT_STAGE2=/home/wangxingjian/PointQA_Eval/trainer/PointLLM/outputs/PointQA_what_distance_farthest/stage2

# ===== Runtime =====
# GPU 进程数（通常 = 使用 GPU 数）。OOM 时可先减小。
NPROC_PER_NODE=2
# 指定可见 GPU；做对照实验时常改。
export CUDA_VISIBLE_DEVICES=0,1

MASTER_PORT=$((RANDOM % (65535 - 49152 + 1) + 49152))
# 可选：none / wandb / tensorboard
REPORT_TO=none
# 多卡下 Stage-2 是否启用 FSDP：true / false / auto
USE_FSDP_STAGE2=auto

FSDP_ARGS=()
if [[ "$USE_FSDP_STAGE2" == "true" || ("$USE_FSDP_STAGE2" == "auto" && "$NPROC_PER_NODE" -gt 1) ]]; then
  FSDP_ARGS=(
    --fsdp "full_shard auto_wrap"
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
  )
fi

cd "$POINTLLM_ROOT"

if [[ -f "$ANNO_STAGE1" && -f "$ANNO_STAGE2" ]]; then
  echo "[1/3] Found existing annotation JSON files, skip data preparation"
else
  echo "[1/3] Prepare PointLLM-format data"
  python3 ./scripts/prepare_what_distance_farthest.py \
    --dataset-root "$DATASET_ROOT" \
    --pointllm-root "$POINTLLM_ROOT" \
    --pointnum 8192
fi

if [[ ! -d "$MODEL_NAME_OR_PATH" ]]; then
  echo "Missing model init checkpoint dir: $MODEL_NAME_OR_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_STAGE1" "$OUTPUT_STAGE2"

# ===== Stage-1 实验常改参数 =====
# 1) --num_train_epochs
# 2) --per_device_train_batch_size / --gradient_accumulation_steps（一起决定有效 batch）
# 3) --learning_rate（Stage-1 常偏大）
# 4) --fix_llm / --fix_pointnet（冻结策略）
# 5) --bf16 / --gradient_checkpointing（显存与速度权衡）
# 6) --evaluation_strategy / --save_strategy（是否做验证与存 ckpt）
echo "[2/3] Stage-1 training"
PYTHONPATH="$POINTLLM_ROOT:${PYTHONPATH:-}" \
torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" pointllm/train/train_mem.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --data_path "$DATA_PATH" \
  --anno_path "$ANNO_STAGE1" \
  --output_dir "$OUTPUT_STAGE1" \
  --version v1 \
  --model_max_length 2048 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "no" \
  --save_steps 2400 \
  --save_total_limit 1 \
  --learning_rate 2e-3 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --bf16 True \
  --fix_llm True \
  --fix_pointnet True \
  --gradient_checkpointing True \
  --report_to "$REPORT_TO" \
  --run_name "PointQA_what_distance_farthest_stage1" \
  --point_backbone_ckpt "$POINT_BACKBONE_CKPT" \
  --use_color True

# ===== Stage-2 实验常改参数 =====
# 1) --per_device_train_batch_size（Stage-2 更易 OOM）
# 2) --learning_rate（Stage-2 常偏小）
# 3) --fix_llm（True=只训 projector；False=微调 LLM）
# 4) USE_FSDP_STAGE2 / NPROC_PER_NODE（多卡并行策略）
# 5) --run_name / --output_dir（实验管理，避免覆盖）
echo "[3/3] Stage-2 training"
PYTHONPATH="$POINTLLM_ROOT:${PYTHONPATH:-}" \
torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" pointllm/train/train_mem.py \
  --model_name_or_path "$OUTPUT_STAGE1" \
  --data_path "$DATA_PATH" \
  --anno_path "$ANNO_STAGE2" \
  --output_dir "$OUTPUT_STAGE2" \
  --version v1 \
  --model_max_length 2048 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --eval_steps 100 \
  --save_strategy "no" \
  --save_steps 2400 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --bf16 True \
  --fix_llm False \
  --fix_pointnet True \
  --report_to "$REPORT_TO" \
  --run_name "PointQA_what_distance_farthest_stage2" \
  --gradient_checkpointing True \
  --stage_2 True \
  --conversation_types "single_round" \
  --use_color True \
  "${FSDP_ARGS[@]}"
