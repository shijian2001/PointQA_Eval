export CUDA_VISIBLE_DEVICES=7
export HF_ENDPOINT=https://hf-mirror.com

# 3D-LLaVA
# python main.py \
#   --model_name 3dllava \
#   --tasks_file ./what_distance_farthest/tasks.jsonl \
#   --point_cloud_dir ./what_distance_farthest/pcd \
#   --test_ckpt /home/wangxingjian/model/3D-LLaVA-7B-LoRA \
#   --llava_model_base /home/wangxingjian/model/llava-v1.5-7b \
#   --output_dir ./eval_results/3dllava \
#   --device cuda

# ShapeLLM
# python main.py \
#   --model_name shapellm \
#   --tasks_file ./what_distance_farthest/tasks.jsonl \
#   --point_cloud_dir ./what_distance_farthest/pcd \
#   --test_ckpt /home/wangxingjian/model/ShapeLLM-7B-General \
#   --llava_model_base /home/wangxingjian/model/llava-v1.5-7b \
#   --output_dir ./eval_results/shapellm \
#   --device cuda

# PointLLM
# python main.py \
#   --model_name pointllm \
#   --tasks_file ./what_distance_farthest/tasks.jsonl \
#   --point_cloud_dir ./what_distance_farthest/pcd \
#   --checkpoint_path /home/wangxingjian/model/PointLLM_7B_v1.2 \
#   --output_dir ./eval_results/pointllm \
#   --device cuda

# GPT4Point (OPT2.7B)
# python main.py \
#   --model_name gpt4point \
#   --checkpoint_path /home/wangxingjian/model/GPT4Point/gpt4point_pretrain_stage2_opt2.7b.pth \
#   --cfg_path models/dependence/gpt4point/lavis/projects/gpt4point/eval/pointqa_mcq_opt2.7b_eval.yaml \
#   --tasks_file ./what_distance_farthest/tasks.jsonl \
#   --point_cloud_dir ./what_distance_farthest/pcd \
#   --output_dir ./eval_results/gpt4point \
#   --device cuda \
#   --num_beams 5 \
#   --max_length 30 \
#   --min_length 1

# MiniGPT-3D
# python /home/wangxingjian/PointQA_Eval/main.py \
#   --model_name minigpt3d \
#   --tasks_file ./what_distance_farthest/tasks.jsonl \
#   --point_cloud_dir ./what_distance_farthest/pcd \
#   --cfg_path /home/wangxingjian/PointQA_Eval/models/dependence/minigpt3d/eval_configs/benchmark_evaluation_paper.yaml \
#   --output_dir ./eval_results/minigpt3d \
#   --device cuda

# greenplm
python main.py \
  --model_name greenplm \
  --tasks_file ./what_distance_farthest/tasks.jsonl \
  --point_cloud_dir ./what_distance_farthest/pcd \
  --model_path /home/wangxingjian/PointQA_Eval/cankao/GreenPLM/lava-vicuna_2024_4_Phi-3-mini-4k-instruct \
  --lora_path /home/wangxingjian/PointQA_Eval/cankao/GreenPLM/release/paper/weight/stage_3 \
  --pretrain_mm_mlp_adapter /home/wangxingjian/PointQA_Eval/cankao/GreenPLM/release/paper/weight/stage_3/non_lora_trainables.bin \
  --pc_ckpt_path /home/wangxingjian/PointQA_Eval/cankao/GreenPLM/pretrained_weight/Uni3D_PC_encoder/modelzoo/uni3d-small/model.pt \
  --pc_encoder_type small \
  --get_pc_tokens_way OM_Pooling \
  --output_dir ./eval_results/greenplm \
  --device cuda