export CUDA_VISIBLE_DEVICES=0
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
python main.py \
  --model_name pointllm \
  --tasks_file ./what_distance_farthest/tasks.jsonl \
  --point_cloud_dir ./what_distance_farthest/pcd \
  --checkpoint_path /home/wangxingjian/model/PointLLM_7B_v1.2 \
  --output_dir ./eval_results/pointllm \
  --device cuda