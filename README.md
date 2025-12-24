# PointQA_Eval

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 2. virtual environment

Choose your environment path and create it:

```bash
cd PointQA_Eval

# Choose your environment path and create it:
bash ./scripts/setup_env.sh ~/.virtualenvs/pointqa_eval/dev
source scripts/activate_env.sh ~/.virtualenvs/pointqa_eval/dev
```

### 3. Install package
```bash
uv sync --active \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match
```

#### Install Pointnet2_PyTorch

First check this [issue](https://github.com/erikwijmans/Pointnet2_PyTorch/issues/174)

```bash
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib
uv pip install -e . --no-build-isolation
```

### 4. Download Recon++ weight(ShapeLLM)

```bash
bash recon_download.sh
```

Then the weight file should be located at: PointQA_Eval/checkpoints/recon/large.pth

## Quick Start

```bash
# run eval
bash run_eval.sh
```

### GPT4Point 多项选择评测

GPT4Point 依赖完整的 [LAVIS](https://github.com/salesforce/LAVIS) 子模块以及 `others` 中的点云编码器，这些内容已经镜像到 `models/dependence/gpt4point/` 下。首次使用会通过 `huggingface_hub` 自动下载 Point-BERT 等权重，请确保能够访问镜像或自行配置 `HF_ENDPOINT`。

运行命令示例（使用 `what_distance_farthest` 中的任务与点云）：

```bash
python main.py \
        --model_name gpt4point \
        --tasks_file ./what_distance_farthest/tasks.jsonl \
        --point_cloud_dir ./what_distance_farthest/pcd \
        --checkpoint_path /path/to/gpt4point_pretrain_stage2_opt2.7b.pth \
        --output_dir ./eval_results/gpt4point \
        --device cuda \
        --num_beams 5 \
        --max_length 30 \
        --min_length 1
```

可选参数：

- `--gpt4point_model_type`：指定 LAVIS 模型类型，默认 `gpt4point_opt2.7b`。
- `--use_text_processor`：设为 `True` 时会复用 LAVIS 的 `blip_caption` 文本预处理；默认关闭以保持题目原文。
- 生成相关超参（`--num_beams`, `--top_p`, `--temperature`, `--length_penalty`, `--repetition_penalty` 等）会直接透传给 GPT4Point 的 `generate` 接口。

评测脚本会读取 `tasks.jsonl` 中的问题、选项与标准答案，并根据 `point_cloud_dir` 中的 `.npy` 点云文件执行批量推理，自动汇总 Accuracy。