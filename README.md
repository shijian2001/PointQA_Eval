# PointQA_Eval

## 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

## 2. Supported Models

### 2.1 Pointllm

#### 2.1.1 Virtual Environment

```bash
cd PointQA_Eval

uv venv ~/.virtualenvs/pointqa_eval/pointllm --python 3.9
source ~/.virtualenvs/pointqa_eval/pointllm/bin/activate
```

#### 2.1.2 Install Packages

```bash
uv --project envs/pointllm sync
```

#### 2.1.3 Download checkpoints

```bash
hf download RunsenXu/PointLLM_7B_v1.2 --local-dir /path/PointLLM_7B_v1.2
```

#### 2.1.4 Run Evaluation
```bash
python3 main.py \
  --model_name pointllm \
  --tasks_file ./what_distance_farthest/tasks.jsonl \
  --point_cloud_dir ./what_distance_farthest/pcd \
  --checkpoint_path /home/wangxingjian/model/PointLLM_7B_v1.2 \
  --output_dir ./eval_results/pointllm \
  --device cuda
```

### 2.2 Shapellm

#### 2.2.1 Virtual Environment

Choose your environment path and create it:

```bash
cd PointQA_Eval

bash ./scripts/setup_env.sh ~/.virtualenvs/pointqa_eval/dev
source scripts/activate_env.sh ~/.virtualenvs/pointqa_eval/dev
```

#### 2.2.2 Install Packages
```bash
uv sync --active \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match
```
Then use [flash-finder](https://flashattn.dev/#finder) to find the right version of flash-attn for your environment, and install it using `uv pip install flash-attn==<version> --no-build-isolation`.

#### 2.2.3 Install Pointnet2_PyTorch

First, check this [issue](https://github.com/erikwijmans/Pointnet2_PyTorch/issues/174).

```bash
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib
uv pip install -e . --no-build-isolation
```

#### 2.2.4 Download Recon++ Weights for ShapeLLM

```bash
bash recon_download.sh
```

After downloading, the weight file should be located at `PointQA_Eval/checkpoints/recon/large.pth`.

### 2.3 GreenPLM

#### 2.3.1 Virtual Environment

Choose your environment path and create it:

```bash
cd PointQA_Eval

bash ./scripts/setup_env.sh ~/.virtualenvs/pointqa_eval/greenplm
source scripts/activate_env.sh ~/.virtualenvs/pointqa_eval/greenplm

uv pip install -r requirements_greenplm.txt
```

#### 2.3.2 Install Pointnet2_PyTorch

First, check this [issue](https://github.com/erikwijmans/Pointnet2_PyTorch/issues/174).

```bash
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib
uv pip install -e . --no-build-isolation
```

### 2.4 MiniGPT3D

#### 2.4.1 Virtual Environment
```bash
cd PointQA_Eval

uv --project envs/pointalign sync
```

#### 2.4.2 Update the Model Configuration
1. Check this [issue](https://github.com/TangYuan96/MiniGPT-3D/issues/6), and move [MiniGPT-3D/modeling_phi.py](https://github.com/TangYuan96/MiniGPT-3D/blob/main/modeling_phi.py) to `transformers/models/phi/modeling_phi.py`.
2. Update the model paths in [benchmark_evaluation_paper.yaml](models/dependence/minigpt3d/eval_configs/benchmark_evaluation_paper.yaml).

#### 2.4.3 Modify Local Model Paths

You need to modify the following files:
- [conversation.py](models/dependence/minigpt3d/minigpt4/conversation/conversation.py) line 20
- [base_model.py](models/dependence/minigpt3d/minigpt4/models/base_model.py) line 55

In both files, change the tokenizer loading line to:
```python
tokenizer = AutoTokenizer.from_pretrained("model/MiniGPT-3D/params_weight/Phi_2")
```

Replace the example path with your actual local model path.

### 2.5 OneLLM
#### 2.5.1 Virtual Environment
```bash
cd PointQA_Eval

bash ./scripts/setup_env.sh ~/.virtualenvs/pointqa_eval/onellm
source scripts/activate_env.sh ~/.virtualenvs/pointqa_eval/onellm

uv pip install -r requirements_onellm.txt

# Install pointnet2
git clone https://github.com/csuhan/OneLLM.git
cd OneLLM/model/lib/pointnet2
python setup.py install
```

#### 2.5.2 Download checkpoints

```bash
hf download timm/vit_large_patch14_clip_224.openai --local-dir /model/vit_large_patch14_clip_224

hf download csuhan/OneLLM-7B --local-dir /model/OneLLM-7B 
```

### 2.6 PointAlign
#### 2.6.1 Virtual Environment

Using `uv`:
```bash
cd PointQA_Eval

uv --project envs/pointalign sync
```

Or using `conda`:
```bash
conda env create -f environment.yml
```

#### 2.6.2 Download Checkpoints
```bash
hf download ShijianW01/PointAlign_weight --local-dir /path

hf download Vision-CAIR/minigpt4 blip2_pretrained_flant5xxl.pth --local-dir /path --repo-type=space
```

#### 2.6.3 Update the Model Configuration
PointAlign uses the same underlying framework as MiniGPT3D. Before running evaluation, make sure to:
- update the model paths in [benchmark_evaluation_paper.yaml](models/dependence/pointalign/eval_configs/benchmark_evaluation_paper.yaml)
- move [modeling_phi.py](models/dependence/pointalign/minigpt4/models/modeling_phi.py) to `transformers/models/phi/modeling_phi.py`

## Quick Start

Before running evaluation, update the model paths in the script to match your local environment.
```bash
bash run_eval.sh
```
