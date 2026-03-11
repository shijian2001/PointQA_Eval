# PointQA_Eval

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```
### 2. Shapellm/Pointllm

#### virtual environment

Choose your environment path and create it:

```bash
cd PointQA_Eval

bash ./scripts/setup_env.sh ~/.virtualenvs/pointqa_eval/dev
source scripts/activate_env.sh ~/.virtualenvs/pointqa_eval/dev
```

#### Install package
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

#### Download Recon++ weight(ShapeLLM)

```bash
bash recon_download.sh
```

Then the weight file should be located at: PointQA_Eval/checkpoints/recon/large.pth

### 3. GreenPLM

#### Virtual environment

Choose your environment path and create it:

```bash
cd PointQA_Eval

bash ./scripts/setup_env.sh ~/.virtualenvs/pointqa_eval/greenplm
source scripts/activate_env.sh ~/.virtualenvs/pointqa_eval/greenplm

uv pip install -r requirements_greenplm.txt
```

#### Install Pointnet2_PyTorch

First check this [issue](https://github.com/erikwijmans/Pointnet2_PyTorch/issues/174)

```bash
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib
uv pip install -e . --no-build-isolation
```

### 4. MiniGPT3D

#### Virtual environment
```bash
cd PointQA_Eval

bash ./scripts/setup_env.sh ~/.virtualenvs/pointqa_eval/minigpt3d
source scripts/activate_env.sh ~/.virtualenvs/pointqa_eval/minigpt3d

uv pip install -r requirements_minigpt3d.txt
```

#### Update the model config
1. Check this issue: [issue](https://github.com/TangYuan96/MiniGPT-3D/issues/6), move [MiniGPT-3D/modeling_phi.py](https://github.com/TangYuan96/MiniGPT-3D/blob/main/modeling_phi.py) to `model/phi/modeling_phi.py` in the transformers library.
2. Modify the model path in benchmark config file: [benchmark_evaluation_paper.yaml](models/dependence/minigpt3d/eval_configs/benchmark_evaluation_paper.yaml)

#### Modify the model path

The files to be modified:
- [conversation.py](models/dependence/minigpt3d/minigpt4/conversation/conversation.py) line 20
- [base_model.py](models/dependence/minigpt3d/minigpt4/models/base_model.py) line 55

In both files, change the tokenizer loading line to:
```python
tokenizer = AutoTokenizer.from_pretrained("model/MiniGPT-3D/params_weight/Phi_2")
```

Change the tokenizer path to your real path.

### 5. OneLLM
#### Virtual environment
```bash
cd PointQA_Eval

bash ./scripts/setup_env.sh ~/.virtualenvs/pointqa_eval/onellm
source scripts/activate_env.sh ~/.virtualenvs/pointqa_eval/onellm

uv pip install -r requirements_onellm.txt

# install pointnet2
git clone https://github.com/csuhan/OneLLM.git
cd OneLLM/model/lib/pointnet2
python setup.py install
```

#### Download checkpoints

```bash
hf download timm/vit_large_patch14_clip_224.openai --local-dir /model/vit_large_patch14_clip_224

hf download csuhan/OneLLM-7B --local-dir /model/OneLLM-7B 
```

### 6. PointAlign
#### Virtual environment
Use uv
```bash
cd PointQA_Eval

bash ./scripts/setup_env.sh ~/.virtualenvs/pointqa_eval/minigpt3d
source scripts/activate_env.sh ~/.virtualenvs/pointqa_eval/minigpt3d

uv pip install -r requirements_minigpt3d.txt
```

or use conda
```bash
conda env create -f environment.yml
```

#### Download checkpoints
```bash
hf download ShijianW01/PointAlign_weight --local-dir /path

wget -P "/path" "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
```

#### Update the model config
PointAlign uses the same framework as MiniGPT3D, so you need to modify: 
- the model path in [benchmark_evaluation_paper.yaml](models/dependence/minigpt3d/eval_configs/benchmark_evaluation_paper.yaml)
- move the [modeling_phi.py](models/dependence/minigpt3d/minigpt4/models/modeling_phi.py) to `model/phi /modeling_phi.py` in the transformers library.

## Quick Start

For all the models, you must modify the model path to your own path in the bash script firstly.
```bash
bash run_eval.sh
```