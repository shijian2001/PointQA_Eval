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
```

#### Install packages
```bash
uv pip install -r requirements_greenplm.txt
```

#### Install Pointnet2_PyTorch

First check this [issue](https://github.com/erikwijmans/Pointnet2_PyTorch/issues/174)

```bash
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib
uv pip install -e . --no-build-isolation
```


## Quick Start

```bash
# run eval
bash run_eval.sh
```