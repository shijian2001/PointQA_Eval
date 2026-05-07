# PointQA Trainer

## 1. PointLLM

### 1.1 Install Packages

```bash
uv --project envs/pointllm sync
source envs/pointllm/bin/activate
```

### 1.2 Download checkpoints

```bash
hf download RunsenXu/PointLLM_7B_v1.1_init --local-dir /path/PointLLM_7B_v1.1_init
```

### 1.3 Update the Model Configuration

Modify the path in [run_train_what_distance_farthest.sh](/PointQA_Eval/trainer/PointLLM/run_train_what_distance_farthest.sh)

### 1.4 Run Trainer

```bash
cd PointQA_Eval/trainer/PointLLM
bash run_train_what_distance_farthest.sh
```
## 2. PointAlign

### 2.1 Install Packages

```bash
uv --project envs/pointalign sync
source envs/pointalign/bin/activate
```

### 2.2 Update the Model Configuration

Modify the model path in [finetune_custom_dataset.yaml](/PointQA_Eval/trainer/PointAlign/finetune_custom_dataset.yaml)

### 2.3 Run Trainer

```bash
cd PointQA_Eval/trainer/PointAlign
bash run_train_what_distance_farthest.sh
```