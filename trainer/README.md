# PointQA Trainer

## 1. PointLLM

### 1.1 Install Packages

```bash
uv --project envs/pointllm sync
```

### 1.2 Download checkpoints

```bash
# point backbone checkpoints
hf download RunsenXu/PointLLM_7B_v1.1_init --local-dir /path/PointLLM_7B_v1.1_init
```
### 1.3 Run Trainer

```bash
cd PointQA_Eval/trainer/PointLLM
bash run_train_what_distance_farthest.sh
```