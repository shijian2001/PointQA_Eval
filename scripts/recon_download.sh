#!/usr/bin/env bash
set -euo pipefail

REPO="qizekun/ReConV2"
REMOTE="zeroshot/large/best_lvis.pth"
OUT="checkpoints/recon/large.pth"
mkdir -p "$(dirname "$OUT")"
URL="https://huggingface.co/${REPO}/resolve/main/${REMOTE}"

# 若仓库需要授权，先导出 HF_TOKEN
# export HF_TOKEN="hf_xxx"
if [ -n "${HF_TOKEN:-}" ]; then
  wget --tries=6 --waitretry=2 --header="Authorization: Bearer ${HF_TOKEN}" "$URL" -O "$OUT"
else
  wget --tries=6 --waitretry=2 "$URL" -O "$OUT"
fi

echo "Downloaded -> $OUT"