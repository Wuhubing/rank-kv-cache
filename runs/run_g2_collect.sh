#!/bin/bash
# followup-v3 G2 (exploratory): Qwen3-8B dev, 14 documents, decode cap raised to 4096 tokens -> data/8b_dev_long.
set -e
cd /data/weibing/projects/rkv-v3
export TMPDIR=/data/weibing/tmp HF_HOME=/data/weibing/cache/huggingface
GPU=$(free-gpus 1)
[ -z "$GPU" ] && { echo "no free GPU"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU
echo "[$(date -u -Is)] G2 collect on GPU $GPU $(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)"
.venv/bin/python -m collect.long --split splits/longctx_8b_v1.json --subset dev --out data/8b_dev_long --max-new 4096
echo "[$(date -u -Is)] G2 DONE"
