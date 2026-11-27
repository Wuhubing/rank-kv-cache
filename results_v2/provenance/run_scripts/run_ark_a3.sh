#!/bin/bash
set -e
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp HF_HOME=/data/weibing/cache/huggingface
export CUDA_VISIBLE_DEVICES=$(free-gpus 1)
[ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "no free GPU"; exit 1; }
echo "[$(date -Is)] card $CUDA_VISIBLE_DEVICES"; nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=index,uuid,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
PY=.venv/bin/python
echo "[$(date -Is)] dev with digest read-outs"
$PY -m collect.long --split splits/longctx_v1.json --subset dev --out data/dev_digest --record-arkvale-digest
echo "[$(date -Is)] dev with the ArkVale selector"
$PY -m collect.long --split splits/longctx_v1.json --subset dev --out data/dev_arkvale_sel --selector arkvale-mean --arkvale-sink 1 --arkvale-recent 1
echo "[$(date -Is)] A3COLLECTDONE"
