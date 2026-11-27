#!/bin/bash
set -e
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp HF_HOME=/data/weibing/cache/huggingface
until grep -q A3COLLECTDONE logs/ark_a3_collect.log; do sleep 30; done
export CUDA_VISIBLE_DEVICES=2
others=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep GPU-28e78787 || true)
[ -z "$others" ] || { echo "card 2 busy: $others"; exit 1; }
echo "[$(date -Is)] card 2 free; 8B dev with digest read-outs"
.venv/bin/python -m collect.long --split splits/longctx_8b_v1.json --subset dev --out data/8b_dev_digest --record-arkvale-digest
echo "[$(date -Is)] A3_8B_DONE"
