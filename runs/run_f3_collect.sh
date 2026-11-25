#!/bin/bash
# followup-v3 F3: collect the third held-out set ONCE (after docs/preregistration_v3.md and splits/heldout_frozen_v3.json
# are committed and pushed). Writes the UTC start time, the commit and the GPU UUID first.
set -e
cd /data/weibing/projects/rkv-v3
export TMPDIR=/data/weibing/tmp HF_HOME=/data/weibing/cache/huggingface
GPU=$(free-gpus | tr ',' '\n' | grep -v '^2$' | head -1)
[ -z "$GPU" ] && GPU=$(free-gpus 1)
[ -z "$GPU" ] && { echo "no free GPU"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU
echo "F3 collection start (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "commit: $(git rev-parse HEAD)  (origin: $(git ls-remote origin refs/heads/mechanism-v3 | cut -f1))"
echo "GPU $GPU $(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)"
.venv/bin/python -m collect.long --split splits/longctx_v3.json --subset heldout --out data/heldout_v3_06b
echo "0.6B done (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
.venv/bin/python -m collect.long --split splits/longctx_v3_8b.json --subset heldout --out data/heldout_v3_8b
echo "F3 collection end (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
