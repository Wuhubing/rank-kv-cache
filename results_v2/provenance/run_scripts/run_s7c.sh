#!/bin/bash
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp CUDA_VISIBLE_DEVICES=0
until grep -q "S7B DONE" logs/s7b.log 2>/dev/null; do sleep 30; done
[ -e STOP ] && { echo STOP; exit 0; }
echo "[$(date -Is)] 8B batch at 12288 tokens | load $(cut -d" " -f1-3 /proc/loadavg) | gpu0 apps: $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | tr "\n" " ")"
.venv/bin/python -m analysis_v2.s7_bench batch --split splits/longctx_8b_v1.json --ndocs 35 --tokens 12288 --batches 1 16 32 --budget 96 --steps 64 --repeats 2 --frozen splits/heldout_frozen_8b.json --out out/results_v2/gpu/batch_8b_12k || echo FAILED
echo "[$(date -Is)] S7C DONE"
