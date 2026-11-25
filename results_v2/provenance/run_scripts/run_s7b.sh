#!/bin/bash
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp CUDA_VISIBLE_DEVICES=0
until grep -q "S6C DONE" logs/s7.log 2>/dev/null; do sleep 30; done
[ -e STOP ] && { echo STOP; exit 0; }
echo "[$(date -Is)] batch R=24,32 | load $(cut -d" " -f1-3 /proc/loadavg) | gpu0 apps: $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | tr "\n" " ")"
.venv/bin/python -m analysis_v2.s7_bench batch --ndocs 35 --tokens 16384 --batches 24 32 --budget 96 --steps 128 --repeats 2 --out out/results_v2/gpu/batch_06b_r32 || echo FAILED
.venv/bin/python -m analysis_v2.s7_bench batch --ndocs 35 --tokens 16384 --batches 1 16 --budget 96 --steps 64 --repeats 1 --timing --out out/results_v2/gpu/batch_06b_timed || echo FAILED
echo "[$(date -Is)] S7B DONE"
