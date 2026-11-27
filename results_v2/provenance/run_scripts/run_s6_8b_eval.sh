#!/bin/bash
set -e
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp
mkdir -p results_v2/ext/8b_heldout
echo "[$(date -Is)] eval"; .venv/bin/python heldout.py eval data/8b_heldout --frozen splits/heldout_frozen_8b.json --split splits/longctx_8b_v1.json --out results_v2/ext/8b_heldout/eval --procs 21
echo "[$(date -Is)] criteria"; .venv/bin/python recompute_heldout.py results_v2/ext/8b_heldout/eval/misses.csv > results_v2/ext/8b_heldout/criteria.txt
echo "[$(date -Is)] alloc_oracle"; .venv/bin/python validation/alloc_oracle.py heldout data/8b_heldout --out results_v2/ext/8b_heldout/alloc_oracle --misses results_v2/ext/8b_heldout/eval/misses.csv --frozen splits/heldout_frozen_8b.json --workers 14
echo "[$(date -Is)] audit"; .venv/bin/python -m analysis_v2.audit_v2 data/8b_heldout --misses results_v2/ext/8b_heldout/eval/misses.csv --out results_v2/ext/8b_heldout/audit.txt --budgets 72 80 96 128 256
echo "[$(date -Is)] S6_8B_EVAL_DONE"
