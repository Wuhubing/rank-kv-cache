#!/bin/bash
set -e
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp
mkdir -p results_v2/heldout
echo "[$(date -Is)] step2"; .venv/bin/python heldout.py eval data/heldout --frozen splits/heldout_frozen_v1.json --out results_v2/heldout/eval --procs 21
echo "[$(date -Is)] step3"; .venv/bin/python recompute_heldout.py results_v2/heldout/eval/misses.csv > results_v2/heldout/criteria.txt
echo "[$(date -Is)] step4"; .venv/bin/python validation/alloc_oracle.py heldout data/heldout --out results_v2/heldout/alloc_oracle --misses results_v2/heldout/eval/misses.csv --frozen splits/heldout_frozen_v1.json --workers 14
echo "[$(date -Is)] step5"; .venv/bin/python -m analysis_v2.audit_v2 data/heldout --misses results_v2/heldout/eval/misses.csv --out results_v2/heldout/audit.txt --budgets 72 80 96 128 256
echo "[$(date -Is)] S5DONE"
