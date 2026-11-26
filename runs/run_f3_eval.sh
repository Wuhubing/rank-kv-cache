#!/bin/bash
# followup-v3 F3 evaluation: exactly the commands of docs/preregistration_v3.md, run once. The C/D queue is paused at a job
# boundary (out/v3/PAUSE) so that the task stays within 32 processes.
cd /data/weibing/projects/rkv-v3
export TMPDIR=/data/weibing/tmp
touch out/v3/PAUSE
trap 'rm -f out/v3/PAUSE' EXIT
while pgrep -u weibing -f "analysis_v3.(run|noise|instrument|b2_model|b2_stats|c4_alloc) " > /dev/null; do sleep 30; done
echo "F3 evaluation start (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)  commit $(git rev-parse HEAD)"
set -x
python3 -c "print('using .venv')" >/dev/null
PY=.venv/bin/python
$PY -m analysis_v3.f_eval replay data/heldout_v3_06b --frozen splits/heldout_frozen_v3.json --model 06b \
    --split splits/longctx_v3.json --out results_v3/f3/06b --procs 16
$PY -m analysis_v3.f_eval criteria --frozen splits/heldout_frozen_v3.json --model 06b --out results_v3/f3/06b
$PY -m analysis_v3.f_eval replay data/heldout_v3_8b --frozen splits/heldout_frozen_v3.json --model 8b \
    --split splits/longctx_v3_8b.json --out results_v3/f3/8b --procs 16
$PY -m analysis_v3.f_eval criteria --frozen splits/heldout_frozen_v3.json --model 8b --out results_v3/f3/8b
$PY -m analysis_v2.audit_v2 data/heldout_v3_06b --misses results_v3/f3/06b/misses.csv --out results_v3/f3/06b/audit.txt \
    --budgets 96 128
$PY -m analysis_v2.audit_v2 data/heldout_v3_8b --misses results_v3/f3/8b/misses.csv --out results_v3/f3/8b/audit.txt \
    --budgets 96 128
set +x
echo "F3 evaluation end (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
