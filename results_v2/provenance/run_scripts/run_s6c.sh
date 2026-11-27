#!/bin/bash
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp
while tmux has-session -t wb-rkv-collect-scbench 2>/dev/null; do sleep 30; done
echo "[$(date -Is)] scbench explore"
.venv/bin/python -m analysis_v2.explore data/scbench_dev --out out/results_v2/ext/scbench_dev --procs 14
while tmux has-session -t wb-rkv-s6-sweep 2>/dev/null; do sleep 30; done
for d in data/sweep/ps*_k*; do
  echo "[$(date -Is)] sweep explore $d"
  .venv/bin/python -m analysis_v2.explore $d --out out/results_v2/ext/sweep/$(basename $d) --procs 32
done
for d in data/sample/seed*; do
  echo "[$(date -Is)] sampling explore $d"
  .venv/bin/python -m analysis_v2.explore $d --out out/results_v2/ext/sampling/$(basename $d) --procs 14
done
echo "[$(date -Is)] S6C DONE"
