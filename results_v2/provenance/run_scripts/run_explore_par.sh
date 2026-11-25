#!/bin/bash
# Exploratory replays for the page/K sweep and the sampling seeds, three settings at a time (14 processes each).
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp
one() {
  d=$1; kind=$2; name=$(basename "$d")
  echo "[$(date -Is)] explore $d"
  .venv/bin/python -m analysis_v2.explore "$d" --out "out/results_v2/ext/$kind/$name" --procs 14 \
      > "logs/explore_${kind}_${name}.log" 2>&1 || echo "FAILED $d"
}
export -f one
{ for d in data/sweep/ps*_k*; do echo "$d sweep"; done; for d in data/sample/seed*; do echo "$d sampling"; done; } \
  | xargs -P 3 -L 1 bash -c 'one "$0" "$1"'
echo "[$(date -Is)] S6C DONE" >> logs/s7.log
echo "[$(date -Is)] all explores done"
