#!/bin/bash
set -e
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=""
PY=.venv/bin/python; O=out/results_v2/dev/arkvale
echo "[$(date -Is)] extra replay"; $PY -m analysis_v2.arkvale_full replay data/dev_digest --out $O/extra_tmp --procs ${PROCS:-6} --streams plain page --digests quest cuboid_mean --budgets 192 384 512 --no-eq
mv $O/extra_tmp/misses.csv $O/main/misses_extra.csv && rmdir $O/extra_tmp
echo "[$(date -Is)] summarize"; $PY -m analysis_v2.arkvale_full summarize --out $O/main
$PY -m analysis_v2.arkvale_full tables --out $O/main
echo "[$(date -Is)] audit"; $PY -m analysis_v2.arkvale_full audit data/dev_digest --out $O/main --procs 8
echo "[$(date -Is)] REPLAYDONE extra"
