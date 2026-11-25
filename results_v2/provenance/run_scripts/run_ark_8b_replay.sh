#!/bin/bash
set -e
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=""
PY=.venv/bin/python; O=out/results_v2/dev/arkvale/8b
echo "[$(date -Is)] replay 8B"; $PY -m analysis_v2.arkvale_full replay data/8b_dev_digest --out $O --procs 26 --streams plain page token --digests quest cuboid_mean
echo "[$(date -Is)] summarize"; $PY -m analysis_v2.arkvale_full summarize --out $O
echo "[$(date -Is)] tables"; $PY -m analysis_v2.arkvale_full tables --out $O
echo "[$(date -Is)] audit"; $PY -m analysis_v2.arkvale_full audit data/8b_dev_digest --out $O --procs 14
echo "[$(date -Is)] REPLAYDONE 8b"
