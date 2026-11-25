#!/bin/bash
# usage: run_ark_replay.sh main|e2e [traces_dir] [out_name]
set -e
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=""
PY=.venv/bin/python; O=out/results_v2/dev/arkvale; mkdir -p $O
if [ "$1" = main ]; then
  T=${2:-data/dev_digest}; N=${3:-main}
  echo "[$(date -Is)] compare"; $PY -m analysis_v2.arkvale_full compare data/dev $T --out $O/compare_readout_$N.txt
  echo "[$(date -Is)] replay"; $PY -m analysis_v2.arkvale_full replay $T --out $O/$N --procs 32
  echo "[$(date -Is)] summarize"; $PY -m analysis_v2.arkvale_full summarize --out $O/$N
  echo "[$(date -Is)] audit"; $PY -m analysis_v2.arkvale_full audit $T --out $O/$N --procs 14
else
  T=${2:-data/dev_arkvale_sel}; N=${3:-e2e}
  echo "[$(date -Is)] compare e2e"; $PY -m analysis_v2.arkvale_full compare data/dev $T --e2e --out $O/compare_$N.txt
  echo "[$(date -Is)] replay e2e"; $PY -m analysis_v2.arkvale_full replay $T --e2e --out $O/$N --procs ${PROCS:-32}
  echo "[$(date -Is)] summarize e2e"; $PY -m analysis_v2.arkvale_full summarize --e2e --out $O/$N
  echo "[$(date -Is)] audit e2e"; $PY -m analysis_v2.arkvale_full audit $T --e2e --out $O/$N --procs ${PROCS:-14}
fi
echo "[$(date -Is)] REPLAYDONE $1"
