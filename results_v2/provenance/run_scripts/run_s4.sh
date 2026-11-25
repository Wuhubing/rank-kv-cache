#!/bin/bash
set -e
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp MPLBACKEND=Agg
PY=.venv/bin/python; T=data/dev; O=out/results_v2/dev
mkdir -p $O
echo "[$(date -Is)] check_traces"
$PY -c "import json,glob,heldout;print(len(heldout.check_traces(sorted(glob.glob(\"$T/*.npz\")),json.load(open(\"splits/longctx_v1.json\")),\"dev\",28)),\"dev traces OK\")"
echo "[$(date -Is)] efficient_dev"; $PY efficient_dev.py $T --out $O/efficient_dev --procs 32
echo "[$(date -Is)] checks"; $PY checks.py $T --res $O/efficient_dev --procs 32
echo "[$(date -Is)] audit"; $PY -m analysis_v2.audit_v2 $T --misses $O/efficient_dev/misses.csv --out $O/audit.txt --procs 14
echo "[$(date -Is)] rank_control"; $PY validation/rank_control_dev.py $T --out $O/rank_control --workers 32 --archive $O/efficient_dev
echo "[$(date -Is)] alloc_oracle"; $PY validation/alloc_oracle.py dev $T --out $O/alloc_oracle --misses $O/efficient_dev/misses.csv --alloc $O/efficient_dev/alloc.csv --workers 14
echo "[$(date -Is)] s4 replay"; $PY -m analysis_v2.s4_dev replay $T --out $O/s4 --procs 32
echo "[$(date -Is)] s4 summarize"; $PY -m analysis_v2.s4_dev summarize --out $O/s4
echo "[$(date -Is)] audit s4"; $PY -m analysis_v2.audit_v2 $T --misses $O/s4/misses.csv --out $O/s4/audit.txt --budgets 64 96 128 256 512 --procs 14
echo "[$(date -Is)] predict"; $PY -m analysis_v2.s4_dev predict $T --out $O/prediction --procs 14
echo "[$(date -Is)] noise"; $PY -m analysis_v2.s4_dev noise $T --out $O/noise --procs 32
echo "[$(date -Is)] S4DONE"
