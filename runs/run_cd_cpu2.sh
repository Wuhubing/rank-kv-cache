#!/bin/bash
# followup-v3 parts C/D (without C4), CPU replays (dev only), after the part-B queue. <= 16 processes, nice 10.
# A file out/v3/PAUSE makes the queue wait between jobs (used to keep the host quiet for the timing runs of part E).
cd /data/weibing/projects/rkv-v3
export TMPDIR=/data/weibing/tmp
P="nice -n 10 .venv/bin/python"
STOPF=/data/weibing/projects/rank-kv-cache/STOP
while tmux has-session -t wb-rkv-fu-b-cpu 2>/dev/null; do sleep 60; done
gate() { [ -e $STOPF ] && { echo "STOP file"; exit 0; }; while [ -e out/v3/PAUSE ]; do sleep 60; done; }
run() {  # exp setting traces
  gate
  echo "[$(date -u -Is)] $1 $2"
  O=out/v3/$1_$2
  $P -m analysis_v3.run replay $1 $3 --out $O --procs 16 > $O.log 2>&1 || echo "FAILED replay $1 $2"
  $P -m analysis_v3.run summarize $1 --out $O > $O/summary.txt 2>&1 || echo "FAILED summarize $1 $2"
  $P -m analysis_v2.audit_v2 $3 --misses $O/misses.csv --out $O/audit.txt --budgets 80 96 128 256 --procs 16 > /dev/null 2>&1 \
    || echo "AUDIT FAILED $1 $2"
  head -3 $O/audit.txt
}
gate; echo "[$(date -u -Is)] d3 noise"
$P -m analysis_v3.noise pool --attn data/dev_attn_all --out out/v3/d3 || echo "FAILED d3 pool"
$P -m analysis_v3.noise replay data/dev --out out/v3/d3 --procs 16 > out/v3/d3.log 2>&1 || echo "FAILED d3 replay"
$P -m analysis_v3.noise summarize --out out/v3/d3 > out/v3/d3/summary.txt 2>&1 || echo "FAILED d3 summarize"
# C4 runs on the mini (analysis_v3.c4_alloc on a copy of the dev traces) to shorten the critical path
run c3 dev data/dev
run d5 dev data/dev
run c3 8b data/8b_dev
run d5 8b data/8b_dev
echo "[$(date -u -Is)] CD CPU DONE"
