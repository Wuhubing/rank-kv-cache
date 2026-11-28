#!/bin/bash
# followup-v3 part B, CPU replays (dev only). Runs after the part-A replay; one job at a time, <= 16 processes, nice 10.
cd /data/weibing/projects/rkv-v3
export TMPDIR=/data/weibing/tmp
P="nice -n 10 .venv/bin/python"
STOPF=/data/weibing/projects/rank-kv-cache/STOP
while tmux has-session -t wb-rkv-fu-a-replay 2>/dev/null; do sleep 60; done
while tmux has-session -t wb-rkv-fu-b4-collect 2>/dev/null; do sleep 60; done
run() {  # exp setting traces
  [ -e $STOPF ] && { echo "STOP file"; exit 0; }
  echo "[$(date -u -Is)] $1 $2"
  O=out/v3/$1_$2
  $P -m analysis_v3.run replay $1 $3 --out $O --procs 16 > $O.log 2>&1 || echo "FAILED replay $1 $2"
  $P -m analysis_v3.run summarize $1 --out $O > $O/summary.txt 2>&1 || echo "FAILED summarize $1 $2"
  $P -m analysis_v2.audit_v2 $3 --misses $O/misses.csv --out $O/audit.txt --budgets 80 96 128 256 --procs 16 > /dev/null 2>&1 \
    || echo "AUDIT FAILED $1 $2"
  head -3 $O/audit.txt
}
mkdir -p out/v3
run b4 dev data/dev_attn_all
run b4_quest dev data/dev_attn_all
run b1 dev data/dev
run b7 dev data/dev
run b3 dev data/dev
run b1 8b data/8b_dev
run b7 8b data/8b_dev
run b3 8b data/8b_dev
echo "[$(date -u -Is)] b2 stats"
mkdir -p out/v3/b2
for s in dev:data/dev 8b:data/8b_dev 64k:data/64k_dev ruler:data/ruler_dev scbench:data/scbench_dev \
         seed0:data/sample/seed0 seed1:data/sample/seed1 seed2:data/sample/seed2; do
  [ -e $STOPF ] && { echo "STOP file"; exit 0; }
  $P -m analysis_v3.b2_stats ${s%%:*} ${s#*:} --out out/v3/b2/${s%%:*}.csv --procs 16 || echo "FAILED b2 $s"
done
for d in data/sweep/*; do
  [ -e $STOPF ] && { echo "STOP file"; exit 0; }
  n=$(basename $d)
  $P -m analysis_v3.b2_stats $n $d --out out/v3/b2/$n.csv --procs 16 || echo "FAILED b2 $n"
done
echo "[$(date -u -Is)] B CPU DONE"
