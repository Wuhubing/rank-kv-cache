#!/bin/bash
# followup-v3 G2 replays on data/8b_dev_long (after the other CPU queues).
cd /data/weibing/projects/rkv-v3
export TMPDIR=/data/weibing/tmp
P="nice -n 10 .venv/bin/python"
STOPF=/data/weibing/projects/rank-kv-cache/STOP
while tmux has-session -t wb-rkv-fu-more-cpu 2>/dev/null; do sleep 60; done
gate() { [ -e $STOPF ] && { echo "STOP file"; exit 0; }; while [ -e out/v3/PAUSE ]; do sleep 60; done; }
gate; echo "[$(date -u -Is)] base 8b_long"
O=out/v3/base_8b_long
$P -m analysis_v3.run replay base data/8b_dev_long --out $O --procs 16 > $O.log 2>&1 || echo "FAILED replay"
$P -m analysis_v3.run summarize base --out $O > $O/summary.txt 2>&1 || echo "FAILED summarize"
$P -m analysis_v2.audit_v2 data/8b_dev_long --misses $O/misses.csv --out $O/audit.txt --budgets 80 96 128 256 --procs 16 > /dev/null 2>&1 || echo "AUDIT FAILED"
gate; echo "[$(date -u -Is)] instrument 8b_long"
$P -m analysis_v3.instrument data/8b_dev_long --out out/v3/inst_8b_long --B 96 --procs 14 || echo "FAILED instrument"
echo "[$(date -u -Is)] G2 CPU DONE"
