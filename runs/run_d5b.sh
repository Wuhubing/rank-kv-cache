#!/bin/bash
# followup-v3 D5b: protection zone on an H2O-style accumulated score, 0.6B and 8B dev (after the learn queue).
cd /data/weibing/projects/rkv-v3
export TMPDIR=/data/weibing/tmp OMP_NUM_THREADS=1
P="nice -n 10 .venv/bin/python"
while tmux has-session -t wb-rkv-fu-learn 2>/dev/null; do sleep 60; done
for m in dev:data/dev 8b:data/8b_dev; do
  [ -e out/v3/PAUSE ] && while [ -e out/v3/PAUSE ]; do sleep 60; done
  O=out/v3/d5b_${m%%:*}
  $P -m analysis_v3.run replay d5b ${m#*:} --out $O --procs 14 > $O.log 2>&1 || echo FAILED
  $P -m analysis_v3.run summarize d5b --out $O > $O/summary.txt 2>&1
  $P -m analysis_v2.audit_v2 ${m#*:} --misses $O/misses.csv --out $O/audit.txt --budgets 96 128 --procs 14 > /dev/null 2>&1 || echo AUDIT FAILED
done
echo "[$(date -u -Is)] D5B DONE"
