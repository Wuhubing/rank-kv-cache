#!/bin/bash
# followup-v3 C2 (linear) and B3 (boosted trees) learned models on wulab2, in parallel with the other queues
# (task section 12 as revised 2026-11-25: <= 32 processes for the task, nice 10, no new process while load > 44).
cd /data/weibing/projects/rkv-v3
export TMPDIR=/data/weibing/tmp
P="nice -n 10 .venv/bin/python"
STOPF=/data/weibing/projects/rank-kv-cache/STOP
gate() { [ -e $STOPF ] && { echo "STOP file"; exit 0; }; while [ -e out/v3/PAUSE ]; do sleep 60; done
         while awk '{exit !($1 > 44)}' /proc/loadavg; do sleep 60; done; }
for kind in linear gbdt; do
  for m in dev:data/dev 8b:data/8b_dev; do
    gate; echo "[$(date -u -Is)] learn $kind ${m%%:*}"
    $P -m analysis_v3.learn ${m#*:} --out out/v3/learn_${kind}_${m%%:*} --model $kind --procs 14 > out/v3/learn_${kind}_${m%%:*}.log 2>&1 \
      || echo "FAILED learn $kind $m"
  done
done
echo "[$(date -u -Is)] LEARN DONE"
