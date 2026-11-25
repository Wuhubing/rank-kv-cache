#!/bin/bash
# followup-v3: instrumented replays, learned models, B2 regression / synthetic model, LRU fine grids (dev only).
cd /data/weibing/projects/rkv-v3
export TMPDIR=/data/weibing/tmp
P="nice -n 10 .venv/bin/python"
STOPF=/data/weibing/projects/rank-kv-cache/STOP
while tmux has-session -t wb-rkv-fu-cd-cpu 2>/dev/null; do sleep 60; done
gate() { [ -e $STOPF ] && { echo "STOP file"; exit 0; }; while [ -e out/v3/PAUSE ]; do sleep 60; done; }
for m in dev:data/dev 8b:data/8b_dev; do
  gate; echo "[$(date -u -Is)] instrument ${m%%:*}"
  $P -m analysis_v3.instrument ${m#*:} --out out/v3/inst_${m%%:*} --B 96 --procs 14 || echo "FAILED instrument $m"
done
for m in dev:data/dev 8b:data/8b_dev; do
  gate; echo "[$(date -u -Is)] lru_fine ${m%%:*}"
  O=out/v3/lru_fine_${m%%:*}
  $P -m analysis_v3.run replay lru_fine ${m#*:} --out $O --procs 16 > $O.log 2>&1 || echo "FAILED lru_fine $m"
done
gate; echo "[$(date -u -Is)] b2 model"
$P -m analysis_v3.b2_model regress --stats out/v3/b2 --out out/v3/b2_regression > out/v3/b2_regression.log 2>&1 || echo "FAILED regress"
$P -m analysis_v3.b2_model fit --stats out/v3/b2/dev.csv --out out/v3/b2_synth --procs 16 > out/v3/b2_synth_fit.log 2>&1 || echo "FAILED fit"
$P -m analysis_v3.b2_model curves --out out/v3/b2_synth --procs 16 > out/v3/b2_synth_curves.log 2>&1 || echo "FAILED curves"
for kind in linear gbdt; do
  for m in dev:data/dev 8b:data/8b_dev; do
    gate; echo "[$(date -u -Is)] learn $kind ${m%%:*}"
    $P -m analysis_v3.learn ${m#*:} --out out/v3/learn_${kind}_${m%%:*} --model $kind --procs 14 > out/v3/learn_${kind}_${m%%:*}.log 2>&1 \
      || echo "FAILED learn $kind $m"
  done
done
echo "[$(date -u -Is)] MORE CPU DONE"
