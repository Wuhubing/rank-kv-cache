#!/bin/bash
# followup-v3: D1, G1 (post hoc), D5/D6 and instrument summaries, after the G2 queue. <= 16 processes, nice 10.
cd /data/weibing/projects/rkv-v3
export TMPDIR=/data/weibing/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
P="nice -n 10 .venv/bin/python"
STOPF=/data/weibing/projects/rank-kv-cache/STOP
while tmux has-session -t wb-rkv-fu-g2-cpu 2>/dev/null; do sleep 60; done
gate() { [ -e $STOPF ] && { echo "STOP file"; exit 0; }; while [ -e out/v3/PAUSE ]; do sleep 60; done; }
gate; echo "[$(date -u -Is)] D1"
$P -m analysis_v3.d1_iso --lru-fine out/v3/lru_fine_dev/misses.csv --layers 28 --policy ema=out/v3/b1_dev/misses.csv:ema_z:LODO \
   --policy c1=out/v3/b1_dev/misses.csv:pow_norm_kth:1.0 --c4 results_v3/c4/dev/rows.csv.gz --out out/v3/d1_dev || echo FAILED d1 dev
$P -m analysis_v3.d1_iso --lru-fine out/v3/lru_fine_8b/misses.csv --layers 36 --policy ema=out/v3/b1_8b/misses.csv:ema_z:LODO \
   --policy c1=out/v3/b1_8b/misses.csv:pow_norm_kth:1.0 --c4 results_v3/c4/8b/rows.csv.gz --out out/v3/d1_8b || echo FAILED d1 8b
echo "[$(date -u -Is)] G1 (post hoc replays on the evaluated held-out sets)"
for s in heldout_v1:../rank-kv-cache/data/heldout heldout_8b:../rank-kv-cache/data/8b_heldout heldout_v3_06b:data/heldout_v3_06b heldout_v3_8b:data/heldout_v3_8b; do
  gate
  n=${s%%:*}; t=${s#*:}
  $P -m analysis_v3.run replay lru_fine $t --out out/v3/g1_lru_fine_$n --procs 16 > out/v3/g1_lru_fine_$n.log 2>&1 || echo "FAILED lru_fine $n"
  $P -m analysis_v3.run replay g1 $t --out out/v3/g1_$n --procs 16 > out/v3/g1_$n.log 2>&1 || echo "FAILED g1 $n"
  $P -m analysis_v2.audit_v2 $t --misses out/v3/g1_$n/misses.csv --out out/v3/g1_$n/audit.txt --budgets 96 128 --procs 16 > /dev/null 2>&1 || echo "AUDIT FAILED $n"
done
echo "[$(date -u -Is)] summaries"
for m in dev 8b 8b_long; do $P -m analysis_v3.inst_summary out/v3/inst_$m > /dev/null || echo "FAILED inst_summary $m"; done
$P -m analysis_v3.d5_summary --misses out/v3/d5_dev/misses.csv --stats out/v3/b2/dev.csv --out out/v3/d5_sum_dev || echo FAILED d5 dev
$P -m analysis_v3.d5_summary --misses out/v3/d5_8b/misses.csv --stats out/v3/b2/8b.csv --out out/v3/d5_sum_8b || echo FAILED d5 8b
$P -m analysis_v3.d6_alpha --synth out/v3/b2_synth --stats out/v3/b2/dev.csv --misses out/v3/b1_dev/misses.csv --out out/v3/d6_dev || echo FAILED d6 dev
$P -m analysis_v3.d6_alpha --synth out/v3/b2_synth --stats out/v3/b2/8b.csv --misses out/v3/b1_8b/misses.csv --out out/v3/d6_8b || echo FAILED d6 8b
echo "[$(date -u -Is)] FINAL CPU DONE"
