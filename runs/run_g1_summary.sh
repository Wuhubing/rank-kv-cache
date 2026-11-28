#!/bin/bash
# followup-v3 G1 (post hoc, exploratory): iso-miss capacity of fixed configurations on the three evaluated held-out sets.
# EMA and allocation + EMA come from the preregistered evaluation tables (frozen alpha / allocation); C1 from the post-hoc g1 replay.
cd /data/weibing/projects/rkv-v3
P="nice -n 10 .venv/bin/python"
R=/data/weibing/projects/rank-kv-cache/out/results_v2
F=out/v3/f3_results
$P -m analysis_v3.d1_iso --lru-fine out/v3/g1_lru_fine_heldout_v1/misses.csv --layers 28 --budgets 96 128 \
   --policy ema=$R/heldout/eval/misses.csv:ema --policy alloc+ema=$R/heldout/eval/misses.csv:alloc_ema \
   --policy c1=out/v3/g1_heldout_v1/misses.csv:pow_norm_kth:1.0 --out out/v3/g1_iso_heldout_v1
$P -m analysis_v3.d1_iso --lru-fine out/v3/g1_lru_fine_heldout_8b/misses.csv --layers 36 --budgets 96 128 \
   --policy ema=$R/ext/8b_heldout/eval/misses.csv:ema --policy alloc+ema=$R/ext/8b_heldout/eval/misses.csv:alloc_ema \
   --policy c1=out/v3/g1_heldout_8b/misses.csv:pow_norm_kth:1.0 --out out/v3/g1_iso_heldout_8b
for m in 06b:28 8b:36; do
  n=${m%%:*}
  $P -m analysis_v3.d1_iso --lru-fine out/v3/g1_lru_fine_heldout_v3_$n/misses.csv --layers ${m#*:} --budgets 96 128 \
     --policy ema=$F/$n/misses.csv:ema --policy alloc+ema=$F/$n/misses.csv:alloc_ema --policy c1=$F/$n/misses.csv:p2 \
     --policy pool_percentile=$F/$n/misses.csv:p4 --out out/v3/g1_iso_heldout_v3_$n
done
