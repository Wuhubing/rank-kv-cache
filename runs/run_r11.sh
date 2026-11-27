#!/bin/bash
# rkv-r11 (PR #11 review): R11-6 (clean re-run of 8B batch 1 and 4) and R11-7 (policy-kernel timing and memory by phase,
# 8B batch 16 / 12K batch 32 and 0.6B batch 32). Same preconditions as run_e.sh: one card, never card 2, no other process
# on the card, 1-minute load < 4; checked every 30 min, an item is skipped after 6 h of accumulated waiting.
# Every run additionally waits for a clean card (--wait-clean) and repeats an unclean run (kept as discarded_runs).
cd /data/weibing/projects/rkv-v3
export TMPDIR=/data/weibing/tmp HF_HOME=/data/weibing/cache/huggingface OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=".venv/bin/python"
STOPF=/data/weibing/projects/rank-kv-cache/STOP
pick() {
  for g in $(free-gpus | tr ',' ' '); do
    [ "$g" = 2 ] && continue
    [ -z "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader)" ] && { echo $g; return; }
  done
}
ready() {  # sets GPU; per-item waiting budget 6 h (running time does not count)
  WAITED=0
  while true; do
    [ -e $STOPF ] && { echo "STOP file"; exit 0; }
    GPU=$(pick); L=$(cut -d" " -f1 /proc/loadavg)
    echo "[$(date -u -Is)] check: gpu='$GPU' load=$L waited=${WAITED}s"
    if [ -n "$GPU" ] && awk "BEGIN{exit !($L < 4)}"; then return 0; fi
    [ $WAITED -ge 21600 ] && return 1
    sleep 1800; WAITED=$((WAITED + 1800))
  done
}
GPUTEST=0
run() {  # name, args...
  name=$1; shift
  if ls results_v3/e/$name/raw/batch*.json.gz > /dev/null 2>&1; then echo "[$(date -u -Is)] $name already in results, skipped"; return; fi
  if ready; then
    echo "[$(date -u -Is)] START $name on GPU $GPU $(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader) load $(cut -d' ' -f1 /proc/loadavg)"
    if [ $GPUTEST = 0 ]; then
      CUDA_VISIBLE_DEVICES=$GPU $P -m pytest -q -p no:cacheprovider tests/test_gpu_bench_batch2.py > out/v3/r11/gputest.log 2>&1 \
        || { echo "GPU TESTS FAILED (out/v3/r11/gputest.log)"; exit 1; }
      GPUTEST=1; echo "[$(date -u -Is)] GPU tests passed"
    fi
    mkdir -p out/v3/r11/$name
    CUDA_VISIBLE_DEVICES=$GPU $P -m analysis_v3.e_bench --out out/v3/r11/$name --wait-clean "$@" > out/v3/r11/$name.log 2>&1 \
      || { echo "FAILED $name"; return; }
    mkdir -p results_v3/e/$name/raw
    for f in out/v3/r11/$name/batch*.json; do gzip -c "$f" > results_v3/e/$name/raw/$(basename $f).gz; done
    cp out/v3/r11/$name/meta.json results_v3/e/$name/raw/ && cp out/v3/r11/$name.log results_v3/e/$name/
    echo "[$(date -u -Is)] DONE $name"
  else
    echo "SKIPPED $name: no free card with load < 4 within 6 h"
  fi
}
mkdir -p out/v3/r11
S8="--split splits/longctx_8b_v1.json --frozen splits/heldout_frozen_8b.json"
TP="--rounds 5 --steps 72 --policies lru score ema alloc_ema belady --timing"
run batch_8b_rerun $S8 --batches 1 4 --rounds 5
run timing_8b/b16_16k $S8 --batches 16 $TP
run timing_8b/b32_12k $S8 --tokens 12288 --batches 32 $TP
run timing_06b_b32 --batches 32 $TP
echo "[$(date -u -Is)] R11 GPU DONE"
