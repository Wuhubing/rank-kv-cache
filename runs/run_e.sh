#!/bin/bash
# followup-v3 part E: timing runs. Preconditions (task section 9): the chosen card has no other process and the host's
# 1-minute load average is < 4; checked before every configuration (every 30 min for at most 6 h, else that part is
# skipped and logged). This task's CPU queues are paused with out/v3/PAUSE while E runs.
cd /data/weibing/projects/rkv-v3
export TMPDIR=/data/weibing/tmp HF_HOME=/data/weibing/cache/huggingface
P=".venv/bin/python"
STOPF=/data/weibing/projects/rank-kv-cache/STOP
touch out/v3/PAUSE
trap 'rm -f out/v3/PAUSE' EXIT
# wait until no replay of this task is running (the queues stop between jobs while PAUSE exists)
while pgrep -u weibing -f "analysis_v3.(run|noise|c4_alloc|instrument|learn|b2_model|b2_stats) " > /dev/null; do sleep 60; done
pick() {  # free card without other processes; card 2 is never used (three CUDA faults on it: ArkVale task, followup-v3
          # smoke test, and the first E2 attempt "illegal memory access" at 04:22Z); empty if none
  for g in $(free-gpus | tr ',' ' '); do
    [ "$g" = 2 ] && continue
    [ -z "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader)" ] && { echo $g; return; }
  done
}
WAITED=0
ready() {  # sets GPU; returns 1 after 6 h of accumulated WAITING (30-min checks); running time does not count (task section 9)
  while true; do
    [ -e $STOPF ] && exit 0
    GPU=$(pick); L=$(cut -d" " -f1 /proc/loadavg)
    echo "[$(date -u -Is)] check: gpu='$GPU' load=$L waited=${WAITED}s"
    if [ -n "$GPU" ] && awk "BEGIN{exit !($L < 4)}"; then return 0; fi
    [ $WAITED -ge 21600 ] && return 1
    sleep 1800; WAITED=$((WAITED + 1800))
  done
}
run() {  # name, args...
  name=$1; shift
  if ls out/v3/e/$name/batch*.json > /dev/null 2>&1; then echo "[$(date -u -Is)] $name already done, skipped"; return; fi
  if ready; then
    echo "[$(date -u -Is)] E $name on GPU $GPU $(nvidia-smi -i $GPU --query-gpu=uuid --format=csv,noheader)"
    CUDA_VISIBLE_DEVICES=$GPU $P -m analysis_v3.e_bench --out out/v3/e/$name "$@" > out/v3/e/$name.log 2>&1 || echo "FAILED $name"
  else
    echo "SKIPPED $name: no free card with load < 4 within 6 h"
  fi
}
mkdir -p out/v3/e
run batch_06b --batches 1 4 8 16 32 --rounds 5
run timing_06b --batches 1 16 --rounds 2 --steps 72 --policies lru ema belady --timing
run slow_06b --batches 8 32 --rounds 5 --policies lru ema alloc_ema belady allmiss nofetch \
    --delay-ns-per-byte 0.04 0.12 0.28 0.6
run ps32_06b --batches 1 8 32 --rounds 5 --page-size 32 --budget 96 --alpha 0.5 --policies lru ema belady allmiss nofetch
run ps64_06b --batches 1 8 32 --rounds 5 --page-size 64 --budget 96 --alpha 0.5 --policies lru ema belady allmiss nofetch
run k128_06b --batches 1 8 32 --rounds 5 --top-k 128 --budget 192 --alpha 0.5 --policies lru ema belady allmiss nofetch
run batch_8b --split splits/longctx_8b_v1.json --frozen splits/heldout_frozen_8b.json --batches 1 4 8 16 --rounds 5
run batch_8b_12k --split splits/longctx_8b_v1.json --frozen splits/heldout_frozen_8b.json --tokens 12288 --batches 32 --rounds 5
echo "[$(date -u -Is)] E DONE"
