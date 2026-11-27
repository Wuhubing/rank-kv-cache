#!/bin/bash
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp CUDA_VISIBLE_DEVICES=0
while tmux has-session -t wb-rkv-s6-collect 2>/dev/null; do sleep 30; done
C=".venv/bin/python -m collect.long --split splits/longctx_v1.json --subset dev"
for ps in 8 16 32 64; do for k in 32 64 128; do
  [ "$ps" = 16 ] && [ "$k" = 64 ] && continue
  [ -e STOP ] && { echo STOP; exit 0; }
  echo "[$(date -Is)] sweep ps=$ps k=$k"
  $C --page-size $ps --top-k $k --out data/sweep/ps${ps}_k${k} || echo "FAILED ps=$ps k=$k"
done; done
for s in 0 1 2; do
  [ -e STOP ] && { echo STOP; exit 0; }
  echo "[$(date -Is)] sampling seed=$s"
  $C --temperature 0.6 --seed $s --out data/sample/seed$s || echo "FAILED seed $s"
done
echo "[$(date -Is)] S6B DONE"
