#!/bin/bash
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp CUDA_VISIBLE_DEVICES=0
while tmux has-session -t wb-rkv-collect-dev-attn 2>/dev/null; do sleep 30; done
C=".venv/bin/python -m collect.long --record-attn-mass --record-true-attn-rank"
for job in "splits/longctx_8b_v1.json data/8b_dev" "splits/longctx64k_v1.json data/64k_dev" "splits/ruler16k_v1.json data/ruler_dev"; do
  set -- $job
  [ -e STOP ] && { echo "STOP file present"; exit 0; }
  echo "[$(date -Is)] $1"
  $C --split $1 --subset dev --out $2 || echo "FAILED $1"
done
[ -e STOP ] && exit 0
echo "[$(date -Is)] scbench one session"
D=$(python3 -c "import json;print(json.load(open(\"splits/scbench_v1.json\"))[\"prompts\"][0][\"rid\"])")
$C --split splits/scbench_v1.json --subset dev --docs $D --out data/scbench_check || echo "FAILED scbench"
echo "[$(date -Is)] S6A DONE"
