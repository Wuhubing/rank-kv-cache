#!/bin/bash
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp CUDA_VISIBLE_DEVICES=0
while tmux has-session -t wb-rkv-s6-sweep 2>/dev/null; do sleep 30; done
B=".venv/bin/python -m analysis_v2.s7_bench"
O=out/results_v2/gpu
run() { [ -e STOP ] && { echo STOP; exit 0; }; echo "[$(date -Is)] $* | load $(cut -d" " -f1-3 /proc/loadavg) | gpu0 apps: $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | tr "\n" " ")"; "$@" || echo "FAILED $*"; }
run $B single --ndocs 3 --budgets 96 128 --steps 384 --repeats 3 --trace-dir data/heldout --out $O/single_06b
run $B batch --ndocs 32 --tokens 16384 --batches 1 4 8 16 32 --budget 96 --steps 128 --repeats 2 --out $O/batch_06b
run $B single --ndocs 3 --budgets 96 --steps 384 --repeats 2 --cut 8192 --out $O/single_06b_8k
run $B single --ndocs 3 --budgets 80 256 --steps 384 --repeats 2 --trace-dir data/heldout --out $O/single_06b_b80_256
run $B single --split splits/longctx_8b_v1.json --subset heldout --ndocs 2 --budgets 96 --steps 256 --repeats 2 --frozen splits/heldout_frozen_8b.json --trace-dir data/8b_heldout --out $O/single_8b
run $B batch --split splits/longctx_8b_v1.json --ndocs 35 --tokens 16384 --batches 1 4 8 16 32 --budget 96 --steps 64 --repeats 2 --frozen splits/heldout_frozen_8b.json --out $O/batch_8b
run $B flops --out $O/flops.csv
echo "[$(date -Is)] S7 DONE"
for d in data/sweep/ps*_k*; do
  echo "[$(date -Is)] sweep explore $d"; .venv/bin/python -m analysis_v2.explore $d --out out/results_v2/ext/sweep/$(basename $d) --procs 32
done
for d in data/sample/seed*; do
  echo "[$(date -Is)] sampling explore $d"; .venv/bin/python -m analysis_v2.explore $d --out out/results_v2/ext/sampling/$(basename $d) --procs 14
done
echo "[$(date -Is)] S6C DONE"
