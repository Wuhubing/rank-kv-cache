#!/bin/bash
# WRITTEN AFTER THE FACT (followup-v3 A9, 2026-11-25). The Qwen3-8B held-out traces (data/8b_heldout) were collected
# on 2026-11-25 08:35:54Z-10:06:51Z (manifest started_at / finished_at, results_v2/provenance/manifests/8b_heldout.jsonl)
# with a command typed into a tmux session; no script was kept. The command below is reconstructed from the
# manifests (split_file, subset, flags: record_attn_mass, record_true_attn_rank, greedy, max_new 1024; GPU index 0,
# UUID GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3) and the log logs/collect_8b_heldout.log
# (results_v2/provenance/logs/). Evaluation: results_v2/provenance/run_scripts/run_s6_8b_eval.sh (the script that was run).
set -e
cd /data/weibing/projects/rank-kv-cache
export TMPDIR=/data/weibing/tmp CUDA_VISIBLE_DEVICES=0
echo "[$(date -u -Is)] collect 8B held-out"
.venv/bin/python -m collect.long --split splits/longctx_8b_v1.json --subset heldout --out data/8b_heldout \
    --record-attn-mass --record-true-attn-rank 2>&1 | tee logs/collect_8b_heldout.log
bash run_s6_8b_eval.sh 2>&1 | tee logs/s6_8b_eval.log
