# Replication preregistration: score-EMA page cache, cross-hardware (replication-v2)

Written and committed **before any held-out trace of this replication is collected**. The commit time of
this file and the `started_at` timestamp in the manifest of the first held-out trace
(`/data/weibing/projects/rank-kv-cache/data/heldout/*.json` on wulab2) are both recorded in
`docs/rerun/s5_replication.md`.

## What is replicated
The preregistered held-out evaluation of `docs/efficient_policy_dev.md` (run once on 2026-11-23, Colab L4,
results in `docs/heldout_results.md`), re-collected on different hardware:

| | Original | This replication |
|---|---|---|
| GPU | NVIDIA L4 (Colab) | NVIDIA RTX 6000 Ada, wulab2 card 0, UUID `GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3` |
| Software | torch 2.14.1+cu130 | torch 2.6.0+cu124, transformers 4.53.3 (`docs/rerun/environment.md`) |
| Collection code | `notebooks/long_trace_capture.py` | `collect/long.py` (same selector, prompts, stop rule, precision; `tests/test_collect_long.py` checks identical selection) |

Quest-style selection is known to depend on the GPU (STATUS_REPORT: A100 and L4 select different pages from
step 0 on the same prompt), so the traces will differ from the L4 traces. The question is whether the
**conclusions** hold on traces produced by a different GPU and software stack.

## Frozen inputs (unchanged; nothing from S3–S4 of this replication may change them)
- Documents: the 21 `heldout` documents of `splits/longctx_v1.json` (15 gov_report, 6 qmsum).
- Model and data: `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, `zai-org/LongBench@5e628be450b7e67fb7ae6e201bd6d8f7056f7672`.
- Selector and decoding: page 16, K=64, greedy, thinking on, max 1024 new tokens, stop at EOS / max / newest 32-gram repeat (`splits/longctx_v1.json`).
- Policy parameters: `splits/heldout_frozen_v1.json` (EMA α per budget, per-layer allocation, linear coefficients), fitted on the original L4 dev traces. They are **not** refitted on the new dev traces, even though S3–S4 re-collect dev on the new GPU. Baseline parameters are those fixed in `heldout.py configs()`.

## Commands (run once each, in this order, on wulab2, card 0)
```
# 1. collect (CUDA_VISIBLE_DEVICES=0); resumable per document
python -m collect.long --split splits/longctx_v1.json --subset heldout --out data/heldout --dense-reference
# 2. evaluate (checks the folder against the split: exactly the 21 held-out rids, 28 layers, K=64)
python heldout.py eval data/heldout --frozen splits/heldout_frozen_v1.json --out results_v2/heldout/eval --procs 21
# 3. criteria and budget curve from the CSV
python recompute_heldout.py results_v2/heldout/eval/misses.csv > results_v2/heldout/criteria.txt
# 4. same-allocation anchors for Alloc+EMA
python validation/alloc_oracle.py heldout data/heldout --out results_v2/heldout/alloc_oracle \
    --misses results_v2/heldout/eval/misses.csv --frozen splits/heldout_frozen_v1.json --workers 14
# 5. independent LRU/Bélády check of the replay
python -m analysis_v2.audit_v2 data/heldout --misses results_v2/heldout/eval/misses.csv \
    --out results_v2/heldout/audit.txt --budgets 72 80 96 128 256
```
Step 1 may be re-run only to resume after an interruption: documents with a complete trace and matching
manifest sha256 are skipped, so no document is ever collected twice. Steps 2–5 read only the frozen file
and the new held-out traces.

## Decision criteria (B = 96, identical to the original preregistration)
| # | Criterion | Threshold |
|---|---|---|
| (i) | Score-EMA G = (LRU − EMA)/(LRU − Bélády), pooled over documents and layers, 95 % document-bootstrap CI lower bound (2000 reps, seed 0) | > 0.15 |
| (ii) | Documents on which EMA has fewer misses than LRU | ≥ 18 / 21 |
| (iii) | Allocation + EMA beats EMA alone: paired bootstrap CI of ΔG (uniform-budget anchors) | lower bound > 0 |

**Replication succeeds** if all three criteria pass. Each criterion is reported separately; a partial
pass is reported as such.

Also reported (not criteria): the whole budget curve (B = 72, 80, 96, 128, 256) for every policy in
`heldout.py`; the repeat-miss drop vs LRU at equal total slots; for Allocation + EMA, the G against the
**same-allocation** Bélády (`G_vs_alloc_belady`) and the replacement-only gain at that allocation
(`G_within_alloc`) from `validation/alloc_oracle.py`; per-document G; the L4 numbers side by side; the share
of tied scores (float16) in the new traces; and whether generated tokens match between the sparse
and dense reference runs.

## Deviations
A deviation is any departure from the above: a different command or argument, a re-run of steps 2–5,
a document that could not be collected or was collected more than once, a change of GPU (then the new UUID
is recorded and one already-collected document is re-collected on the new card and compared), any change
to a frozen file, or any use of held-out output to choose a parameter. Every deviation is listed in
`docs/rerun/s5_replication.md` and in the PR. Known, preregistered differences from the original (GPU,
torch version, collection script) are not deviations.

## If the replication fails
Report the failing criterion with its point estimate and CI, the L4 value, the per-document distribution,
and the LRU–Bélády gap on the new traces (a small gap makes G noisy). No parameter is re-tuned on held-out
data and the held-out set is not evaluated again. The thesis (`docs/rerun/thesis.md`) is weakened
accordingly.
