# Preregistration: third held-out set (followup-v3 F2; confirmatory)

Written after all dev analyses of followup-v3 parts B–D and **committed and pushed before any trace of the third held-out set is
collected**. The commit time of this file (GitHub) and the UTC start time written by the collection script into
`results_v3/provenance/f3_collect_*.log` (and the first manifest's `started_at`) are both reported in `docs/v3/f3_heldout.md`.

## Setting
- Documents: `splits/longctx_v3.json` (21 GovReport documents from the original GovReport dataset, none in any earlier split;
  survey `docs/v3/f1_pool.md`). The 8B collection uses `splits/longctx_v3_8b.json`, the same prompts with the 8B model.
- Models: `Qwen/Qwen3-0.6B@c1899de2…` and `Qwen/Qwen3-8B@b968826d…` (same revisions as before).
- Prompt (collect.long gov_report template in the Qwen3 chat template, thinking on), selector (Quest, page 16, K=64), decoding
  (greedy, max 1024 new tokens, stop at EOS / max / repeated 32-gram): identical to `longctx_v1` / `longctx_8b_v1`.
- GPU: one free wulab2 card chosen by `free-gpus` (card 2 avoided if another is free; UUID recorded in the manifests).
  The traces do not depend on the card (deterministic selection; earlier checks were bit-identical across collections).

## Frozen parameters: `splits/heldout_frozen_v3.json` (dev only)
Written by `python -m analysis_v3.freeze_v3` (command below) from dev results only:
- **EMA α**: the values already frozen for the two earlier confirmations (`splits/heldout_frozen_v1.json` for 0.6B: B=96 α 0.5,
  B=128 α 0.7; `splits/heldout_frozen_8b.json` for 8B: 0.6 / 0.7).
- **Offline allocation** (reported for H3 and the share): the per-layer budgets frozen in the same two files.
- **P2 candidate**: normalised power-law kernel of z (`kvpolicy_v3.Kernel(kernel="power", normalize=True, zero="kth")`), γ = the
  value with the fewest pooled dev misses (0.6B and 8B separately; every LODO fold chose the same γ: 1.0 at B=96 and 128 for both
  models). Dev evidence: `docs/v3/b1_powerlaw.md` (C gate passed on both models).
- **P4 candidate**: one global pool of L·B slots for all layers (`analysis_v3.c4_alloc.pool_run`), victim = the resident
  page with the lowest value, value = EMA (α 0.5) of the page's all-page score percentile in its layer + β × the layer's running
  miss rate; β with the fewest pooled dev misses (every LODO fold chose the same value): 0.6B β = 0.5 at B=96 and 2.0 at B=128;
  8B β = 2.0 at both. No offline statistic is used. Dev evidence: `docs/v3/c4_allocation.md` (share of the offline allocation's
  gain at B=96: 0.95 on 0.6B, 0.93 on 8B; at B=128: 0.74 / 0.60).

## Hypotheses (B = 96 primary, B = 128 secondary; each model separately; 21 documents; document bootstrap 2000, seed 0)
| # | Hypothesis | Criterion |
|---|---|---|
| H1 | The score EMA still narrows the LRU–Bélády gap on a third document set | G CI lower bound > 0.20 **and** ≥ 18/21 documents with fewer misses than LRU. The threshold 0.20 was set from the two earlier results (0.279 on 0.6B, 0.272 on 8B), i.e. after seeing them, as a margin below both. |
| H2 | The P2 candidate beats the EMA | paired ΔG (P2 − EMA) CI lower bound > 0 |
| H3 | The P4 candidate (global pool, no offline statistics) beats uniform per-layer budgets with the EMA | paired ΔG (P4 − uniform EMA, both against uniform-budget LRU/Bélády anchors) CI lower bound > 0. Also reported, without a threshold: the share of the frozen offline allocation's gain, (EMA_uniform − P4)/(EMA_uniform − alloc+EMA) in misses. |

Each hypothesis is evaluated at B=96 for the decision; B=128 is reported with the same statistics as the secondary budget (a
pass at 96 with a negative point estimate at 128 is reported as such). H1–H3 are reported per model; there is no combined test.

## Commands (run once each, in order)
```
# collection (wulab2; the script prints the UTC start time and the GPU UUID into its log)
runs/run_f3_collect.sh            # collect.long for 0.6B (splits/longctx_v3.json) then 8B (splits/longctx_v3_8b.json),
                                  # --subset heldout, outputs data/heldout_v3_06b and data/heldout_v3_8b
# evaluation (CPU)
python -m analysis_v3.f_eval replay data/heldout_v3_06b --frozen splits/heldout_frozen_v3.json --model 06b \
    --split splits/longctx_v3.json --out results_v3/f3/06b --procs 16
python -m analysis_v3.f_eval criteria --frozen splits/heldout_frozen_v3.json --model 06b --out results_v3/f3/06b
python -m analysis_v3.f_eval replay data/heldout_v3_8b --frozen splits/heldout_frozen_v3.json --model 8b \
    --split splits/longctx_v3_8b.json --out results_v3/f3/8b --procs 16
python -m analysis_v3.f_eval criteria --frozen splits/heldout_frozen_v3.json --model 8b --out results_v3/f3/8b
python -m analysis_v2.audit_v2 data/heldout_v3_06b --misses results_v3/f3/06b/misses.csv --out results_v3/f3/06b/audit.txt \
    --budgets 96 128
python -m analysis_v2.audit_v2 data/heldout_v3_8b --misses results_v3/f3/8b/misses.csv --out results_v3/f3/8b/audit.txt \
    --budgets 96 128
```
Also reported (not criteria), all produced by the commands above (`criteria.json`): G and CI of every frozen policy at both
budgets, documents better than LRU, G of the offline allocation within its own allocation (same-allocation LRU/Bélády), and
the P4 share of the offline-allocation gain. Any analysis of these traces that is not listed here is a deviation and will be
labelled post-hoc.

The collection may only be resumed after an interruption (finished documents are skipped); no document is collected twice.
Nothing measured on these traces is used to choose or tune anything.

## If a hypothesis fails
It is reported as failed with its numbers; the dev result stays exploratory. If H1 fails, the EMA claim is reported as not
replicated on this document set, with the per-document G. If H2 fails, P2 is reported as "better on dev only". If H3 fails, P4
as "dev only". A partial collection (fewer than 21 documents after the interruption rule) is evaluated on the documents
collected, and the sample size is stated.

## Deviations
Any change of these commands or arguments, a re-run of an evaluation step, a document not collected or collected twice, an
edit of a frozen file, or any use of held-out output for a decision. All are listed in `docs/v3/f3_heldout.md` and
`docs/v3/pr_description.md`.
