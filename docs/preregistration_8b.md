# Preregistration: score-EMA page cache on Qwen3-8B (confirmatory, new held-out)

Written after the 8B dev analysis and **committed and pushed before any 8B held-out trace is collected**. The commit time of
this file and the `started_at` of the first 8B held-out manifest (`data/8b_heldout/*.json` on wulab2) are recorded in
`docs/rerun/s6_extensions.md`.

## Setting
- Model: `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` (36 layers, 32 query heads, 8 KV heads, head dim 128,
  bf16). 64 KiB per page per layer, as for 0.6B.
- Documents: `splits/longctx_8b_v1.json`: 14 dev + 21 held-out GovReport/QMSum documents that are **not** in
  `splits/longctx_v1.json` (25 at 16–32K tokens, all available; 10 at 12–16K, because the 16–32K pool was too small;
  see the split's `rationale`). Held-out: 20 gov_report, 1 qmsum.
- Prompt, selector, decoding and stop rule: identical to longctx_v1 (`collect/long.py`; page 16, K=64, greedy, thinking
  on, max 1024 new tokens, stop at EOS / max / repeated 32-gram). Collection records `attn_mass` and `true_rho`
  (read-outs that do not change the selection; `tests/test_collect_long.py`).
- GPU: wulab2 card 0, `GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3`.

## Frozen parameters: `splits/heldout_frozen_8b.json`
Fitted on the 14 8B dev traces only (`python -m analysis_v2.freeze_8b data/8b_dev --split splits/longctx_8b_v1.json
--out splits/heldout_frozen_8b.json`): per budget B ∈ {72, 80, 96, 128, 256}, the EMA α with the fewest pooled dev misses
over {0.1, …, 0.9, 0.95} (the choice LODO converges to; the per-fold LODO choices are stored in the file); then the
original `heldout.py freeze` procedure, unchanged: per-layer allocation from the dev EMA curves (greedy, total 36·B),
two-iteration DAgger linear coefficients, and dev check counts. The values are in the committed file. Baseline
parameters are those fixed in `heldout.py configs()`.

Frozen values (from `splits/heldout_frozen_8b.json`, committed together with this file):
- EMA α: B=72 0.4, 80 0.5, **96 0.6**, 128 0.7, 256 0.95 (every LODO fold picked the same value at every budget).
- Allocation at B=96 (36 layers, total 3 456 slots): 72–128 slots per layer, non-uniform (so criteria iii and iv apply).
  B=72 and 256 are near-uniform/uniform by the grid limits.
- Dev check (unchanged freeze code: LRU, EMA at α=0.5, Bélády at B=96 on the 14 8B dev traces): accesses 32 997 888;
  misses LRU 1 511 135, EMA 1 367 622, Bélády 982 501 (dev G ≈ 0.27).

## Commands (run once each, in order, on wulab2 card 0)
```
python -m collect.long --split splits/longctx_8b_v1.json --subset heldout --out data/8b_heldout \
    --record-attn-mass --record-true-attn-rank
python heldout.py eval data/8b_heldout --frozen splits/heldout_frozen_8b.json --split splits/longctx_8b_v1.json \
    --out results_v2/ext/8b_heldout/eval --procs 21
python recompute_heldout.py results_v2/ext/8b_heldout/eval/misses.csv > results_v2/ext/8b_heldout/criteria.txt
python validation/alloc_oracle.py heldout data/8b_heldout --out results_v2/ext/8b_heldout/alloc_oracle \
    --misses results_v2/ext/8b_heldout/eval/misses.csv --frozen splits/heldout_frozen_8b.json --workers 14
python -m analysis_v2.audit_v2 data/8b_heldout --misses results_v2/ext/8b_heldout/eval/misses.csv \
    --out results_v2/ext/8b_heldout/audit.txt --budgets 72 80 96 128 256
```
The first command may only be resumed after an interruption (finished documents are skipped); no document is collected twice.

## Decision criteria (B = 96; the three of `heldout_frozen_v1`, plus the same-allocation version)
| # | Criterion | Threshold |
|---|---|---|
| (i) | Score-EMA G (uniform anchors, pooled), 95 % document-bootstrap CI lower bound (2000 reps, seed 0) | > 0.15 |
| (ii) | Held-out documents with EMA misses < LRU misses | ≥ 18 / 21 |
| (iii) | Allocation + EMA beats EMA: paired ΔG CI (uniform anchors) | lower bound > 0 |
| (iv) | Allocation + EMA against the **same-allocation** anchors: `G_within_alloc` CI from `validation/alloc_oracle.py` (EMA vs LRU, both at the frozen per-layer budgets) | lower bound > 0.15 |

The 8B result is **confirmed** if (i)–(iii) pass, as for 0.6B. (iv) is reported with the same weight and decides
whether the allocation claim can be stated against the oracle at the same allocation. If the frozen allocation at
B=96 is uniform, (iii) and (iv) are reported as "not applicable" (the allocation found no gain on dev).

Also reported (not criteria): budget curve for every `heldout.py` policy, repeat-miss drop, per-document G, the
InfLLM baselines with the true attention mass, and the Spearman ρ between Quest scores and true attention.

## Deviations
Any change to the commands or arguments, a re-run of evaluation steps, a document not collected or collected twice, a GPU
change, any edit of a frozen file, or any use of held-out output for a decision. All are listed in
`docs/rerun/s6_extensions.md` and the PR. If the criteria fail, the 8B claim is reported as not confirmed. Nothing is
re-tuned and the held-out set is not evaluated again.
