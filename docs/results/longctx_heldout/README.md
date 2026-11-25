# Held-out results (21 documents, preregistered, run once)

Produced by `python heldout.py eval HELDOUT_TRACES --frozen splits/heldout_frozen_v1.json --out DIR`.
Write-up: [`docs/heldout_results.md`](../../heldout_results.md). The traces (npz) are not in git; they are on Drive
under `MyDrive/rank-kv-cache/longctx_v1_heldout/`.

| File | Contents |
|---|---|
| `misses.csv` | One row per (config, doc, layer, B). Columns: `config` (lru, belady, ema, score, dsum = InfLLM-style decayed sum, dcount = decayed count, linear = DAgger next-use regression, alloc_ema = per-layer allocation + EMA), `doc` (task/doc id), `layer` (0–27), `B` (total per-layer slot budget; for alloc_ema it is the average budget, and the layer's own budget is in `frozen.json`), `accesses` (selected pages over all decode steps), `misses` (pages not resident when selected), `compulsory` (first-time accesses). It is the source for both tables in `heldout_results.md`: the success criteria and the budget curve. |
| `doc_level.csv` | Per (config, B, doc): `accesses`, `misses`, `rep` = misses − compulsory (repeat misses). It is used for the per-document win counts and the repeat-miss drop. |
| `frozen.json` | A copy of `splits/heldout_frozen_v1.json`: per-budget EMA α, the per-layer allocation, linear coefficients and the dev check, all fitted on dev only. |

G = (LRU misses − policy misses) / (LRU misses − Bélády misses), pooled over docs and layers. The CIs use a
document-clustered bootstrap (2000 resamples, seed 0).

Recompute the criteria and the budget curve from `misses.csv` alone (needs numpy only):

    python recompute_heldout.py docs/results/longctx_heldout/misses.csv
