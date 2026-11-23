# Dev results (14 documents, longctx_v1 dev split)

Produced by `efficient_dev.py` (main study) and `checks.py` (checks A–E). Write-up:
[`docs/efficient_policy_dev.md`](../../efficient_policy_dev.md). The traces are not in git.

Shared columns: `doc` = task/doc id; `B` = per-layer slot budget (K=64); `misses` / `accesses` are page
counts over all decode steps; `compulsory` = first-time accesses; G = (LRU − policy)/(LRU − Bélády) misses.

| File | Contents | Doc section |
|---|---|---|
| `misses.csv` | Per doc, layer, config, B: accesses, misses, compulsory, for every fixed config. This is the raw source of everything below. | all |
| `misses_ext.csv` | Same layout, for the ablation configs added by `checks.py`. | C. Mechanism and ablations |
| `summary.csv` | Per policy family and B: misses, miss rate, repeat-miss rate, G with bootstrap CI (`G_lo`, `G_hi`), repeat-miss drop vs LRU. | Headline (B=96) |
| `lodo.csv` | Leave-one-doc-out: the config chosen per family on the other 13 docs, and its misses on the held-out doc. | Headline, B. Hyperparameter robustness |
| `learned.csv` | Linear next-use regression (`model` = BC / DAgger): misses per doc, plus fitted coefficients. | Learned policies |
| `alloc.csv` | Per-layer allocation per held-out doc: base policy, config, misses, slots per layer. | Headline (alloc + EMA), B6/C10 |
| `gap.csv` | LRU eviction decisions at B=96: feature AUCs and error stats (`kind`, `item`, `value`, `note`). | Gap analysis |
| `bypass.csv` | Evictions and use-once evictions (page admitted on the previous step and never hit). | C10 |
| `checks_stats.csv` | Per policy and B: miss rate, G and CI, repeat-miss drop and CI, docs better than LRU (of 14), worst-doc relative change, LODO choice. | A. Statistical stability |
| `checks_alloc.csv` | Allocation: uniform vs LODO vs oracle misses, share of the oracle gain, fold std, median allocation. | B6/C10 |
| `checks_alloc_methods.csv` | G of greedy vs exact knapsack vs gap-proportional allocation. | B6/C10 |
| `checkD.csv` | External-validity checks (second model / second prompt): misses per run. | D. External validity |
| `checks_report.txt` | Text report from `checks.py` with all the above tables. | Checks A–E |
