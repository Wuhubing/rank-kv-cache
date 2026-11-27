# S4: all baselines and analyses on the new dev traces (CPU replay, dev only)

14 longctx_v1 dev documents re-collected on the RTX 6000 Ada (S3). Everything here is exploratory and uses dev only.
Results: `results_v2/dev/`. One page per experiment:

| Page | Plan item | One-line conclusion |
|---|---|---|
| [01_existing_baselines.md](01_existing_baselines.md) | S4.1, §6.1 | All 6.1 baselines reproduce L4; EMA G 0.279, current score 0.240, linear 0.300; the selected-only InfLLM rule ≈ LRU |
| [02_allrank_control.md](02_allrank_control.md) | S4.2 | **Global ordering, not magnitude**: all-page percentile EMA keeps 97 % of G from LRU (0.271 vs 0.279), about 80 % of the history's gain over the current score (ordering ≈ 4/5, magnitude ≈ 1/5) |
| [03_alloc_oracle.md](03_alloc_oracle.md) | S4.3 | Allocation and replacement add up; vs the same-allocation Bélády, alloc+score closes ~33 % of the gap |
| [04_new_baselines.md](04_new_baselines.md) | S4.1/S4.6, §6.2 | Random/SIEVE/S3-FIFO ≤ LRU; ArkVale pins and InfLLM rules weaker; Holt/CO2 add nothing; **power-law kernel 0.310** |
| [09_arkvale_full.md](09_arkvale_full.md) | follow-up | ArkVale as released (forced pages, window rule, layers 0–1 unbudgeted, cuboid-mean digest): G −0.219 at B=96 vs EMA 0.278; best digest 0.200; EMA advantage holds under ArkVale's own selector |
| [05_budget_sweep.md](05_budget_sweep.md) | S4.4 | EMA = 5–7 % more LRU slots; Bélády = 18–46 %; capacity dominates |
| [06_alpha_curve.md](06_alpha_curve.md) | S4.5 | One curve from current score (α=0) to frequency (α→1); optimum α ≈ 0.4–0.7 |
| [07_prediction_quality.md](07_prediction_quality.md) | S4.7, §5.2 | Top-K persists 84 %; the score predicts reuse of *unselected* pages beyond recency/frequency (stratified AUC 0.635) |
| [08_stats_noise_layers.md](08_stats_noise_layers.md) | S4.8, §5.4 | 14/14 wins; noisy scores fall below random, not back to LRU; per-layer G 0.16–0.38 |

## Check
- LRU and Bélády counts of every replay agree with the independent simulator in `replay.py`:
  `results_v2/dev/audit.txt` (efficient_dev, 3 136 streams) and `results_v2/dev/s4/audit.txt` (3 920 streams), 0 mismatches,
  Bélády ≤ LRU everywhere, no zero-miss stream.
- `validation/rank_control_dev.py` cross-checked its replay against the new `efficient_dev`/`checks` rows.
- Thesis point 1 is decided: "global ordering" (`docs/rerun/thesis.md`).

## How to reproduce (wulab2, traces in `data/dev`)
```
python efficient_dev.py data/dev --out out/results_v2/dev/efficient_dev --procs 32
python checks.py data/dev --res out/results_v2/dev/efficient_dev --procs 32
python validation/rank_control_dev.py data/dev --out out/results_v2/dev/rank_control --workers 32 --archive out/results_v2/dev/efficient_dev
python validation/alloc_oracle.py dev data/dev --out out/results_v2/dev/alloc_oracle --misses .../efficient_dev/misses.csv --alloc .../efficient_dev/alloc.csv
python -m analysis_v2.s4_dev replay data/dev --out out/results_v2/dev/s4 --procs 32   # then: summarize, predict, noise
python -m analysis_v2.audit_v2 data/dev --misses out/results_v2/dev/s4/misses.csv --out out/results_v2/dev/s4/audit.txt --budgets 64 96 128 256 512
```
Wall time on 32 cores: efficient_dev 16 min, checks <1 min, rank control 7 min, s4 replay 46 min, predict + noise 4 min (2026-11-25 06:57–08:15 UTC, shared with the S5/S6 GPU collection).
