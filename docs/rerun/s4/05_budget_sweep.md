# S4.5 Budget sweep B/K = 1 … 8 and iso-miss capacity (dev)

**Conclusion.** Capacity matters much more than the policy. The score EMA's saving equals **5–7 % more LRU slots**
at every budget from B/K = 1.25 to 6 (iso-miss capacity ratio 1.04–1.07). The optimum (Bélády) is worth 18–46 % more
slots. Going from B=96 to B=128 cuts LRU misses by 41 %; replacing LRU with EMA at B=96 cuts them by 10 %. This
agrees with FlashOffloading's "capacity over policy" and with HiSparse (doubling the cache gains more than the whole
LRU–Bélády gap). It supports describing the policy gain as a modest but free gain at a fixed memory budget, not as a
substitute for capacity.

Source: `results_v2/dev/s4/budget_sweep/` (`iso_capacity.csv`, `miss_budget.png`) and `baselines/summary.csv` (every
policy × every budget, G with CI, compulsory share, iso capacity). Iso capacity = the LRU budget with the same total misses,
linearly interpolated on an LRU grid of B = 64, 72, …, 1024.

| B (B/K) | LRU miss % | Bélády % | EMA (LODO) G [CI] | EMA iso-capacity | current-score iso-capacity | Bélády iso-capacity | compulsory share of LRU misses |
|---|---|---|---|---|---|---|---|
| 64 (1) | 15.87 | 15.87 | – (no choice at B=K) | 1.000 | 1.000 | 1.000 | 0.051 |
| 80 (1.25) | 9.01 | 6.26 | 0.274 [0.271, 0.276] | 1.043 | 1.038 | 1.182 | 0.090 |
| 96 (1.5) | 6.04 | 3.87 | 0.279 [0.274, 0.283] | 1.058 | 1.050 | 1.275 | 0.135 |
| 128 (2) | 3.57 | 2.18 | 0.267 [0.258, 0.276] | 1.070 | 1.055 | 1.377 | 0.228 |
| 192 (3) | 1.94 | 1.25 | 0.238 [0.221, 0.253] | 1.070 | 1.044 | 1.449 | 0.420 |
| 256 (4) | 1.36 | 0.97 | 0.236 [0.213, 0.257] | 1.068 | 1.031 | 1.464 | 0.599 |
| 384 (6) | 0.95 | 0.83 | 0.253 [0.197, 0.304] | 1.059 | 1.021 | 1.428 | 0.855 |
| 512 (8) | 0.84 | 0.81 | 0.254 [0.114, 0.344] | 1.035 | 1.013 | 1.411 | 0.968 |

From B/K = 3 on, most LRU misses are compulsory (first use; no policy can avoid them), so G is computed on a small
remainder and its CI widens. The current score's advantage fades with capacity (iso 1.05 → 1.01), while the EMA's stays near
1.06–1.07 up to B/K = 4.

- At B=K every policy is identical (no unselected resident to choose).
- ARC, SIEVE, S3-FIFO and InfLLM's rules stay at iso capacity ≈ 1.00 everywhere. Random is 0.80–0.91.
- The figure `miss_budget.png` plots miss rate and G against B/K for the main policies.
