# D7 (N7): high budgets — report absolute miss reductions and iso-miss capacity, not G

**Conclusion: above B/K = 4 the gap between LRU and Bélády almost vanishes, so G is a ratio of two tiny numbers and its CI
becomes uninformative; the absolute saving and the iso-miss capacity stay well defined.** At B=512 (B/K = 8) 97 % of LRU's misses
are first accesses, LRU is 0.026 percentage points above Bélády, and the EMA's G 0.254 has the CI [0.114, 0.344]. The same EMA
saves 0.026 pp of accesses there (1 283 of 19.3 M accesses) and is worth 3.5 % more LRU slots. G is reported only for
B/K ≤ 4 in the followup-v3 pages; above that, the tables below are used.

Source: replication-v2 dev replay (`results_v2/dev/s4/baselines/summary.csv`; 14 0.6B dev documents, 19 308 800 page accesses
per budget; LODO-tuned EMA; iso-miss capacity = LRU budget with the same total misses, linear interpolation on the LRU grid
64–1024 step 8). No new replay.

| B (B/K) | LRU miss % | Bélády miss % | LRU − Bélády (pp) | compulsory share of LRU misses | EMA miss % | EMA saving vs LRU (pp, misses) | EMA repeat-miss drop | EMA iso-miss capacity | EMA G [CI] |
|---|---|---|---|---|---|---|---|---|---|
| 96 (1.5) | 6.037 | 3.867 | 2.170 | 0.135 | 5.433 | 0.604 (116 694) | 11.6 % | 1.058 | 0.279 [0.274, 0.283] |
| 128 (2) | 3.575 | 2.183 | 1.392 | 0.228 | 3.203 | 0.372 (71 829) | 13.5 % | 1.070 | 0.267 [0.258, 0.276] |
| 192 (3) | 1.938 | 1.245 | 0.693 | 0.420 | 1.773 | 0.165 (31 803) | 14.7 % | 1.070 | 0.238 [0.221, 0.253] |
| 256 (4) | 1.361 | 0.967 | 0.394 | 0.599 | 1.268 | 0.093 (17 927) | 17.0 % | 1.068 | 0.236 [0.213, 0.257] |
| 384 (6) | 0.952 | 0.829 | 0.123 | 0.855 | 0.921 | 0.031 (6 013) | 22.6 % | 1.059 | (0.253 [0.197, 0.304]) |
| 512 (8) | 0.841 | 0.815 | 0.026 | 0.968 | 0.834 | 0.007 (1 283) | 25.0 % | 1.035 | (0.254 [0.114, 0.344]) |

Reading:
- **Why G degrades.** G = (LRU − pol)/(LRU − Bélády). The denominator falls from 419 k misses (B=96) to 5 k (B=512) because almost
  every remaining miss is compulsory, which no policy can avoid. Per-document bootstrap noise in a few thousand avoidable
  misses then dominates (CI width 0.01 at B=96, 0.23 at B=512). The current score's G at B=512 has a CI that includes 0
  ([−0.146, 0.243]) although it still beats LRU in pooled misses.
- **What stays meaningful.** The repeat-miss drop (the share of avoidable misses removed) keeps growing with B (11.6 % → 25 %),
  and the iso-miss capacity stays at 1.03–1.07: at every budget the EMA is worth 3–7 % more LRU slots (D1 relates this to
  bytes and MiB).
- **Rule used in followup-v3:** G with CI for B/K ≤ 4 only; for B/K > 4, absolute miss reduction (pp and count), repeat-miss
  drop and iso-miss capacity. N7 ("high-budget CI is wide") is therefore a property of the metric, not of the policy.
