# B3: where the remaining gap is

**Conclusion: the remaining 72 % of the gap is almost entirely about the next ~16 steps.** A Bélády that knows the future only
H steps ahead (and orders the pages not needed within H by LRU or by the EMA) closes 24–38 % of the gap with H = 1, 65–69 %
with H = 4, 96 % with H = 16 and 99.9 % with H = 64 at B=96 on 0.6B dev (8B: 23–38 %, 62–67 %, 94 %, 99.7 %). The EMA's 0.279 is
about what one step of perfect lookahead with LRU ordering buys (0.237); with EMA ordering a single step of lookahead is worth
0.376. The needed horizon grows with the budget (B=256 needs H ≈ 64 for 70 %). [Error classes and the learned upper bound:
sections below.]

Source: `analysis_v3.run b3` (`kvpolicy_v3.Lookahead`; tests: H = 0 equals LRU / the EMA, H = ∞ equals Bélády, misses fall with H);
`results_v3/b3/{b3_dev,b3_8b}/`; audits 3 136 / 4 032 streams, 0 mismatches.

## Finite-lookahead Bélády (G; the order beyond the horizon is LRU / EMA α 0.5)
| H (steps) | 0.6B B=80 | 0.6B **B=96** | 0.6B B=128 | 0.6B B=256 | 8B B=96 | 8B B=128 |
|---|---|---|---|---|---|---|
| 1 | 0.445 / 0.537 | **0.237 / 0.376** | 0.112 / 0.286 | 0.027 / 0.168 | 0.230 / 0.375 | 0.100 / 0.271 |
| 4 | 0.910 / 0.920 | **0.651 / 0.692** | 0.356 / 0.435 | 0.099 / 0.201 | 0.617 / 0.667 | 0.324 / 0.417 |
| 16 | 0.998 / 0.998 | **0.962 / 0.964** | 0.794 / 0.811 | 0.300 / 0.358 | 0.937 / 0.942 | 0.749 / 0.775 |
| 64 | 1.000 / 1.000 | **0.999 / 0.999** | 0.974 / 0.976 | 0.702 / 0.720 | 0.997 / 0.997 | 0.962 / 0.965 |
| 256 | 1.000 | 1.000 | 0.999 | 0.981 | 1.000 | 0.998 |
| EMA alone (no lookahead) | 0.274 | 0.279 | 0.267 | 0.236 | 0.272 | 0.256 |

Reading: a page evicted wrongly is one that comes back within a few steps. Predicting re-selection 1–16 steps ahead is the whole
problem; long-range knowledge adds nothing at B ≤ 128. This is consistent with B7 (the information that matters is the current
score of the pages just below the selection cut) and with the persistence of the selection (84 % of the top-K is selected again
at the next step).

## EMA errors against Bélády's choice
At every eviction of the EMA (α 0.5, B=96) a victim counts as an error if it comes back sooner than the n-th farthest next use
among the candidates (n = victims at that step), i.e. if Bélády would have kept it. 62 % of the EMA's evictions are errors on
both models (0.6B 627 669 of 1 011 386; 8B 815 808 of 1 319 238). Where they are (`results_v3/inst/{dev,8b}/errors.csv.gz`,
`summary.json`):

| Group | bin | 0.6B victims | 0.6B error rate | share of 0.6B errors | 8B error rate | share of 8B errors |
|---|---|---|---|---|---|---|
| steps resident | 1–3 | 252 531 | 0.50 | 20 % | 0.49 | 18 % |
| | 4–15 | 481 903 | 0.62 | 47 % | 0.61 | 44 % |
| | 16–63 | 212 102 | 0.72 | 24 % | 0.70 | 27 % |
| | ≥ 64 | 64 850 | 0.79 | 8 % | 0.79 | 11 % |
| all-page score percentile | < 0.5 | 8 198 | 0.22 | 0.3 % | 0.26 | 0.7 % |
| | 0.5–0.8 | 211 379 | 0.49 | 17 % | 0.52 | 24 % |
| | **0.8–0.9** | 628 093 | 0.65 | **65 %** | 0.66 | **63 %** |
| | 0.9–0.95 | 163 138 | 0.70 | 18 % | 0.70 | 12 % |
| returns after (steps) | 1–3 | 166 360 | 0.98 | 26 % | 0.93 | 25 % |
| | 4–15 | 331 319 | 0.85 | 45 % | 0.83 | 41 % |
| | 16–63 | 258 186 | 0.57 | 23 % | 0.61 | 25 % |
| | ≥ 64 / never | 255 521 | 0.28 / 0 | 6 % | 0.34 / 0 | 9 % |
| phase | thinking / answer | 423 532 / 587 854 | 0.63 / 0.61 | | 0.64 / 0.59 | |

Reading: the typical error is a page ranked just below the selection cut (percentile 0.8–0.9 of all pages; the cut is near
1 − 64/1 400 ≈ 0.95), resident for a few to a few dozen steps, that Quest selects again within 16 steps (71 % of errors). The EMA
cannot tell these pages from their neighbours in rank that will not return; both look the same in a smoothed score. Errors are
equally frequent in thinking and answering, and slightly more frequent for long-resident pages (their smoothed values are the
most stale).

## Learned upper bound (boosted trees, richer features, LODO + DAgger)
(pending: `results_v3/learn/`)
