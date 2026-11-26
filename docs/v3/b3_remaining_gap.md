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
(pending: `results_v3/inst/{dev,8b}/errors.csv`)

## Learned upper bound (boosted trees, richer features, LODO + DAgger)
(pending: `results_v3/learn/`)
