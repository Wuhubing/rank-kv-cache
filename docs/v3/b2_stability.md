# B2: why G is stable — trace statistics, a regression, and a synthetic model

**Conclusion: G is stable across settings because the temporal statistics of the selection barely change across settings; the
statistics explain per-layer differences only partly, and a simple latent-score model reproduces the shape of the α curves but
not their level.** Across 0.6B dev, 8B dev, 64K, RULER, SCBench, three sampled decodes and eleven page/K cells, the adjacent-step
overlap (0.82–0.87), the lag-1/16/64 autocorrelation of the unselected pages' score percentile (≈0.96 / 0.88 / 0.83) and the
re-selection probabilities are nearly the same, matching the narrow G range 0.27–0.31. A LODO-by-document linear regression on 14
such statistics predicts the per-(setting, layer) G with Spearman 0.60 at B/K = 1.5 and 0.77 at B/K = 2, and the setting-level G
with 0.38 / 0.55 (19 settings). No single statistic correlates with G by more than |ρ| = 0.24. The synthetic model fitted to the
0.6B dev statistics reproduces that the best α grows with B/K and that a large α falls below LRU, but it gives a higher G (0.38 at
B/K 1.5) and almost no benefit of history over the current score at small budgets. P1's support criterion (Spearman ≥ 0.7 at the
primary budget and a reproduced curve) is therefore not met; "G is stable at 0.27–0.31" stays an empirical fact, with the
explanation that the relevant trace statistics do not vary.

Source: `analysis_v3/b2_stats.py` (statistics, `results_v3/b2/stats/*.csv`: per document and layer, with LRU / EMA α 0.5 / Bélády
misses at B/K = 1.25–4), `analysis_v3/b2_model.py regress` (`results_v3/b2/regression/`), `fit` and `curves`
(`results_v3/b2/synthetic/`). Units with fewer than 20 avoidable misses (LRU − Bélády) are dropped from the regression.

## The statistics barely move across settings (means over documents and layers)
| Setting | overlap | ac lag 1 | ac lag 16 | ac lag 64 | P(selected at t+1 \| unselected at t) | P(t+16 \| ·) | reuse gap = 1 | LRU repeat misses back within 16 steps | EMA G at B/K=1.5 |
|---|---|---|---|---|---|---|---|---|---|
| 0.6B dev | 0.842 | 0.963 | 0.876 | 0.827 | 0.008 | 0.014 | 0.848 | 0.402 | 0.279 |
| 8B dev | 0.868 | 0.967 | 0.884 | 0.833 | 0.008 | 0.015 | 0.872 | 0.315 | 0.272 |
| 0.6B 64K (YaRN) | 0.822 | 0.968 | 0.898 | 0.859 | 0.004 | 0.008 | 0.828 | 0.466 | 0.284 |
| 0.6B RULER | 0.868 | 0.973 | 0.910 | 0.871 | 0.008 | 0.014 | 0.874 | 0.337 | 0.255 |
| 0.6B SCBench | 0.871 | 0.971 | 0.894 | 0.863 | 0.006 | 0.010 | 0.875 | 0.323 | 0.285 |

(The page/K cells and the sampled decodes are in the same files; their overlaps are 0.79–0.90.)

## Regression (LODO by document; linear least squares on 14 statistics)
| | B/K = 1.5 | B/K = 2 |
|---|---|---|
| units (setting, document, layer) | 7 341 | 6 686 |
| Spearman, predicted vs measured G, all units | 0.43 | 0.60 |
| **Spearman, (setting, layer) means** (534 / 513) | **0.60** | **0.77** |
| Spearman, setting means (19 settings) | 0.38 | 0.55 |
| Spearman within one setting, over its layers: 0.6B dev / 8B dev / 64K / RULER | 0.59 / 0.31 / 0.76 / 0.42 | 0.79 / 0.49 / 0.66 / 0.71 |
| largest single-statistic |ρ| with G | 0.22 (ac lag 64) | 0.24 (repeat misses back within 64 steps) |

The regression's weight sits on combinations of the re-selection probabilities and the overlap (they enter with opposite signs),
i.e. on how often a page just below the cut comes back, not on the autocorrelation as such.

Case study (0.6B dev, B=96, `results_v3/b2/stats/dev.csv`): layer 10 (EMA G 0.161) vs layer 27 (G 0.377). Both have high overlap
(0.916 / 0.934) and almost the same lag-1 autocorrelation (0.992 / 0.994). Layer 10's ordering of unselected pages is *more*
persistent at long lags (lag 64: 0.965 vs 0.932), and more of its LRU repeat misses come back within 16 steps (5.3 % vs 0.7 %).
In layer 10 the history adds little to the current score (the order hardly changes) and the misses that remain are fast returns a
smoothed score cannot anticipate; in layer 27 the order drifts over tens of steps, which is what an average over the history
captures.

## Synthetic latent-score model
Each page: static part + slow AR(1) + fast AR(1) + a recency boost for new pages; one new page per 16 steps; selection = top-K.
Fitted (grid) to 0.6B dev: φ_fast 0.95, φ_slow 0.998, variance shares 0.5 / 0.3 / 0.2, boost 0.5 (fit: overlap 0.885 vs 0.842,
ac lag 1 / 16 / 64 0.985 / 0.845 / 0.720 vs 0.963 / 0.876 / 0.827).

| B/K | best α (synthetic, fitted) | G at best α | G at α 0.95 / 0.99 | best α (0.6B dev, LODO) | real G (EMA) |
|---|---|---|---|---|---|
| 1.25 | 0.0 (current score) | 0.339 | −0.205 / −0.625 | 0.4 | 0.274 |
| 1.5 | 0.0–0.1 | 0.375 | 0.188 / −0.534 | 0.5 | 0.279 |
| 2 | 0.5 | 0.397 | 0.285 / 0.186 | 0.7 | 0.267 |
| 4 | 0.5–0.8 | 0.404 | 0.396 / 0.309 | 0.95 | 0.236 |

Reproduced: the best α increases with B/K, and a large α falls below LRU at small budgets (N5). Not reproduced: the level of G
and the benefit of history at small budgets (in the model the current score is already optimal at B/K ≤ 1.5). Regimes with lower
persistence (φ_fast 0.5–0.8, overlap 0.65–0.78) give lower G and a larger best α at B/K = 4 (0.7–0.8): history helps more when the
scores are noisier from step to step, which is the direction used for the α rule (D6). The model is used for D5 and D6.
