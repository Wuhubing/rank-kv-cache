# D6 (N5): why the best α grows with the budget, and whether α can be chosen without tuning

**Conclusion: a large α is harmful at small budgets and useful at large ones because the eviction candidates change: at small B
they are pages that left the selection a few steps ago and come back within a few steps, so only the latest score matters; at
large B they are pages deeper below the cut whose single-step scores are noisy, so averaging helps. A tuning-free per-layer α
rule derived from the synthetic model does not work (it loses 0.02–0.08 G to LODO), so the small positive "no tuning" result is
not obtained; per-layer α tuned LODO adds only +0.002 to +0.007.**

Sources: `analysis_v3/d6_alpha.py` (rule fitted on the synthetic model's optimal α only, `results_v3/b2/synthetic/`; applied per
layer with each layer's lag-16 autocorrelation from `results_v3/b2/stats/`; per-layer misses of the EMA α grid from B1),
`results_v3/d6/{dev,8b}/`. The rule uses the document's own trace statistic (no other document, no tuning), measured over the
whole decode (an online version would estimate it from the past).

## Best α against the budget
| B/K | 1.25 | 1.5 | 2 | 4 |
|---|---|---|---|---|
| 0.6B dev, LODO α | 0.4 | 0.5 | 0.7 | 0.9 |
| 8B dev, LODO α | 0.5 | 0.6 | 0.7 | 0.95 |
| synthetic (fitted), best α | 0.0 | 0.0–0.1 | 0.5 | 0.5–0.8 |
| G of α = 0.99 on 0.6B dev (S4) | < 0 | < 0 (z: −0.39) | | |

Mechanism (B3 / B2): the EMA's errors are pages just below the cut that return within 16 steps; at B/K 1.25–1.5 the candidates are
almost only such pages, and an EMA with long memory keeps their old (high) values while their current score has already dropped
(or the reverse), the same staleness that makes selected-only updates fail (B7). At B/K 4 most candidates are long-unselected
pages whose return depends on the slow drift of their score, which a long memory estimates better. The synthetic model reproduces
the direction (best α rising with B/K, collapse at α → 1) but puts the optimum too low at small budgets.

## The rule and the comparison (G; ΔG against one LODO α per budget)
Rule: α = clip(−0.143 + 0.516 · (1 − ac₁₆) + 0.341 · log₂(B/K), 0, 0.95), rounded to the grid (fitted on synthetic optima).

| Method | 0.6B B=80 | 0.6B B=96 | 0.6B B=128 | 0.6B B=256 | 8B B=96 | 8B B=128 | 8B B=256 |
|---|---|---|---|---|---|---|---|
| one α per budget (LODO) | 0.274 | 0.279 | 0.267 | 0.236 | 0.272 | 0.256 | 0.232 |
| per-layer α (LODO) | +0.002 | +0.003 | +0.004 | +0.003 | +0.004 | +0.007 | +0.005 |
| **rule from the synthetic model** | −0.024 | **−0.024** | −0.023 | −0.048 | **−0.035** | −0.047 | −0.082 |
| fixed α = 0.5 | −0.001 | 0.000 | −0.006 | −0.070 | 0.000 | −0.018 | −0.109 |

All rule differences have CIs below 0 (`summary.csv`). The rule fails because the model under-weights history at small budgets,
not because per-layer α is useless (the per-layer LODO optimum is only slightly better than one α). A fixed α = 0.5 is as good as
LODO at B ≤ 96 and loses at B ≥ 128; the practical recommendation is "α ≈ 0.5 at B/K ≤ 1.5, rising to 0.95 at B/K = 4".
