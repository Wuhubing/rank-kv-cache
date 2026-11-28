# C5: G against the policy's per-token cost

**Conclusion: the policy's own work is never the obstacle, and the cheapest way to keep 90 % of the EMA's G depends on the
budget.** At B = 80–96 the cheapest variant that keeps ≥ 90 % refreshes the selected pages plus the 16 lowest-scoring
unselected residents (240 operations per token and layer instead of 288 at B=96, 91 % of the G). At B=128 no cheaper variant
reaches 90 % (selected + 32 lowest: 89 %; lazy every 2 steps: 84 %). The current
score alone, with no per-page state, keeps 86 % at B=96 but only 49 % at B=256. At B=256 lazy refresh every 2 steps keeps 95 %
for half the work, while the bottom-M rule stops working (58 %: the eviction candidates are then not only the lowest-scoring
residents). Lazy refresh at small budgets is harmful (r=4: 26 % at B=96). All of these costs are 10⁻⁵ of the model's arithmetic
(S7, `results_v2/gpu/flops.csv`); the better kernels cost more operations (normalised power law: 12× the EMA) and still nothing
measurable. What matters for the system is fusion and synchronisation (part E), not the operation count.

Source: `analysis_v3/c5_cost.py` → `results_v3/c5/cost.csv`, using the LODO results of replication-v2 S4 (current score, lazy
refresh: `results_v2/dev/s4/baselines/summary.csv`), B7 (scope, percentile; `results_v3/b7/b7_dev/`) and B1 (kernels;
`results_v3/b1/b1_dev/`), 14 0.6B dev documents. Operation count: 3 per refreshed resident value (z of the page, one
multiply-add), as in `results_v2/gpu/flops.csv`; the victim selection (a scan or partial sort of the B residents) is common to
all policies and not counted, except that the bottom-M rule needs one extra partial selection (B comparisons) and the percentile
signal needs a sort of all P ≈ 1 400 eligible pages.

| Policy | ops/token/layer at B=96 | G (share of full EMA) B=80 | **B=96** | B=128 | B=256 |
|---|---|---|---|---|---|
| Current score (no state) | 0 | 0.246 (90 %) | 0.240 (86 %) | 0.216 (81 %) | 0.116 (49 %) |
| EMA, all residents every step | 288 | 0.274 | **0.279** | 0.267 | 0.236 |
| EMA, lazy every 2 / 4 / 8 steps | 144 / 72 / 36 | 66 / 22 / −30 % | 71 / 26 / −35 % | 84 / 51 / −4 % | 95 / 86 / 62 % |
| EMA, selected pages only | 192 | −0.912 | −1.627 | −2.482 | −3.097 |
| EMA, selected + 4 / 8 / 16 / 32 lowest residents | 204 / 216 / 240 / 288 | 61 / 86 / 97 / 100 % | 53 / 82 / **91** / 97 % | 55 / 79 / 85 / 89 % | 57 / 60 / 58 / 54 % |
| EMA of the all-page percentile | ≈ 14 900 | 98 % | 97 % | 97 % | 94 % |
| Two-scale normalised EMA | 576 | 107 % | 108 % | 110 % | 104 % |
| Normalised power-law kernel (8 terms) | 3 360 | 111 % | 115 % | 125 % | 126 % |

(ops at other budgets scale with B; the bottom-M rule always refreshes the ~K selected residents plus M.)

Reading:
- Refreshing only *some* residents is safe only if the refreshed ones are the eviction candidates (B7); refreshing on a schedule
  (lazy) loses the candidates' current scores in between and is only acceptable when the budget is large and evictions are rare.
- The three policies that beat the EMA (two-scale, normalised exponential, normalised power law) cost 2–12× its operations,
  i.e. still ≤ 10⁻⁴ of a 0.6B model's per-token work.
- 8B is not computed separately: the costs are per layer and the same formulas apply (36 layers instead of 28).
