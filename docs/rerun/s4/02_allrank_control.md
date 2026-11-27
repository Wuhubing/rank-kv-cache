# S4.2 All-page percentile rank control: magnitude or global ordering? (decides thesis point 1, D4)

**Conclusion: mostly the global ordering.** Replacing each page's score by its percentile rank among **all** eligible pages
(`allrank`, keeps the full ordering, drops magnitude) keeps 97 % of the score EMA's G measured from LRU: at B=96, G 0.271
[0.266, 0.276] vs 0.279 [0.274, 0.283] for the z-score. **Against the right baseline it keeps about 80 %** (corrected in
followup-v3, R3): at α=0 (no history) the three signals give the same eviction order and all equal the current score (G 0.240),
so the gain *from the history* is 0.038 for z and 0.031 for allrank (ratio 0.80 at B=96; 0.79 / 0.84 / 0.89 at B=80 / 128 /
256; `results_v2/dev/fixes/a3_history_gain.csv`). Of what remembering the scores adds, the ordering accounts for about 4/5 and
the magnitude for about 1/5. Keeping only the rank **among the selected pages** (every unselected page gets 0)
drops G to 0.017. So what matters is how unselected pages are ordered against each other over time. The score
magnitude adds a small, consistent extra on top of the ordering: paired ΔG (allrank − z) = −0.0075 [−0.0088, −0.0064], negative at every
budget.

Source: `validation/rank_control_dev.py` on the 14 new dev traces (`results_v2/dev/rank_control/`). Same update scope
(all residents), same 10-value α grid and same LODO α selection for all three signals. Its replay was cross-checked
against the matching `efficient_dev.py`/`checks.py` rows of the same traces (`--archive results_v2/dev/efficient_dev`).

| B (B/K) | z (score relative to K-th) | allrank (percentile among all pages) | rank (selected only) | allrank − z, paired ΔG [CI] |
|---|---|---|---|---|
| 72 (1.125) | 0.256 [0.254, 0.259] | 0.252 [0.249, 0.255] | 0.076 | −0.004 [−0.005, −0.003] |
| 80 (1.25) | 0.274 [0.271, 0.276] | 0.268 [0.265, 0.270] | 0.046 | −0.006 [−0.007, −0.005] |
| **96 (1.5)** | **0.279 [0.274, 0.283]** | **0.271 [0.266, 0.276]** | **0.017** | **−0.0075 [−0.0088, −0.0064]** |
| 128 (2) | 0.267 [0.258, 0.276] | 0.259 [0.248, 0.269] | 0.008 | −0.008 [−0.011, −0.005] |
| 256 (4) | 0.236 [0.213, 0.257] | 0.223 [0.201, 0.243] | 0.002 | −0.013 [−0.017, −0.009] |

All three signals beat LRU on 14/14 documents at B ≤ 128 (rank: 10/14 at B=256). LODO picked the same α for most folds
(z: 0.5 at B=96; allrank: 0.6).

## Thesis wording that follows (`docs/rerun/thesis.md`)
Point 1 becomes: *"Keeping an EMA of every resident page's score **ordering** narrows the LRU–Bélády gap. The relative
order of the pages that are not selected carries most of the signal: of the gain that the history adds over the current score,
about 4/5 survives when only the ordering is kept, and the magnitude adds about 1/5."* "Score magnitude" is not supported as
the main mechanism. (The replication-v2 wording "magnitude adds about 3 % of the gain" measured against LRU, where most of the
G is already reached by the current score with no history.)
