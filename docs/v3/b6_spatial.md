# B6: spatial structure — do returning pages sit next to pages in use?

**Conclusion: returning pages cluster next to pages in use, but the neighbours' scores add almost no predictive information, so
spatial structure is not a useful extra eviction signal.** 23 % of LRU's repeat misses are within ±1 page of a page selected at
the previous step (35 % within ±2), against 6–8 % (9–14 %) for a random unselected page: a 3–4× enrichment. But among LRU's
eviction candidates, the higher of the two neighbours' current scores predicts "selected again within 32 steps" with AUC 0.53
(8B 0.52), and in the learned models the neighbour feature gets a small weight (C2: +0.03 / +0.06 standardised; tree importance
low). The clustering is already captured by the page's own score: Quest's page bounds of adjacent pages are similar because the
text is.

Source: `analysis_v3/instrument.py` (LRU at B=96: page-id distance of each repeat miss to the previous step's selection; random
unselected eligible pages as the baseline; neighbour-score AUC on every 4th step's candidates, censored at the trace end),
`results_v3/inst/{dev,8b}/spatial.csv`, `summary.json`.

| | 0.6B dev | 8B dev |
|---|---|---|
| repeat misses within ±1 page of the previous selection | 23.2 % | 23.5 % |
| random unselected page within ±1 | 5.9 % | 8.2 % |
| repeat misses within ±2 | 35.4 % | 36.4 % |
| random unselected page within ±2 | 9.5 % | 14.0 % |
| AUC of the neighbours' max score for re-selection within 32 steps (LRU candidates) | 0.532 | 0.519 |

Not used as a C candidate (task: "if there is a signal"); the feature is part of the C2 / B3 learned models.
