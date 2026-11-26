# C2: a learned linear policy with the new features

**Conclusion: a linear next-use model on 13 features, including the B1 features, does not beat the single normalised power-law
kernel; it is slightly worse.** LODO with DAgger at B=96: 0.313 [0.308, 0.318] on 0.6B dev and 0.308 [0.303, 0.312] on 8B dev,
against 0.320 / 0.319 for the normalised power law (paired ΔG −0.007 [−0.010, −0.004] and −0.011 [−0.014, −0.008]; B=128: −0.007
and −0.005). The replication-v2 linear model on 6 features had 0.300. The weights spread over many correlated score features,
so they do not reduce to C1's closed form; the one feature the closed form lacks — recency of the last selection — gets the
largest weight, and the weights on frequency and residency go the *opposite* way of a residency penalty (long-resident, often
selected pages are kept).

Source: `analysis_v3/learn.py` (features in its docstring; behaviour cloning on LRU's decisions, then two DAgger iterations;
ridge regression of log(1 + next-use distance); leave-one-document-out), `analysis_v3/learn_summary.py`; results
`results_v3/learn/linear_{dev,8b}/` (`learned.csv` per document, `models.json` per-fold coefficients, `summary.json`). Tests
`tests/test_learn_v3.py`. Run on wulab2.

| | 0.6B B=96 | 0.6B B=128 | 8B B=96 | 8B B=128 |
|---|---|---|---|---|
| EMA (LODO) | 0.279 | 0.267 | 0.272 | 0.256 |
| linear, behaviour cloning | 0.306 | 0.318 | 0.301 | 0.306 |
| **linear, DAgger** | **0.313 [0.308, 0.318]** | 0.326 | **0.308 [0.303, 0.312]** | 0.314 |
| normalised power law (C1) | 0.320 | 0.333 | 0.319 | 0.319 |
| DAgger − power law, ΔG [CI] (documents with fewer misses) | −0.007 [−0.010, −0.004] (2/14) | −0.007 [−0.011, −0.002] (4/14) | −0.011 [−0.014, −0.008] (1/14) | −0.005 [−0.010, 0.000] (5/14) |

Standardised coefficients at B=96 (mean over the 14 folds; the target is log next-use distance, so a positive weight means
"evict sooner"):

| feature | 0.6B | 8B | | feature | 0.6B | 8B |
|---|---|---|---|---|---|---|
| age (log steps since last selected) | **+0.311** | **+0.327** | | log selection count | −0.161 | −0.247 |
| log residency | −0.189 | −0.139 | | last rank / K | +0.133 | +0.120 |
| EMA α 0.9 of z | −0.147 | −0.140 | | normalised power law of z | −0.122 | −0.092 |
| trend (z − EMA) | −0.110 | −0.100 | | percentile | −0.099 | −0.025 |
| current z | −0.083 | −0.126 | | deficit | −0.051 | −0.044 |
| EMA α 0.5 of z | −0.035 | −0.098 | | neighbours' max z | +0.026 | +0.055 |
| log unselected steps | −0.006 | +0.042 | | | | |

Reading:
- The score features split their weight (correlation between them is high), so the linear model is a noisy average of several
  smoothings of z plus recency; with a squared-error target on next-use distance it is not optimised for the ranking that
  eviction needs, which is probably why one well-chosen kernel beats it.
- Residency and frequency get negative weights (keep), the recency of the last selection a large positive one (evict): the data
  favour "stale = evict" only in the sense of *time since last selected*, matching the small benefit of that term at large
  budgets in B1.
- The learned upper bound with trees and interactions is in B3.
