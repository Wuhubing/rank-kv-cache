# S4.7 Prediction quality: why does the score history help? (section 5.2, dev)

**Conclusion.** The EMA hardly predicts *which pages will be selected next*: its predicted top-K overlaps the next step's
top-K by 84.8 % (α=0.3), against 84.2 % for "the current top-K again". What it does predict is **which unselected
pages come back soon**. Among LRU's eviction candidates at B=96 (8.65 M decisions), adding the score to recency and
frequency raises the leave-one-document-out AUC for "re-selected within 32 steps" from 0.707 to 0.733. Inside strata of equal
recency (age decile) and frequency (selection-count quartile), the EMA of the score alone still has AUC 0.635 (the current score
0.623). The score carries information that recency and frequency do not, and averaging it over time sharpens it a little.

Source: `analysis_v2/s4_dev.py predict` → `results_v2/dev/prediction/` (`overlap_by_layer.csv`, `incremental_auc.csv`).

## Selection dynamics (mean over 14 documents × 28 layers)
| Metric | Value |
|---|---|
| Adjacent-step top-K overlap \|S_t ∩ S_t+1\|/K | 0.842 |
| Top-K churn 1 − Jaccard(S_t, S_t+1) | 0.264 |
| EMA-predicted top-K vs next top-K (fixed set size K), α = 0 / 0.3 / 0.5 / 0.7 / 0.9 | 0.842 / 0.848 / 0.847 / 0.839 / 0.811 |

PRR reports 98 % "recall" with a predicted set *larger* than K, so it cannot be compared with our fixed-size overlap. The
84 % persistence is why LRU is already a strong baseline. The policy question is only about the ~16 % of pages that
change.

## Incremental predictive power (LRU's candidates at B=96; label: re-selected within H=32 steps; base rate 0.773)
| Features (logistic regression, LODO over documents) | AUC (mean ± sd over folds) |
|---|---|
| recency (log age) + frequency (log selection count) | 0.707 ± 0.007 |
| + current z | 0.733 ± 0.006 |
| + EMA of z | 0.732 ± 0.006 |
| + both | 0.734 ± 0.006 |

| Stratified (10 age deciles × 4 count quartiles; size-weighted mean of within-stratum AUC) | AUC |
|---|---|
| EMA of z | 0.635 |
| current z | 0.623 |

The Spearman ρ between Quest scores and the true attention mass needs the attention read-out and is reported in S6
(0.6B dev: mean over layers 0.725, layer range 0.53–0.81).
