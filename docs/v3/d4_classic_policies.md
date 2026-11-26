# D4 (part of P6): why classic cache policies fail on this workload

**Conclusion: not because frequency saturates — it does not — but because the classic policies use recency and frequency in the
ways that suit block-cache workloads (filter one-hit wonders, protect pages hit repeatedly), while here the information that
separates the eviction candidates is the current score ordering just below the selection cut, which they never see.** The
hypothesis in the task ("with K pages per step and 84 % overlap, frequency saturates and cannot separate the unselected pages") is
not supported: among LRU's eviction candidates the log selection count varies widely (mean variance 1.6, mean range 4.2 in log
units; all candidates tie in 0.00 % of decisions), and on its own frequency predicts re-selection within 32 steps slightly *better*
than recency (AUC 0.652 vs 0.634; together 0.707). What the classic policies add on top of LRU goes the wrong way for this
workload, and the score adds what they lack (+0.027 AUC beyond recency + frequency; stratified AUC 0.638).

Sources: `analysis_v3/instrument.py` (`results_v3/inst/{dev,8b}/freq.csv`, LRU candidates at B=96); masked AUCs of followup-v3
A7 (`results_v2/dev/prediction_masked/incremental_auc.csv`); classic policies from replication-v2 S4 and the S3-FIFO fix (A5);
use-once evictions from `results_v2/dev/efficient_dev/bypass.csv`.

| Read-out (0.6B dev, B=96) | Value |
|---|---|
| LRU candidates: mean variance of log(1 + in-cache selection count) per decision | 1.59 (8B 1.79) |
| mean range of that count within a decision (log units) | 4.18 (8B 4.48) |
| decisions in which all candidates have the same count | 0.00 % (8B 0.007 %) |
| AUC for "re-selected within 32 steps", recency alone / frequency alone / both (LODO logistic, last 32 steps masked) | 0.634 / 0.652 / 0.707 |
| + current score / + its EMA | 0.734 / 0.734 |
| stratified AUC of the EMA within (recency decile × frequency quartile) | 0.638 |
| Bélády's evictions that remove a page admitted on the previous step and never hit (use-once) / LRU's | 18 % / 0.4 % |

Classic policies against LRU at B=96 (G): SIEVE −0.019, S3-FIFO −0.095 (Algorithm 1), ARC ≤ 0, LFU ≤ 0, LRU-2 ≤ 0, FIFO < 0,
Random −0.633.

Why each loses:
- **S3-FIFO and SIEVE** treat a newly inserted object as a probable one-hit wonder (S3-FIFO evicts it from the small queue unless it
  is hit again soon; SIEVE evicts unvisited objects first). Here a newly admitted page was just *selected* and, with 84 %
  adjacent-step overlap, is likely to be selected again; demoting it is the wrong default. Bélády does evict use-once pages
  (18 % of its evictions), but only the specific ones that will not return, which a recency/frequency filter cannot identify.
- **LFU, LRU-2, ARC's frequency list** protect pages selected many times. The in-cache count mostly measures how long a page
  has been resident and popular; it carries information (AUC 0.65) but in the wrong place for eviction: the decisive candidates
  are the pages just below the selection cut, whose *current* rank decides whether they come back within a few steps (B3:
  71 % of the EMA's wrong evictions come back within 16 steps; B7: refreshing the lowest-scoring residents' scores is what helps).
- **The score-based EMA** reads exactly that current ordering; its gain over LRU is the information the classic policies lack,
  not a better use of recency or frequency.

Conditions: the classic policies' assumptions (one-hit wonders, skewed long-term popularity) would hold for a selector with low
step-to-step overlap or for accesses without a per-step score; under a Quest-style selector with ≥ 80 % overlap they do not.
