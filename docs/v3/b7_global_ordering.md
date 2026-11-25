# B7: the evidence for "global ordering", as curves

**Conclusion: what the policy needs is the current score of the residents near the eviction boundary, ranked against the pages
just below the selection cut.** Starting from "update only the selected pages" (G −1.63 at B=96) and additionally refreshing the
M unselected residents with the *lowest* current scores, G rises to 0.148 with M=4, 0.228 with M=8, 0.254 with M=16 and 0.271
with M=32, against 0.279 for refreshing all residents. Refreshing the M *highest*-scoring unselected residents instead does nothing
until M=32 (0.061). Likewise, a rank signal that only orders the selected pages (reference set = top K) gives 0.016, ordering the
top 2K pages gives 0.250, and the full ordering 0.268. The −1.63 is caused by stale values: a page that is not refreshed keeps the
value it had when it was last selected, i.e. a high one, so pages that have gone cold are never evicted
[staleness read-out: see "Why −1.63"]. Early rank-based policies and InfLLM's rules sit at the left end of these curves.

Source: `analysis_v3.run b7` (`kvpolicy_v3.ScopedEMA`, `RefRankEMA`; tests: M=0 equals `ScoreEMA(mode="selected")`, M=all
equals the full EMA, R=K equals `signal="rank"`), 14 0.6B dev documents, α tuned LODO over {0.3, 0.5, 0.7} for every point.
Results `results_v3/b7/b7_dev/`; LRU and Bélády re-checked with `replay.py`: 3 136 streams, 0 mismatches. [8B: see the end.]

## Update scope: selected pages + M unselected residents (EMA of z; G)
| M | B=80 bottom | B=80 top | **B=96 bottom** | B=96 top | B=128 bottom | B=128 top | B=256 bottom | B=256 top |
|---|---|---|---|---|---|---|---|---|
| 0 (selected only) | −0.912 | −0.912 | **−1.627** | −1.627 | −2.482 | −2.482 | −3.097 | −3.097 |
| 4 | 0.168 | −0.977 | **0.148** | −1.743 | 0.147 | −2.734 | 0.134 | −3.766 |
| 8 | 0.236 | −0.864 | **0.228** | −1.666 | 0.212 | −2.752 | 0.142 | −4.192 |
| 16 | 0.266 | −0.222 | **0.254** [0.251, 0.258] | −1.092 | 0.227 | −2.236 | 0.137 | −4.266 |
| 32 | 0.273 | 0.255 | **0.271** | 0.061 | 0.237 | −0.932 | 0.128 | −2.875 |
| all residents | 0.274 | 0.274 | **0.279** | 0.279 | 0.267 | 0.267 | 0.236 | 0.236 |

"bottom" = the M unselected residents with the lowest current Quest score; "top" = the highest. The all-page percentile signal
(`allrank`) gives the same picture (B=96 bottom: 0.146 / 0.223 / 0.249 / 0.264 for M = 4 / 8 / 16 / 32; top M=32: 0.048; all
0.271). At B=256 the bottom-M curve saturates at 0.14: with 192 unselected residents, the eviction candidates are no longer only the
lowest-scoring 32.

## Rank reference set: EMA of the rank among the R highest-scoring pages (0 outside)
| R | B=80 | B=96 | B=128 | B=256 |
|---|---|---|---|---|
| 64 = K (rank among the selected only) | 0.045 | 0.016 | 0.005 | −0.005 |
| 128 = 2K | 0.263 | 0.250 [0.245, 0.255] | 0.235 | 0.216 |
| 256 = 4K | 0.267 | 0.269 | 0.260 | 0.279 |
| 512 | 0.267 | 0.268 | 0.259 | 0.242 |
| all eligible pages | 0.267 | 0.268 | 0.258 | 0.184 |

## Why −1.63
A resident that is not refreshed keeps its last value. Under selected-only updates, the last refresh happened when the page was
selected, so its value reflects a moment when it was among the top K. A page that has gone cold therefore keeps a high value
indefinitely and is never chosen as a victim, while the pages just admitted (one sample, often lower) are evicted. The cache
fills with stale "former favourites"; misses rise far above LRU's. Refreshing only the lowest-scoring residents is enough to
correct exactly the pages that matter for the next eviction, which is why the bottom-M curve rises so fast and the top-M curve
does not. [Read-out pending: mean steps since the last update of the eviction candidates and the Spearman correlation between
their stored values and their current z, selected-only vs full, `results_v3/inst/dev/stale.csv`.]

## The same framework explains the earlier failures
- **Rank policy of stages 1–2 (`kvpolicy.RankFull`, G 0.027).** Its signal is the page's last rank *among the selected pages*
  plus a selection count over 8 steps: information about the selected set only, updated only when a page is selected. That is the
  left end of both curves (R = K: 0.016; M = 0 with a decay to a recency order instead of a frozen value), so it can only add a
  little to LRU's recency.
- **InfLLM's rules (G 0.01–0.03; slightly below LRU with the true attention mass).** Both rules give a page attention mass only
  when it is selected and otherwise only decay it. The decay makes every unselected page lose value at the same rate, so the
  order among unselected residents becomes the order of (last mass × decay^age): essentially a recency order weighted by an old
  mass. That avoids the −1.63 catastrophe (values decay instead of freezing) but carries no information about the unselected
  pages' *current* scores, so it lands near LRU. The bottom-M curve shows what is missing: the current score of the
  low-ranked residents.

## 8B dev
(pending: `results_v3/b7/b7_8b/`)
