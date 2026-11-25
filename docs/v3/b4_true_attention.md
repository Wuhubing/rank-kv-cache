# B4: true attention as the eviction signal

**Conclusion: a more accurate importance estimate does not help; it hurts.** With the selection fixed (Quest picks the pages),
evicting by the pages' true dense attention mass instead of their Quest score turns the policy from G +0.24 (current score) /
+0.28 (EMA) into G −0.86 at B=96, worse than random eviction (−0.63), at every budget, on all 14 documents; no history weight
helps (LODO picks α = 0 in every fold). The replacement policy has to predict *which pages the selector will pick next*, and
the best predictor of Quest's next choice is Quest's own score: among the unselected pages it predicts re-selection within 32
steps with AUC 0.946, the true attention mass only with 0.815. Improving the selector's importance estimate is therefore a
separate question from improving the cache; for a given selector, the eviction signal should be that selector's score.

Data: 0.6B dev, 14 documents re-collected with `collect.long --record-attn-all` (dense attention mass of every completed page,
mean over query heads, float16 `[T, L, P]`; `data/dev_attn_all` on wulab2, 0.87 GB). The read-out does not change the selection:
ranked, scores and generated tokens are bit-identical to `data/dev` for 14/14 documents (`results_v3/provenance/b4_collect.log`;
manifests `results_v3/provenance/dev_attn_all_manifests.jsonl`). Collection: wulab2 card 0
(GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3), 21:16–21:25Z. The stored values reproduce the replication-v2 `true_rho` read-out
exactly (same Spearman per step and layer).

Replays: `analysis_v3.run b4` (policies read `attn_all` in place of `scores`; LRU and Bélády unchanged) and `b4_quest` (the same
configurations on the Quest scores of the same traces). Results `results_v3/b4/`; audit of LRU/Bélády with `replay.py`: 3 136
streams each, 0 mismatches. Read-out `analysis_v3/b4_signal.py` → `results_v3/b4/signal.csv`.

## G by budget (14 documents, document-bootstrap 95 % CI; wins = documents better than LRU)
| Signal the policy reads | B=80 | B=96 | B=128 | B=256 |
|---|---|---|---|---|
| Quest score, current | 0.246 | 0.240 [0.235, 0.246] 14/14 | 0.216 | 0.116 |
| Quest score, EMA of z (LODO α) | 0.274 | **0.279 [0.274, 0.283]** 14/14 | 0.267 | 0.236 |
| Quest score, EMA of all-page percentile (LODO α) | 0.268 | 0.271 | 0.259 | 0.223 |
| True attention, current | −0.677 | **−0.858 [−0.883, −0.829]** 0/14 | −0.784 | −0.404 |
| True attention, EMA of z (LODO α; picks α = 0) | −0.659 | −0.842 [−0.865, −0.816] 0/14 | −0.779 | −0.405 |
| True attention, EMA of percentile / raw (LODO α; pick α = 0) | −0.677 | −0.858 | −0.784 | −0.404 |
| (Random eviction, replication-v2, for scale) | | −0.633 | −0.820 | |

(With the attention signal, "z" uses the attention of Quest's top-1 and K-th pages, whose difference can be negative; that is why
its α = 0 row differs slightly from the current-attention row.)

## Why: the cache serves the selector
| Read-out (mean over 14 documents × 28 layers, every 4th step) | Value |
|---|---|
| Spearman ρ(Quest score, true attention), all eligible pages | 0.725 |
| Spearman ρ(Quest score, true attention), unselected pages only | 0.690 |
| AUC for "selected again within 32 steps" among unselected pages: Quest score | **0.946** |
| same: true attention mass | 0.815 |
| base rate (share of unselected pages selected again within 32 steps) | 0.067 |

A page comes back into the cache when *Quest* selects it, and Quest selects by its upper bound, not by attention. Pages whose
true attention is low but whose bound is high (loose bounds: wide key ranges) are exactly the ones an attention-ordered cache
evicts first and Quest then re-selects. Among the ~32 unselected residents at an eviction (recently selected pages) the
attention ordering is worse than no information, which is why the policy falls below random.

## What this answers
- *How much would a better importance estimate raise G?* For the replacement decision under a fixed selector: it would not;
  the "error" of Quest relative to true attention is not noise for the cache, because the cache's target is Quest's next
  choice. This places N4 (score noise, `docs/v3/d3_noise.md`): the harmful noise is noise between the score the policy reads and
  the score the *selector* uses (e.g. a cheaper or staler copy), not the selector's error against the truth.
- *Is improving the selector's estimate worth it for the cache?* Only if the selector itself changes; then the policy must follow
  the new selector's score. This is the same lesson as the ArkVale digests (`docs/rerun/s4/09_arkvale_full.md`: digests that
  disagree with the selector are worse eviction keys) and InfLLM's true-attention rules (S6 §3: slightly worse than LRU).
- Exploratory (dev only, one model). The 8B dev traces were not re-collected with the read-out.
