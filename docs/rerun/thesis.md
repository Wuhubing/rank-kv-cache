# Thesis after replication-v2 (D4, D5)

Basis: section 1.2 of `docs/workshop/rerun_plan.md`, `splits/heldout_frozen_v1.json`, `docs/efficient_policy_dev.md`.
This file is updated as results arrive; the final version is checked point by point in S8 and in the PR description.

## Original wording (plan §1.2)
> In "sparse attention + KV offload + GPU page cache", the temporal continuity of selection scores determines replacement quality.
> 1. An EMA over the score history of all pages steadily narrows the LRU–Bélády gap; per-layer budgets reduce misses further.
> 2. On common hardware these miss and byte savings do not necessarily become latency savings; we characterise when they do.

## Point 1: score magnitude or global ordering? → **global ordering** (S4.2)
Evidence (dev, 14 documents, B=96): an EMA of each page's percentile rank among **all** eligible pages reaches G 0.271
[0.266, 0.276], 97 % of the z-score EMA (0.279 [0.274, 0.283]). An EMA of the rank among the *selected* pages only (unselected =
0) reaches 0.017. Magnitude adds a small, significant ΔG of 0.0075 [0.0064, 0.0088]. Supporting evidence: updating only
selected pages makes the EMA far worse than LRU (−1.63), and InfLLM's selected-only rules stay near LRU even with the true
attention mass. The information sits in how the *unselected* residents are ordered against each other over time
(S4.7: stratified AUC 0.635 beyond recency and frequency).

**Point 1, revised:** *Keeping a decayed history of every resident page's score ordering, not only of the pages selected
now, narrows the LRU–Bélády gap by about 28 % (0.6B, two GPUs, held-out confirmed); per-layer budgets add to it (16 % fewer
repeat misses than LRU at equal total slots; against the same-allocation optimum about 35 % of the gap).*

Qualifications that the paper must carry:
- The gain is modest in capacity terms: it equals 5–7 % more LRU slots (S4.5); capacity matters more than the policy.
- It needs an accurate score: noisy scores make the policy worse than random, not equal to LRU (S4.8).
- "New policy is best" is not claimed: a power-law age kernel (0.310) and a learned linear model (0.300) do better on dev,
  and the EMA is one point on a continuum from current score (ArkVale) to frequency (S4.6).

## Point 2: systems → pending S7
## Generalisation (8B, 64K, SCBench, RULER) → pending S6
