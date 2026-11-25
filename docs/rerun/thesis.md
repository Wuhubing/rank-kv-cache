# Thesis after replication-v2 (D4, D5)

Basis: section 1.2 of `docs/workshop/rerun_plan.md`, `splits/heldout_frozen_v1.json`, `docs/efficient_policy_dev.md`. Every
statement below cites the step that supports it. Confirmatory = preregistered held-out evaluated once; exploratory = dev only.

## Original wording (plan §1.2)
> In "sparse attention + KV offload + GPU page cache", the temporal continuity of selection scores determines replacement quality.
> 1. An EMA over the score history of all pages steadily narrows the LRU–Bélády gap; per-layer budgets reduce misses further.
> 2. On common hardware these miss and byte savings do not necessarily become latency savings; we characterise when they do.

## Revised thesis
> In a sparse-attention KV cache with host offload, **the ordering of the unselected pages' current selector scores, remembered
> over time, is a reliable replacement signal.**
> 1. A per-page EMA of that ordering, updated for every resident page, closes about **27–28 % of the LRU–Bélády gap** on two
>    models (Qwen3-0.6B, Qwen3-8B) and two GPUs. Per-layer budgets add to it: at equal total slots, 16 % (0.6B) / 21 % (8B) fewer
>    repeat misses than LRU, and 35 % / 43 % of the gap at the same allocation.
> 2. On a current PCIe GPU these savings (8–14 % fewer host→GPU bytes) **do not become latency** at batch 1, and become throughput
>    only marginally and only at large batch on a small model (+3–4 % at batch 32 on 0.6B, none on 8B), because transfer is a small
>    part of the decode step. We characterise this.

## Point 1: supported, with sharpened wording
| Claim | Status | Evidence |
|---|---|---|
| EMA narrows the gap (0.6B) | **Supported, confirmatory ×2** | L4 held-out G 0.281; replication on RTX 6000 Ada G 0.279 [0.275, 0.282], 21/21 (S5) |
| EMA narrows the gap on a second model | **Supported, confirmatory** | Qwen3-8B, new preregistered held-out: G 0.272 [0.269, 0.276], 21/21 (S6 §1) |
| "Steadily" | **Supported** | Every document of every setting beats LRU: 0.6B dev/held-out, 8B dev/held-out, 64K (YaRN), RULER, SCBench multi-turn, page 8–64, K 32–128, sampled decoding (S4, S6) |
| The signal is **score magnitude** | **Not supported; replaced** | All-page percentile rank keeps 97 % of the gain (0.271 vs 0.279); magnitude adds 0.0075 [0.0064, 0.0088] (S4.2). On 8B dev: 0.261 vs 0.271 (S6) |
| The signal is the **global ordering of unselected pages** | **Supported** | rank-among-selected only: 0.017; selected-only updates: −1.63; InfLLM's selected-only rules ≈ LRU, slightly worse with the true attention mass (S4.1, S4.2, S6 §3); stratified AUC 0.635 beyond recency and frequency (S4.7) |
| Per-layer budgets reduce misses further | **Supported, now against the right anchor** | Additive on dev (0.279 + 0.124 = 0.402); same-allocation anchors: 35 % (0.6B) / 43 % (8B) of the gap; within the allocation still 28 % / 27 % (S4.3, S5, S6 §1) |
| "EMA is a new, best policy" | **Not claimed** | A power-law age kernel (0.310) and a learned linear model (0.300) do better on dev (S4.4); the EMA is one point on a curve from current score (ArkVale, α=0) to frequency (α→1) (S4.6); InfLLM's decay rule over all residents (dsum, 0.266) is close |
| Size of the effect | **Qualified** | Worth 5–7 % more LRU slots (iso-miss capacity); Bélády is worth 18–46 % (S4.5). Capacity matters more than the policy |
| Robustness to score noise | **Qualified (new limitation)** | Noisy scores push the policy below random, not back to LRU (S4.8). Quest scores have ρ ≈ 0.73–0.75 with true attention (S6 §3) |

## Point 2: supported as a characterisation
| Claim | Status | Evidence |
|---|---|---|
| Byte savings exist on real hardware | **Supported** | EMA −8 to −11 % bytes, allocation + EMA −11 to −20 %, Bélády −18 to −40 %; LRU/Bélády measured misses equal the simulator exactly (S7) |
| They do not become latency at batch 1 | **Supported** | 0.6B: all tiered policies 30.1–31.0 ms/token across B = 80–256; 8B: 40.6 ms; ties within the repeat spread (S7 §1, §3) |
| They barely become throughput with batching | **Supported with a nuance (new measurement)** | Batch 1–24: EMA 0.90–1.00× LRU throughput in an unfused batched implementation, Bélády +0–7 %. Batch 32: 0.6B EMA +3.4 %, alloc+EMA +4.1 %, Bélády +4.8 % (reproducible); 8B within ±2.5 % noise (S7 §2, §3) |
| "Batch 32 would save 2.3 ms/step" | **Removed** | Was a linear extrapolation; replaced by measurement (`docs/gpu_benchmark.md`, STATUS_REPORT) |
| When it would pay off | **Stated as conditions, not measured** | Slower/contended host link, larger KV per page (MHA), more pages per step, fully fused offload path (S7 §4) |

## Limitations that the paper must carry
- One selector family (Quest upper bound, page-level, K=64 by default), one model family (Qwen3), one GPU type in this replication
  (plus L4 in the original).
- 8B documents are partly 12–16K (the 16–32K pool was exhausted); the 8B decodes all hit the 1024-token cap.
- 64K uses YaRN on a 0.6B model, and 6 of 14 documents stopped on the loop rule; RULER and SCBench are dev-only and exploratory.
- Batched systems numbers come from an unfused implementation (lower bound on what an optimised system would see).
