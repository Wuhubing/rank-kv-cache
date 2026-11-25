# Held-out results (preregistered, executed once on 2026-11-23)

Protocol: `docs/efficient_policy_dev.md` → "Held-out preregistration" (approved as drafted).
Data: the 21 held-out documents of `splits/longctx_v1.json` (15 gov_report, 6 qmsum), captured once
with Qwen3, the dev prompt, the Quest selector (page 16, K=64), the dev decoding and stop rule.
Frozen parameters: `splits/heldout_frozen_v1.json` (commit f87e982, committed before any held-out
output was read). They come from the 14 dev documents only. Code: `heldout.py`.
Sanity check: the dev traces were recaptured on a fresh runtime and reproduce the dev report exactly
(LRU 6.03 %, score EMA G 0.279 at B=96).

## Preregistered success criteria (B=96)
| Criterion | Threshold | Result | Pass |
|---|---|---|---|
| (i) Score-EMA G, 95 % CI lower bound | > 0.15 | G = 0.281, CI [0.277, 0.285] | yes |
| (ii) Docs where EMA has fewer misses than LRU | ≥ 18 / 21 | 21 / 21 | yes |
| (iii) Allocation + EMA beats EMA alone (paired CI excludes 0) | CI > 0 | ΔG = 0.120, CI [0.113, 0.128] | yes |

Criterion (iii) is evaluated as preregistered, with G relative to uniform-budget anchors (see the
note under the budget curve); since both policies share those anchors at equal total slots, it is
equivalent to Allocation + EMA having fewer misses than EMA. Bootstrap: 2000 document-clustered replicates. Per-document EMA G is 0.26–0.30 (all 21 documents).

## Budget curve (pooled over 21 documents; G relative to uniform-budget LRU and Bélády)
| B (B/K) | LRU miss % | Bélády % | EMA G | Alloc+EMA G† | InfLLM-style dsum G | Current score G | Decayed count G | Linear G |
|---|---|---|---|---|---|---|---|---|
| 72 (1.125) | 11.50 | 8.81 | 0.257 | 0.257* | 0.239 | 0.229 | 0.110 | – |
| 80 (1.25) | 8.88 | 6.19 | 0.276 | 0.307 | 0.257 | 0.249 | 0.069 | – |
| **96 (1.5)** | **5.98** | **3.85** | **0.281** | **0.402** | 0.270 | 0.246 | 0.044 | 0.303 |
| 128 (2) | 3.56 | 2.19 | 0.271 | 0.528 | 0.266 | 0.224 | 0.052 | 0.314 |
| 256 (4) | 1.38 | 0.98 | 0.242 | 0.242* | 0.216 | 0.145 | 0.011 | 0.336 |

\* The frozen allocation at B=72 and B=256 is uniform (grid limits, as on dev), so it equals EMA.
† G = (LRU − policy) / (LRU − Bélády) with LRU and Bélády at B slots in every layer. For the
uniform-budget policies this is the share of the LRU–Bélády gap closed. Alloc+EMA uses the same total
slots (28·B) but different per-layer budgets, and uniform Bélády is not the optimum for that
allocation, so its G is **not** a fraction of the gap; read it as a miss reduction on the same scale.
An allocation-aware oracle (Bélády at the frozen per-layer budgets) is needed before claiming a gap
fraction: `python validation/alloc_oracle.py heldout HELDOUT_TRACES --out DIR` (pending; needs the
raw traces).
All 95 % CIs are within ±0.015 of the point estimates. Every non-reference policy beats LRU on all
21 documents at every budget, except decayed count at B=256 (20/21).

Repeat-miss drop vs LRU at B=96 and equal total slots (the headline for Alloc+EMA): EMA 11.6 %,
Alloc+EMA 16.6 %, Bélády 41.3 % (`recompute_heldout.py` prints the first two with CIs).
Per task at B=96: gov_report EMA 0.281 / Alloc+EMA 0.398; qmsum EMA 0.281 / Alloc+EMA 0.412.

## Dev → held-out
| B=96 | Dev (14) | Held-out (21) |
|---|---|---|
| EMA G | 0.279 | 0.281 |
| Alloc+EMA G (uniform-budget anchors) | 0.409 | 0.402 |
| InfLLM-style dsum G | 0.269 | 0.270 |
| Current score G | 0.244 | 0.246 |
| Linear (DAgger) G | 0.300 | 0.303 |

The dev results transfer almost exactly, including the ranking of the policies.

## Not claimed
Latency or throughput (needs the GPU benchmark); other models, prompts or selectors (the D-checks
on dev only cover Qwen3-1.7B and one second prompt); token-level granularity.

## Notes
- Held-out traces (21 npz) are backed up to Drive `MyDrive/rank-kv-cache/longctx_v1_heldout/`
  (md5-verified), and `misses.csv`, `doc_level.csv`, `frozen.json` to `MyDrive/rank-kv-cache/results_heldout/`.
