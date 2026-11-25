# S6: second model, longer context, more tasks, selector sweep, sampling, true attention

**Summary.** The score-EMA result generalises. It is **confirmed on Qwen3-8B with a new preregistered held-out set**:
G 0.272 [0.269, 0.276], 21/21 documents, allocation + EMA 21 % fewer repeat misses than LRU, and 27 % of the gap at the same
allocation. In every exploratory setting (0.6B at 64K with YaRN, RULER, SCBench multi-turn, page sizes 8–64, K 32–128,
sampled decoding) the EMA beats LRU on every document, with G of about 0.23–0.31 at B/K = 1.5. InfLLM's
attention-mass rules do not improve with the *true* attention mass; they become slightly worse than LRU everywhere. Quest's
upper-bound scores have a Spearman ρ of about 0.73–0.75 with the true attention mass.

All settings use the same selector code (`collect/long.py`), card 0, and record `attn_mass` and `true_rho` (read-outs that do not
change the selection: on the 14 0.6B dev documents, the traces with read-outs are bit-identical to the S3 traces).

## 1. Qwen3-8B, confirmatory (preregistered, held-out evaluated once)
**Timing.** Preregistration `docs/preregistration_8b.md` + frozen file `splits/heldout_frozen_8b.json` committed and pushed
at **2026-11-25T08:35:29Z** (commit `328059e`, confirmed on GitHub with `git ls-remote`). The first 8B held-out trace started at
**2026-11-25T08:35:54Z** (manifest `started_at`); the last finished at 10:06:50Z. The evaluation ran once, 10:07–10:11Z.

Setting: `splits/longctx_8b_v1.json`. 14 dev + 21 held-out GovReport/QMSum documents that are not in longctx_v1. The 16–32K pool
had only 25 unused documents (all from LongBench's `gov_report_e`), so the floor was relaxed to 12K as the plan prescribes:
25 documents at 16–32K and 10 at 12–16K. Held-out: 20 GovReport, 1 QMSum. Every 8B document generated the full 1024 tokens
(stop = length; 8B thinks longer than 0.6B).

Frozen on 8B dev (`analysis_v2/freeze_8b.py`): α = 0.4/0.5/**0.6**/0.7/0.95 for B = 72/80/96/128/256 (every LODO fold picked the
same α); allocation from the 8B dev EMA curves (non-uniform at B = 80–128); DAgger linear coefficients.

| # | Criterion (B = 96) | Threshold | Result | Pass |
|---|---|---|---|---|
| (i) | EMA G, CI lower bound | > 0.15 | **0.272 [0.269, 0.276]** | yes |
| (ii) | Documents with EMA < LRU | ≥ 18/21 | **21/21** | yes |
| (iii) | Alloc+EMA − EMA, ΔG CI | > 0 | **0.252 [0.243, 0.260]** | yes |
| (iv) | Alloc+EMA vs LRU within the same allocation, G CI lower bound | > 0.15 | **0.273 [0.268, 0.277]** | yes |

**The 8B claim is confirmed.** Sources: `results_v2/ext/8b_heldout/` (`criteria.txt`, `eval/misses.csv`, `alloc_oracle/`,
`audit.txt`: 7 560 streams, 0 mismatches).

Budget curve (held-out, G against uniform anchors):
| B | LRU % | Bélády % | EMA | Alloc+EMA | dsum | score | dcount | linear |
|---|---|---|---|---|---|---|---|---|
| 72 | 9.40 | 7.24 | 0.259 | 0.263 | 0.226 | 0.227 | 0.068 | – |
| 80 | 7.09 | 5.01 | 0.270 | 0.373 | 0.250 | 0.234 | 0.031 | – |
| **96** | 4.69 | 3.05 | **0.272** | 0.524 | 0.271 | 0.222 | 0.033 | 0.296 |
| 128 | 2.70 | 1.66 | 0.261 | 0.653 | 0.260 | 0.182 | 0.044 | 0.305 |
| 256 | 0.98 | 0.71 | 0.234 | 0.234 | 0.183 | 0.066 | 0.005 | 0.320 |

Allocation with same-allocation anchors (B=96): G against the same-allocation Bélády 0.426 [0.420, 0.431]; within the
allocation 0.273; repeat-miss drop vs uniform LRU 21.0 % (EMA alone 10.9 %). On 8B, allocation contributes more than on 0.6B
(ΔG 0.25 vs 0.12): the 36 layers differ more. Per document: EMA G 0.260–0.291 (median 0.272); qmsum (1 document) 0.260.

Note: `recompute_heldout.py` prints the label "28*96 slots", hard-coded for 0.6B; for 8B it is 36·96. The numbers are correct.

## 2. Exploratory settings (dev only, 14 documents each, LODO-tuned families; `analysis_v2/explore.py`)
G at B/K = 1.5 (B=96) with document-bootstrap CI; wins = documents with fewer misses than LRU.

| Setting | LRU miss % | Bélády % | current score | **EMA (LODO)** | all-page rank EMA | InfLLM Eq. 1 (proxy) | InfLLM Eq. 1 (true mass) | InfLLM `lru-s` (true, LODO) |
|---|---|---|---|---|---|---|---|---|
| 0.6B, 16–32K (S3 traces, read-outs) | 6.04 | 3.87 | 0.240 | **0.279** [0.274, 0.283] 14/14 | 0.270 | 0.016 | −0.044 | −0.048 |
| 8B dev, 12–32K | 4.58 | 2.98 | 0.221 | **0.271** [0.267, 0.275] 14/14 | 0.261 | 0.009 | −0.058 | −0.063 |
| 8B held-out (fixed configs only) | 4.69 | 3.05 | (0.222) | (0.272, preregistered) | – | 0.010 | −0.058 | – |
| 0.6B, 32–64K, YaRN ×2 | 7.24 | 4.60 | 0.257 | **0.284** [0.281, 0.289] 14/14 | 0.279 | – | −0.041 | −0.047 |
| 0.6B, RULER 16K (4 tasks) | 4.42 | 2.74 | 0.207 | **0.255** [0.233, 0.292] 14/14 | 0.244 | – | −0.030 | −0.035 |
| 0.6B, SCBench 5-turn sessions | 4.24 | 2.68 | 0.247 | **0.285** [0.283, 0.287] 14/14 | 0.275 | – | −0.031 | −0.035 |

Notes:
- **64K/YaRN** (`splits/longctx64k_v1.json`): a different model configuration, reported separately. 7 NarrativeQA, 4 Multi-News,
  3 GovReport documents (2 middle-truncated to 62K tokens to reach 14). 6 of 14 stopped on the 32-gram loop rule, 5 at
  1024 tokens, 3 at EOS. At B=256 EMA 0.260, current score 0.161.
- **RULER** (`splits/ruler16k_v1.json`): cwe, niah_multivalue, niah_multiquery, niah_multikey_3. With thinking on, every document has
  ≥ 172 decode steps (none below the 100-step exclusion rule); 6 of 14 stopped on the loop rule. The CIs are wider (shorter
  traces); at B=256 EMA 0.313, current score 0.105.
- **SCBench** (`splits/scbench_v1.json`): 7 En.Sum + 7 En.QA sessions, 5 turns each, context middle-truncated to 24K tokens,
  ≤ 512 new tokens per turn; 768–1701 decode steps per session (12 EOS, 1 loop, 1 length on the last turn). The format and
  the replay were first checked on one session (`scbench_summary/9`): page ids continue across turns, and keeping the cache across turns gives
  fewer LRU misses than a cold cache per turn (78 272 vs 85 831 at B=96). Turns 2–5 often end with `<|endoftext|>` after
  thinking-style text (previous turns' thinking stays in the KV), which is acceptable for cache behaviour. At B=256 the current score falls to
  LRU (−0.019, 6/14) while the EMA keeps 0.192 (14/14): with turn boundaries, the score history matters more.

## 3. True attention mass and the two InfLLM rules (S6.7)
With the true post-softmax attention mass of each selected page (summed over heads; `--record-attn-mass`), both InfLLM rules,
paper Eq. 1 (s ← d·s + mass) and code `lru-s` (selected: reset to this step's mass), are **slightly worse than LRU** at every
budget and in every setting (table above; 0/14 or 0/21 documents better at B=96). With the softmax proxy they were just above LRU.
Neither rule updates unselected pages with information about their *current* relevance (only decay), so their order of
unselected residents reflects past attention mass, which turns out to be less predictive than recency. This matches S4.2:
the useful signal is the current score of the *unselected* pages, which InfLLM's rules never read.

Spearman ρ between the Quest page scores and the true dense attention mass of every complete page (mean over query heads;
`--record-true-attn-rank`; mean over steps, then over layers; `analysis_v2/rho_summary.py`):
| Setting | mean ρ over layers | weakest layer | strongest layer |
|---|---|---|---|
| 0.6B 16–32K | 0.725 | 0.527 (layer 6) | 0.805 (layer 8) |
| 0.6B 64K/YaRN | 0.740 | 0.615 (layer 6) | 0.818 (layer 7) |
| 0.6B RULER | 0.739 | 0.502 (layer 6) | 0.828 (layer 0) |
| 0.6B SCBench | 0.744 | 0.549 (layer 3) | 0.840 (layer 22) |
| 8B dev | 0.750 | 0.572 (layer 32) | 0.903 (layer 6) |
| 8B held-out (read-out only) | 0.745 | 0.552 (layer 30) | 0.901 (layer 6) |

The upper bound is a reasonably good but far from perfect proxy for attention. That matters for the noise sensitivity in S4.8.

## 4. Page size and K (0.6B dev, 14 documents per cell; 11 new collections + the S3 traces for page 16 / K 64)
G at B/K = 1.5 (`results_v2/ext/sweep/sweep_table.csv` has B/K = 1.25, 1.5, 2, 4 with CIs; `analysis_v2/sweep_table.py`):
| page \ K | 32 | 64 | 128 |
|---|---|---|---|
| 8 | EMA 0.269 · score 0.233 · allrank 0.256 (LRU 7.63 %) | 0.280 · 0.247 · 0.270 (7.27 %) | 0.291 · 0.263 · 0.285 (6.39 %) |
| 16 | 0.266 · 0.225 · 0.254 (6.40 %) | **0.279** · 0.240 · 0.270 (6.04 %) | 0.296 · 0.263 · 0.291 (4.82 %) |
| 32 | 0.271 · 0.235 · 0.262 (5.58 %) | 0.291 · 0.257 · 0.286 (4.54 %) | 0.306 · 0.270 · 0.303 (2.85 %) |
| 64 | 0.286 · 0.251 · 0.280 (4.39 %) | 0.307 · 0.269 · 0.303 (2.67 %) | 0.314 · 0.265 · 0.310 (1.00 %) |

The EMA beats LRU on 14/14 documents in all 12 cells; its G rises slightly with larger pages and larger K (0.27 → 0.31), and its
lead over the current score (0.03–0.05) and the all-page rank's closeness to it (within 0.013) hold everywhere. Each cell is its
own collection (the selector changes, so the generated text changes too); collection stops: mostly EOS.

## 5. Sampled decoding (0.6B dev, T=0.6, top-k 20, top-p 0.95, 3 seeds)
| Seed | LRU miss % | current score G | EMA (LODO) G [CI] | all-page rank EMA G | EMA wins |
|---|---|---|---|---|---|
| 0 | 5.99 | 0.245 | 0.281 [0.276, 0.285] | 0.273 | 14/14 |
| 1 | 5.95 | 0.242 | 0.280 [0.275, 0.284] | 0.271 | 14/14 |
| 2 | 5.90 | 0.247 | 0.282 [0.277, 0.286] | 0.274 | 14/14 |

Greedy decoding on the same documents: 0.279. Sampling changes every generated sequence, but not the result (spread across seeds 0.002).
`results_v2/ext/sampling/sampling_table.csv`.

## Deviations in S6
- 8B split: floor relaxed to 12K (plan-sanctioned). 25 of 35 documents come from LongBench's `gov_report_e` file (same GovReport
  source, contexts not in longctx_v1); recorded in the split's `rationale`.
- 64K: 2 documents middle-truncated (LongBench rule) to reach 14; 6 of 14 stopped on the loop rule.
- SCBench: contexts middle-truncated to 24K tokens to fit 32K with 5 turns (plan did not specify; recorded in the split).
- The 8B held-out evaluation, the 0.6B dev read-out collection and the other S6 collections ran on card 0 concurrently with
  each other (up to three processes on the same card). That does not affect traces (selection is deterministic; verified
  bit-identical on 0.6B dev), only collection time.
