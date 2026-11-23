# Efficient replacement policy — dev results (longctx_v1, dev only)

Status: **dev only, directional.** Nothing here has touched the 21 held-out documents of
`splits/longctx_v1.json`. No GPU latency has been measured. Every hyperparameter was chosen by
leave-one-document-out (LODO) on the 14 dev documents unless the text says "fixed".

## Setup
- Qwen3-0.6B, Quest-style selector (min/max bounds, 16-token pages, K=64), unchanged from stage 1/2.
- Prompts: Qwen3 chat template, thinking enabled, "summarise the whole document". Greedy decoding;
  stop at EOS, 1024 tokens, or a repeated 32-gram (`notebooks/long_trace_capture.py`, `docs/long_trace_pilot.md`).
- Dev: 14 distinct documents (10 GovReport, 4 QMSum; 10 in 16–24K tokens, 4 in 24–32K), 473–1016
  decode steps each, all ending at EOS. 28 layers per document; every layer is an independent page cache.
- Semantics (HiSparse): every top-K page of a step must be resident. A step is a batch of compulsory
  admissions, and a policy only chooses which unselected residents to evict.
- Budgets: primary B=96 (B/K=1.5), curve B ∈ {72, 80, 96, 128, 256} (B/K 1.125–4), frozen before collection.
- G = (LRU misses − policy misses) / (LRU misses − Bélády misses), pooled over all layers and documents.
  CIs come from a 2000-replicate document-clustered bootstrap. "Repeat-miss drop" is the relative
  reduction in misses on pages that had been selected before.
- Code: `kvpolicy.py` (simulator and policies; 63 tests in the repo), `efficient_dev.py` (grid, LODO, allocation,
  learned models, gap analysis), `checks.py` (robustness, ablations). Raw outputs are backed up to
  Google Drive `rank-kv-cache/results_efficient_dev/` (misses per document × layer × config × budget).

## Headline (B=96, LRU 6.03%, Bélády 3.86% miss rate)

| policy | miss rate | G [95% CI] | repeat-miss drop | docs better than LRU |
|---|---|---|---|---|
| LRU | 6.03% | 0 | 0 | — |
| **score EMA** (α by LODO = 0.5 in every fold) | **5.42%** | **+0.279 [0.274, 0.284]** | 11.6% | 14/14 |
| current score (no averaging) | 5.50% | +0.244 [0.236, 0.251] | 10.1% | 14/14 |
| **per-layer allocation + score EMA** | **5.14%** | **+0.409** | ~16% | — |
| per-layer allocation + LRU | 5.75% | +0.127 [0.113, 0.140] | 5.3% | — |
| linear next-use regression (DAgger) | 5.38% | +0.300 [0.295, 0.305] | 12.4% | 14/14 |
| InfLLM-style decayed attention frequency | 5.97% | +0.028 | 1.2% | 14/14 |
| rank policy from stage 1/2 (LODO) | 5.97% | +0.027 | 1.1% | 14/14 |
| ARC / LRU-2 / LFU / FIFO | 6.04 / 6.13 / 7.85 / 8.18% | −0.01 / −0.05 / −0.84 / −1.00 | — | 3 / 0 / 0 / 0 |
| Bélády (offline) | 3.86% | 1 | 41.5% | — |

**Score EMA** keeps one float per resident page. Each step it updates
`e ← α·e + (1−α)·z`, with `z = (s − s_K) / (s_1 − s_K)`, where `s` is the page's current Quest score
and `s_1`, `s_K` are the best and K-th selected scores. It evicts the lowest `e`. A page admitted on a
miss starts at its current `z`.

## Checks

### A. Statistical stability
1. **Paired bootstrap.** At B=96 every score-aware variant has a CI far from 0. The CIs are narrow
   because the effect is very consistent across documents, not because there are many documents
   (only 14, from 2 tasks and one prompt family).
2. **Per-document consistency.** Score EMA beats LRU on 14/14 documents at every budget. Its worst
   document still has 9.3% fewer misses than LRU at B=96 (4.0% at B=256). Current score (no EMA)
   loses on 4/14 documents at B=256.
3. **By task and length (B=96).** Score EMA gives G +0.284 on GovReport (10/10 documents better)
   and +0.267 on QMSum (4/4). By length band it gives +0.282 (16–24K) and +0.273 (24–32K). The
   other policies rank the same way in every subgroup.
4. **Budget curve.** Score EMA G (LODO) is 0.256, 0.275, 0.279, 0.269 and 0.220 at B/K = 1.125, 1.25,
   1.5, 2 and 4. The advantage is consistent. The absolute saving is largest at tight budgets: 0.70 pp
   of miss rate at B=72 versus 0.08 pp at B=256.

### B. Hyperparameter robustness
5. **Decay α.** The curve is flat near the optimum. At B=96, α between 0.3 and 0.7 gives G between
   0.268 and 0.279. LODO picked the same α in all 14 folds at every budget. The optimum moves with the
   budget, from α≈0.4 at B/K≤1.25 to 0.5 at 1.5, 0.7 at 2 and 0.95 at 4. A held-out test must
   therefore freeze α per budget, or use a single α=0.5, which costs ≤0.02 G at B/K=1.5–2 and about
   0.07 at B/K=4.
6. **Per-layer allocation.** A greedy allocation over per-layer miss curves, in steps of 8 slots
   (64…256), with the total held at 28·B.
   - The LODO allocation recovers 89–94% of the oracle gain (the oracle is the best allocation chosen
     on the evaluated document itself) at B=80, 96 and 128.
   - It is almost identical across folds: the mean per-layer standard deviation across folds is 0.1–1.6
     slots.
   - Median allocation at B=96 (base EMA) gives layers 11 and 24–26 128/112 slots, and most others
     80–96.
   - Limitation: the grid gives no gain at B=72 (every layer needs more than K) or at B=256 (the grid
     stops at 256). These two budgets would need a wider grid.

### C. Mechanism and ablations (B=96, LODO G)

| signal averaged | residents updated every step (needs every resident's current score) | updated only when the page is selected |
|---|---|---|
| normalised score z | **+0.279** | −1.596 |
| raw score | +0.269 | — |
| rank only (schema v1: (K−r+1)/K if selected, else 0) | +0.016 | −1.849 |
| current z, no averaging | +0.244 | — |

7. **Which pages' scores are needed.** The gain depends entirely on updating unselected resident pages
   with their current scores.
   - If the average is updated only when a page is selected, it is far worse than LRU. Stale high
     values pin pages that are no longer near the top-K.
   - The key information is therefore the **current score of candidate pages that were not selected**.
     Quest computes it for every page at every step anyway, so it costs nothing to obtain.
8. **Magnitude versus rank.** Replacing scores by ranks, the only information in trace v1, removes
   nearly all of the gain (G 0.279 → 0.016, about LRU).
   - This explains the negative result of the rank project: the v1 trace discarded exactly the useful
     signal.
   - Averaging on top of the current score adds about +0.035 G.
   - "Score gap to the K-th page" orders candidates the same way as the current score within a step,
     so it is the same policy. Normalising the gap by (s_1 − s_K) matters only through the EMA (z vs
     raw: 0.279 vs 0.269).
9. **InfLLM-style baseline.** f ← d·f + a, where a is the page's share of the step's attention.
   - Approximation: the trace has no true attention mass, so a is the softmax of the Quest upper bounds
     over the selected pages with temperature √128, and unselected pages get 0.
   - LODO G is +0.028 at B=96 (+0.01 to +0.11 across budgets, decay 0.5 chosen). With decay 0.9 it is
     worse than LRU (G −0.11 to −0.14).
   - Like the rank variant, it only sees pages when they are selected, which is why it falls into the
     same regime.
   - Caveat: with true attention mass it might do better. Our version is a proxy.
10. **"Don't admit" and additivity.**
   - Under HiSparse semantics a selected page must be loaded, so "not admitting" can only mean
     "evict right after use". That action is already in every policy's action space.
   - Use-once evictions are 18.2% of Bélády's evictions at B=96 (13.3% at B=128), against 0.3% for
     LRU. This is the analogue of the MoE finding that non-admission is a sizeable share of Bélády's
     advantage.
   - Allocation and the score rule are close to additive: at B=96, EMA alone gives 0.279, allocation
     with LRU 0.127, and allocation with EMA 0.409 (sum 0.406). At B=128 it is 0.540 against a sum of
     0.579, and at B=80 0.309 against 0.302.

### Gap analysis (LRU decisions at B=96)
- 1.14 M LRU evictions; **68.5% of LRU's victims are selected again within 32 steps**, against 78.9% of
  the candidates LRU keeps. LRU's choice is only weakly better than random among candidates.
- Single-feature AUC for "reused within 32 steps" among candidates (values below 0.5 mean that a
  smaller value predicts reuse):
  - z_ema 0.70
  - current z 0.67
  - in-cache selection count 0.65
  - time resident 0.59
  - last rank 0.41
  - steps since last selection 0.36
- Correlation with log next-use distance: z_ema −0.37, z −0.32, age +0.29.
- DAgger regression coefficients (median over folds, raw feature units; target is log next-use
  distance) at B=96: z −4.0, z_ema −1.1, last rank +0.97, log age +0.53, log count −0.22, log
  residency −0.03. A higher current score means a sooner reuse; the score features dominate.

### Learned policies (no RL)
- A linear next-use-distance regression over 6 features (age, z, z_ema, log count, last rank, log
  residency), ridge-fitted with LODO.
  - Behaviour cloning on LRU-visited states: G 0.293, 0.313 and 0.313 at B=96, 128 and 256.
  - Two DAgger rounds: 0.300, 0.317 and 0.317. DAgger adds about 0.005.
- It is the best non-allocation policy, especially at B/K=4 (0.317 vs 0.220 for EMA). It needs about
  5 floats per resident plus 6 multiply-adds per candidate at eviction time.
- A tiny-student imitation of Bélády's victim choice was not tried separately. The DAgger regression
  on Bélády-style labels (true next-use distance) covers it at this capacity.

### D. External validity (checks only; not part of the main claim)
11. **Qwen3-1.7B, 4 dev documents, same frozen rules.** One QMSum document stopped at 292 steps on the
    32-gram loop rule; the others reached 1024 tokens.
    - At B=96: LRU 6.19%, Bélády 3.99%. Score EMA α=0.5 gives G +0.281 (0.6B on the same documents:
      +0.272), current score +0.246, InfLLM-style −0.105.
    - The best α is still 0.5 (0.3: 0.274, 0.7: 0.269). 4/4 documents beat LRU.
12. **Different summary prompt, 3 dev documents** (topic/section headings): EMA α=0.5 gives G +0.276
    (0.6B with the original prompt on the same documents: +0.273). 3/3 documents beat LRU. B=128
    behaves the same way.

### E. Overhead per layer per decode step (B residents, B−K eviction candidates)

| policy | state per resident | per-step compute | inputs it needs |
|---|---|---|---|
| LRU | 1 int | update K timestamps; argmin over candidates | selection |
| score (current) | 0 | gather B scores; argmin | Quest scores (already computed) |
| **score EMA** | 1 float | gather B scores, B FMAs, normalise by s_1, s_K; argmin | Quest scores |
| EMA, selected-only | 1 float | K FMAs | (fails) |
| InfLLM-style | 1 float | softmax over K, B FMAs | Quest scores |
| linear next-use | ~5 floats | about 6 MACs × candidates | Quest scores and counters |
| per-layer allocation | 0 | 0 (a 28-entry budget table) | offline dev curves |
| Bélády | — | future knowledge | not implementable |

For scale, Quest itself scores about 1,500–2,000 pages × 16 query heads × 128 dims per layer-step
(about 6–8 MFLOP). Score EMA adds about 100 gathers and 400 FLOPs, under 0.01%. Its state is
96 floats × 28 layers ≈ 11 KB per request.

Per-layer allocation fits HiSparse as long as each layer's GPU page pool is sized separately (same
total memory). It has no runtime cost, but it needs offline per-layer curves, which may depend on
the model.

## What is stable and what is not
**Stable on dev:**
- Score EMA beats LRU on every document, task, length band and budget, on a second model and on a
  second prompt.
- The mechanism (current scores of unselected residents) and the failure of rank-only information.
- LODO α choice per budget.
- Allocation recovers about 90% of its oracle gain and adds to the score rule at B/K 1.25–2.

**Directional or limited:**
- Only 14 dev documents from 2 tasks and one prompt family; the CIs reflect consistency, not breadth.
- The best α depends on the budget.
- Allocation is untested at B/K=1.125 and 4 because of grid limits.
- The InfLLM baseline uses a proxy for attention mass.
- G is measured against page-level Bélády.
- Miss rates are not latency. No GPU benchmark has been run.

## Suggested GPU benchmark (2–3 policies)
1. LRU (baseline).
2. Score EMA with α frozen from dev (0.5 at B=96): the lowest-cost policy with most of the gain.
3. Per-layer allocation + score EMA, the best G. If only two policies can be benchmarked, drop the
   linear model; its 0.02–0.1 extra G costs more state.

## Held-out preregistration draft (not executed; needs team sign-off)
- **Data:** the 21 held-out documents of `splits/longctx_v1.json` (sha of the commit that froze it),
  with the same model, prompts, selector, decoding and stop rule as dev. Collected once, with no
  re-runs.
- **Policies, frozen from dev:**
  - LRU.
  - Score EMA with α = 0.5 at B=96 and the per-budget dev α elsewhere (0.4, 0.4, 0.5, 0.7, 0.95 for
    B = 72, 80, 96, 128, 256).
  - Per-layer allocation + score EMA, with the allocation computed on all 14 dev documents (per-layer
    budgets written into the preregistration).
  - Secondary: current score, linear next-use model (coefficients refit on all 14 dev documents and
    frozen), InfLLM-style (decay 0.5).
  - Bélády as the reference.
- **Primary outcome:** G of score EMA versus LRU at B=96, pooled over held-out documents, with a
  2000-replicate document-clustered bootstrap 95% CI.
- **Success criteria (proposal):**
  - (i) the CI lower bound for G is above 0.15;
  - (ii) score EMA has fewer misses than LRU on at least 18/21 documents;
  - (iii) allocation + EMA has a higher G than EMA alone (paired bootstrap CI excluding 0).
- **Secondary:** the full budget curve, repeat-miss drop, per-task results.
- **Not claimed:** latency (it needs the GPU benchmark).

## Decisions for the team
1. Approve or edit the held-out preregistration above, in particular the success thresholds and
   whether α is frozen per budget or as a single value.
2. Whether per-layer allocation (offline, model-specific curves) is in scope as a "policy" for the
   report.
3. Whether to widen the allocation grid (needed for B/K=1.125 and 4) before freezing.
4. Which 2–3 policies go to the GPU benchmark.
5. Whether the InfLLM comparison needs a version with true attention mass, which needs a new
   capture field.

Housekeeping: `distill_kvrl.py` (abandoned RL distillation) is still untracked, unused and not
deleted.
