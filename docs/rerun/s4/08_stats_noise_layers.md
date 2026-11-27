# S4.8 Statistics, noise injection and per-layer results (section 5.4, dev)

**Conclusions.**
1. **Stable across documents.** Every score policy with G > 0.2 beats LRU on 14/14 documents at B = 96 and 128. The paired
   bootstrap (same document resamples for every policy) separates even small differences: EMA − all-page rank +0.0075
   [0.0064, 0.0088], EMA − current score +0.038 [0.036, 0.040] (`stats/contrasts.csv`). Per-document G of the EMA
   (B=96) is plotted in `stats/per_doc_G_B96.png` (data `stats/per_doc_G.csv`).
2. **Noise does not fall back to LRU; it falls below random** (under this noise model; its limitations are listed at the end).
   Gaussian noise on the scores the policy reads (σ in units of
   that step's score spread across pages; the selection is unchanged): EMA(z, α=0.5) keeps G 0.246 at σ=0.1, drops to 0.079
   at σ=0.3, and goes negative from σ=1 (−0.359), reaching −0.81 at σ ≥ 10, worse than random eviction (−0.63). The current
   score behaves the same (−0.63 at σ=100, i.e. random). The ties to LRU only break exact ties, so a noisy score replaces
   LRU's information instead of adding to it. **The policy needs a reasonably accurate score.**
3. **Large differences between layers.** At B=96 the LRU miss rate ranges from 1.0 % to 16.1 % across the 28 layers, and the
   EMA's G from 0.16 (layer 10) to 0.38 (layer 27); the current score's from 0.05 to 0.38 (`per_layer/per_layer_B96.csv`).
   This variation is what the per-layer allocation exploits (S4.3).

## Noise injection (B=96, 14 documents, 3 noise seeds; G mean over seeds, the spread is < 0.01)
| σ | EMA z α=0.5 | EMA z α=0.7 | EMA allrank α=0.5 | current score |
|---|---|---|---|---|
| 0 | 0.279 | 0.267 | 0.270 | 0.240 |
| 0.1 | 0.246 | 0.239 | 0.236 | 0.191 |
| 0.3 | 0.079 | 0.074 | 0.059 | −0.017 |
| 1 | −0.359 | −0.420 | −0.317 | −0.352 |
| 3 | −0.718 | −0.849 | −0.549 | −0.525 |
| 10 | −0.799 | −0.933 | −0.671 | −0.600 |
| 100 | −0.807 | −0.941 | −0.726 | −0.631 |

At very high noise the EMA is *worse* than random (−0.81 vs −0.63). A page admitted this step starts at one noisy value,
while older pages average many noisy values toward the mean, so new pages land in the tails more often and are evicted
first (with α=0.7 even more so). This is a design hazard for any decayed-score policy fed by a noisy importance
estimate.

Source: `analysis_v2/s4_dev.py noise` → `results_v2/dev/noise/noise_summary.csv` (and the gzipped per-layer counts).

## Limitations of this noise model (added in followup-v3, R12)
The numbers above depend on the noise model, which is not how a real estimator errs:
1. **The noise does not change the selection.** It is added only to the scores the replacement policy reads; `ranked` (the
   selector's top-K, which decides the access stream) is unchanged. A real noisy estimator would also pick different pages, so
   the access stream, LRU and Bélády would all change.
2. **The noise is independent at every step.** A real estimation error is correlated over time (the same page is over- or
   under-estimated for many steps, because its keys do not change). Independent noise is the case that an EMA averages out best
   for old pages and worst for newly admitted ones.
3. **The z denominator is contaminated too.** z = (s − s_K)/(s_1 − s_K) is computed from the noisy scores, so the noise also
   changes the scale and zero point of every page's signal at that step (the top-1 and K-th scores are themselves noisy).
More realistic models (time-correlated AR(1) noise, order-preserving systematic distortion, the measured Quest-vs-true-attention
error, and a counterfactual in which noise also changes the selection) are followup-v3 D3. Until then, read point 2 of the
conclusions as "under independent per-step Gaussian noise on the policy's input".
