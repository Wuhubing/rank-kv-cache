# S4.8 Statistics, noise injection and per-layer results (section 5.4, dev)

**Conclusions.**
1. **Stable across documents.** Every score policy with G > 0.2 beats LRU on 14/14 documents at B = 96 and 128. The paired
   bootstrap (same document resamples for every policy) separates even small differences: EMA − all-page rank +0.0075
   [0.0064, 0.0088], EMA − current score +0.038 [0.036, 0.040] (`stats/contrasts.csv`). Per-document G of the EMA
   (B=96) is plotted in `stats/per_doc_G_B96.png` (data `stats/per_doc_G.csv`).
2. **Noise does not fall back to LRU; it falls below random.** Gaussian noise on the scores the policy reads (σ in units of
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
