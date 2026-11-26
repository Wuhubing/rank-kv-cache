# D3 (N4): when do noisy scores make the policy worse than random?

**Conclusion: "worse than random" happens only when the policy reads scores that differ from the ones the selector uses, and then
quickly: a rank correlation of 0.95 between the two is already enough to wipe out the gain, whatever the noise's time structure.
When the same noisy scores also drive the selection, the policy keeps its gain (G 0.16–0.34 at B=96 for every noise model
tried).** The harmful noise is therefore a mismatch between the policy's signal and the selector's, not inaccuracy of the
selector against the true attention. Quest's own error against the true attention (ρ ≈ 0.73–0.81) is of the second kind and does
not hurt the cache (B4: the cache's target is Quest's next choice). No policy variant removes the effect: shrinking the initial
value (C3) changes G by ≤ 0.02 under noise, and the normalised power law, best without noise, is the most sensitive at high noise.

Source: `analysis_v3/noise.py` (models, empirical error pool, counterfactual; tests `tests/test_noise_v3.py`), 14 0.6B dev
documents, 3 noise seeds (G = mean; the seed range is < 0.01 everywhere), B=96 (B=128 in `results_v3/d3/summary.csv`). The
selection, LRU and Bélády are the trace's except in the counterfactual rows. ρ = mean Spearman correlation between the noisy and
the clean score rows.

## G at B=96
| Noise model | ρ(noisy, clean) | current score | EMA z α 0.5 | EMA percentile α 0.5 | normalised power law | EMA, shrunk initial value (toward K-th score) |
|---|---|---|---|---|---|---|
| none | 1.000 | 0.240 | 0.279 | 0.270 | **0.320** | 0.281 |
| iid σ=0.1 | 0.994 | 0.191 | 0.247 | 0.236 | 0.286 | 0.249 |
| iid σ=0.3 | 0.951 | −0.016 | 0.080 | 0.061 | 0.123 | 0.085 |
| iid σ=1 | 0.684 | −0.352 | −0.360 | −0.318 | −0.498 | −0.351 |
| AR(1) ρ=0.5, σ=0.1 / 0.3 / 1 | 0.994 / 0.951 / 0.684 | 0.183 / −0.059 / −0.442 | 0.227 / −0.016 / −0.498 | 0.219 / −0.022 / −0.470 | 0.267 / 0.029 / −0.617 | 0.229 / −0.008 / −0.487 |
| AR(1) ρ=0.9, σ=0.1 / 0.3 / 1 | 0.994 / 0.951 / 0.684 | 0.160 / −0.209 / −0.808 | 0.191 / −0.211 / −0.830 | 0.186 / −0.209 / −0.866 | 0.217 / −0.206 / −0.952 | 0.195 / −0.194 / −0.816 |
| order-preserving distortion k=0.5 (+ small iid) | 0.996 | 0.231 | 0.273 | 0.264 | 0.300 | 0.276 |
| order-preserving distortion k=2 (+ small iid) | 0.349 | −0.323 | −0.272 | −0.303 | −0.263 | −0.266 |
| measured Quest-vs-attention error (resampled, other documents) | 0.809 | −0.879 | −0.903 | −0.979 | −1.049 | −0.886 |
| **counterfactual: iid σ=0.3, selection from the noisy scores** | 0.951 | 0.201 | 0.218 | 0.217 | **0.340** | 0.233 |
| **counterfactual: AR(1) 0.9, σ=0.3** | 0.951 | 0.302 | 0.303 | 0.305 | 0.322 | 0.305 |
| **counterfactual: measured error** | 0.809 | 0.155 | 0.211 | 0.194 | **0.305** | 0.216 |
| (random eviction, replication-v2) | | −0.633 | | | | |

The counterfactual rows recompute LRU and Bélády on the noisy selection; the generation was **not** re-run (the trace's tokens
are kept), so the stream is not exactly what a noisy selector would produce. The "order-preserving" model with k=2 compresses the
low scores so much that the small added noise reorders them (ρ 0.35), so it is not order-preserving in effect.

## Findings
1. **Time correlation makes it worse, not better.** At the same ρ (0.95), AR(1) noise with ρ_t = 0.9 costs more than iid noise
   (EMA −0.211 vs +0.080): an EMA averages independent noise away for old pages, but a persistent error is a persistent wrong
   ranking.
2. **Where Quest's real error sits.** Placed on these curves, the measured Quest-vs-attention error (ρ 0.81) as *policy-only*
   noise would put every policy far below random (−0.88 to −1.05). In reality the selector is Quest itself, so that error is
   shared by selector and policy: the counterfactual with the measured error keeps G 0.21 (EMA) and 0.31 (power law). This is
   the experimental version of B4's point.
3. **Below random, explained.** With policy-only noise the policy evicts pages Quest is about to select, systematically: a
   page that Quest keeps choosing has a noisy value that sometimes falls low, and a resident page that Quest will not choose
   sometimes looks high. Random eviction at least does not prefer the wrong pages.
4. **C3 does not remove it.** Shrinking a new page's initial value toward the K-th score (or toward the layer's running mean)
   changes G by ≤ 0.02 at every noise level (shrink-to-mean column in `summary.csv`); the "new pages land in the tails" story of
   replication-v2 S4.8 is not the main mechanism.
5. **The best clean policy is the most fragile.** The normalised power law leads at low noise and in every counterfactual, but
   falls furthest at ρ ≤ 0.68 (its long memory keeps stale noisy values).

## Conditions (for the paper)
- Holds: the policy reads a copy of the scores that disagrees with the selector's (a cheaper estimate, a stale copy, a different
  quantization) at ρ ≲ 0.95.
- Does not hold: the policy reads the selector's own scores, however inaccurate those are as importance estimates; or the
  disagreement is tiny (ρ ≥ 0.99: the policy keeps 80–90 % of its G).
- Paper wording: "the replacement signal must be the selector's own score; a mismatch between the two, not selector error,
  is what pushes score-based eviction below random."
