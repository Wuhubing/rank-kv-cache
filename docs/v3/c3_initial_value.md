# C3: the initial value of a newly admitted page

**Conclusion: lowering a new page's optimistic admission value helps the EMA only slightly, and does nothing against noise.**
Shrinking the initial z halfway toward the K-th score (z = 0) gives ΔG +0.003 [0.003, 0.003] (0.6B) and +0.005 [0.004, 0.006]
(8B) at B=96, better on 14/14 documents; shrinking toward the layer's running mean of admission values is slightly worse than the
plain EMA (−0.002). For the percentile signal, shrinking toward 0 (the lowest rank) is harmful (−0.087). Under noise the shrunken
variants move G by ≤ 0.02 (D3), so they do not remove the "worse than random" effect. The direction matches B1's mechanism (the
EMA over-trusts the admission sample), but the size shows that the normalised kernel's gain (+0.042 / +0.047) comes from the
whole weighting of a young page's samples, not just from the first value.

Source: `analysis_v3.run c3` (`kvpolicy_v3.ShrinkEMA`: initial value (1 − c)·x + c·prior; prior = 0 or the running mean;
c ∈ {0.25, 0.5, 0.75} and α ∈ {0.3, 0.5, 0.7} tuned LODO; tests: c = 0 equals the EMA). A count-weighted prior c = k/(k+n) with
n = 1 sample at admission is the same as a fixed c, so it is covered. Results `results_v3/c3/{c3_dev,c3_8b}/`; audits 3 136 / 4 032
streams, 0 mismatches. Noise: `docs/v3/d3_noise.md`.

| Policy (LODO) | 0.6B B=96 | 0.6B B=128 | 8B B=96 | 8B B=128 |
|---|---|---|---|---|
| EMA of z | 0.279 | 0.267 | 0.272 | 0.256 |
| EMA of z, initial value shrunk toward the K-th score | 0.281 (+0.003 [0.003, 0.003], 14/14) | 0.270 (+0.003) | 0.277 (+0.005 [0.004, 0.006], 14/14) | 0.257 (+0.001) |
| EMA of z, initial value shrunk toward the running mean | 0.277 (−0.002) | 0.267 (0.000) | 0.270 (−0.002) | 0.255 (−0.001) |
| EMA of percentile | 0.271 | 0.259 | 0.263 | 0.241 |
| EMA of percentile, shrunk toward the running mean | 0.269 (−0.002) | 0.259 (0.000) | 0.260 (−0.002) | 0.238 (−0.003) |
| EMA of percentile, shrunk toward 0 | 0.184 (−0.087) | 0.229 (−0.030) | 0.176 (−0.087) | 0.192 (−0.049) |
| (normalised power law, C1) | 0.320 | 0.333 | 0.319 | 0.319 |
