# C1: the best simple policy — a normalised power-law kernel of the score

**Conclusion: replacing the EMA by a power-law-weighted mean of the same z signal (γ = 1, zero at the K-th score) is the best
policy with O(1) state per page that we found, it is confirmed on a third held-out set, and it costs nothing measurable.**
Dev (LODO γ = 1 in every fold at B=96/128): G 0.320 vs EMA 0.279 on 0.6B, 0.319 vs 0.272 on 8B. Preregistered held-out (F3):
0.328 vs 0.281 (0.6B) and 0.320 vs 0.277 (8B), better than the EMA on 21/21 documents each, ΔG +0.048 [0.044, 0.051] and +0.043
[0.040, 0.046]. Its per-page state is 16 floats (8-term exponential mixture for the numerator and the normaliser), its work
3 360 operations per token and layer at B=96 (12× the EMA, still ~10⁻⁴ of a 0.6B model's work). It is the most sensitive policy
when the scores it reads disagree strongly with the selector's (D3), and the best one when they agree.

Rule: for every resident page keep a(p) = Σ_τ w(t−τ) z_τ and n(p) = Σ_τ w(t−τ) over its residency, with w(age) = (age+1)^−1
approximated by 8 exponentials (max relative error 2.7 % over ages 0–1023); evict the lowest a/n (ties: LRU). Code:
`kvpolicy_v3.Kernel(kernel="power", gamma=1, normalize=True, zero="kth")` (equal to `kvpolicy_v2.PowerLawDecay(..., normalize=True)`).

| | 0.6B dev | 8B dev | 0.6B held-out v3 | 8B held-out v3 |
|---|---|---|---|---|
| EMA (LODO / frozen α), B=96 | 0.279 | 0.272 | 0.281 | 0.277 |
| **normalised power law, B=96** | **0.320 [0.315, 0.326]** | **0.319 [0.314, 0.324]** | **0.328 [0.324, 0.333]** | **0.320 [0.315, 0.325]** |
| ΔG vs EMA, B=96 | +0.042 [0.036, 0.048] 14/14 | +0.047 [0.043, 0.051] 14/14 | +0.048 [0.044, 0.051] 21/21 | +0.043 [0.040, 0.046] 21/21 |
| ΔG vs EMA, B=128 | +0.065 | +0.063 | +0.066 [0.063, 0.069] | +0.055 [0.050, 0.059] |
| G at B=80 / 256 (dev) | 0.303 / 0.298 | 0.297 / 0.276 | – | – |

Noise (D3, 0.6B dev, B=96): 0.286 / 0.123 / −0.498 under policy-only iid noise σ = 0.1 / 0.3 / 1 (EMA 0.247 / 0.080 / −0.360); with
the selection also from the noisy scores 0.340 (iid σ 0.3) and 0.305 (measured error), the best of all policies.

Cost (C5): 3 360 ops/token/layer at B=96 (EMA 288); a two-scale normalised EMA (2 floats per page, 576 ops) gets about half of the
gain (+0.022 / +0.025 on dev).

Why it works (B1): the kernel weights a young page's few samples almost equally instead of letting the optimistic admission
value dominate, and its heavy tail keeps a long memory for old pages; it is not a residency penalty (explicit penalties hurt).

Sources: `results_v3/b1/`, `results_v3/d3/`, `results_v3/c5/`, `results_v3/f3/`; pages `b1_powerlaw.md`, `d3_noise.md`,
`c5_cost.md`, `f3_heldout.md`.
