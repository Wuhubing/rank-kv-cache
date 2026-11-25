# B1: why the power-law kernel beats the EMA

**Conclusion: the advantage comes from the kernel's shape (a heavy tail) and from averaging (normalisation), not from an
implicit residency penalty.** On 0.6B dev at B=96, the normalised power-law kernel (a power-law-weighted mean of z) reaches G
0.320, the unnormalised one 0.310, the normalised exponential kernel 0.301 and the EMA 0.279. Every explicit residency-type term
added to the EMA makes it worse (time since admission −0.247, unselected-step count −0.093, cumulative deficit −0.009, time since
last selection −0.012; LRU + score bonus −0.055), and moving the zero point so that residency *adds* value collapses the
unnormalised kernel (zero at the minimum score: −0.115). The unnormalised sum does carry an implicit penalty, but it only works
when the zero point is the K-th score, and normalising it removes the penalty and improves G. The EMA's weakness is its weighting
of young pages: it keeps a fixed weight α^n on the admission value, whereas a normalised kernel averages a young page's few
samples equally (see "mechanism"). A two-time-scale EMA (2 floats per page) recovers about half of the power law's gain over the EMA.
[8B dev: see the section at the end.]

Source: `analysis_v3.run b1` (`analysis_v3/experiments.py: b1`), policies `kvpolicy_v3.Kernel`, `EMAAge`, `EMADeficit`,
`EMAUnsel`, `TwoScale` and `kvpolicy.Hybrid`; tests `tests/test_kvpolicy_v3.py` (the power kernel equals
`kvpolicy_v2.PowerLawDecay` for every normalisation / zero point; the exponential kernel equals the direct sum). Families tuned
LODO (fewest misses on the other 13 documents); paired document bootstrap (2000, seed 0). Results `results_v3/b1/b1_dev/`; LRU and
Bélády re-checked with `replay.py`: 3 136 streams, 0 mismatches.

## Factorial: kernel shape × normalisation × zero point (0.6B dev, G; CI at B=96)
| Kernel | Normalised | Zero point | B=80 | B=96 | B=128 | B=256 |
|---|---|---|---|---|---|---|
| EMA of z (reference; LODO α) | (yes) | K-th | 0.274 | 0.279 [0.274, 0.283] | 0.267 | 0.236 |
| exponential (rate 0.3–0.9) | no | K-th | 0.279 | 0.281 [0.276, 0.285] | 0.269 | 0.236 |
| exponential | **yes** | K-th | 0.298 | 0.301 [0.297, 0.304] | 0.299 | 0.241 |
| exponential | no | median | 0.280 | 0.274 | 0.255 | 0.192 |
| exponential | yes | median | 0.289 | 0.296 | 0.286 | 0.240 |
| exponential | no | min | 0.211 | 0.198 | 0.227 | 0.142 |
| exponential | yes | min | 0.274 | 0.280 | 0.264 | 0.221 |
| power law (γ 0.5 / 1 / 2) | no | K-th | 0.292 | 0.310 [0.305, 0.315] | 0.295 | 0.244 |
| power law | **yes** | **K-th** | **0.303** | **0.320 [0.315, 0.326]** | **0.333** | **0.298** |
| power law | no | median | 0.170 | 0.203 | 0.258 | 0.171 |
| power law | yes | median | 0.300 | 0.298 | 0.315 | 0.294 |
| power law | no | min | −0.014 | −0.115 | −0.120 | 0.064 |
| power law | yes | min | 0.272 | 0.284 | 0.266 | 0.232 |

LODO choices: power law γ = 1 in every fold at every budget; exponential rate 0.7 (B=96) / 0.9 (B=128) normalised, 0.5 / 0.7 not.

Main effects at B=96 (paired ΔG, 14/14 documents unless stated):
- **shape** (power − exponential): normalised +0.020, unnormalised +0.029 (power − EMA +0.032 [0.027, 0.036]);
- **normalisation**: exponential +0.020, power +0.010 [0.008, 0.012]; normalised power − EMA **+0.042 [0.036, 0.048]**
  (B=128: +0.065 [0.061, 0.069]);
- **zero point**: matters a lot for the unnormalised sum (K-th 0.310, median 0.203, min −0.115) and little once normalised
  (0.320 / 0.298 / 0.284): it changes whether a resident page gains or loses value merely by staying.

## Candidate explanations as explicit terms on the EMA (LODO over α and λ; 0.6B dev)
| Policy (key) | B=80 | B=96 | B=128 | B=256 | ΔG vs EMA at B=96 [CI] |
|---|---|---|---|---|---|
| EMA(z) − λ log(1 + steps since admission) | −0.038 | 0.032 | 0.109 | 0.140 | −0.247 [−0.254, −0.240] |
| EMA(z) − λ log(1 + steps since last selected) | 0.237 | 0.267 | 0.276 | 0.268 | −0.012 [−0.014, −0.010] (B=128 +0.008, B=256 +0.032) |
| EMA(z) + λ Σ min(z, 0) (cumulative deficit) | 0.269 | 0.270 | 0.243 | 0.119 | −0.009 [−0.010, −0.008] |
| EMA(z) − λ log(1 + unselected steps since admission) | 0.168 | 0.186 | 0.192 | 0.171 | −0.093 [−0.096, −0.089] |
| last + w · EMA(z) (`kvpolicy.Hybrid`, linear recency; not the same as the log terms) | 0.214 | 0.224 | 0.217 | 0.142 | −0.055 [−0.058, −0.052] |
| w · EMA_fast + (1 − w) · EMA_slow (two time scales, normalised) | 0.293 | 0.300 [0.296, 0.304] | 0.294 | 0.245 | +0.022 [0.019, 0.024] |
| Normalised power law (for comparison) | 0.303 | 0.320 | 0.333 | 0.298 | +0.042 [0.036, 0.048] |

λ grids start at a small positive value (0.05, 0.01 for the deficit), so these rows test whether *adding* the term helps; LODO
chose the smallest λ in every fold, i.e. as little of the term as the grid allowed. `kvpolicy.Hybrid` is linear in the
recency and adds the EMA to the LRU key, so it is not equivalent to the log-penalty variants; its results are reported as a
separate row. The only term that helps anywhere is a recency term (time since last selected) at large budgets (B ≥ 128),
where LRU itself is nearly optimal.

## Mechanism
- Most unselected pages have z < 0, so an unnormalised sum of z lowers a page's value every step it stays unselected: the
  reviewer's "implicit residency penalty". But that penalty is not what helps: making it explicit hurts (table above), and
  removing it by normalisation helps.
- What the normalised kernels share and the EMA lacks is the weighting of a young page's few samples. The EMA starts at the
  admission value x₀ and gives it weight α^n after n steps (0.5^n; with a long memory, e.g. α=0.9, x₀ dominates for tens of
  steps); a normalised kernel weights the n + 1 samples by k(age)/Σk, i.e. close to an equal average while the page is young
  and the kernel's long-memory weights once it is old. The admission value is the score at which the page was *selected*
  (z ≥ 0 by definition), so the EMA protects newly admitted pages by a stale, optimistic value. The power law adds a heavier
  tail than any single exponential, which is why it beats the normalised exponential (by 0.020) and the two-scale mixture.
- Practical consequence: the best simple policy found is the normalised power-law kernel of z with γ=1 (16 floats per page with
  the 8-term mixture, still O(1)); a two-scale normalised EMA (2 floats) gets about half of the gain over the EMA. C1 freezes
  the candidate; C3 tests the initial value directly.

## 8B dev
(pending: `results_v3/b1/b1_8b/`)
