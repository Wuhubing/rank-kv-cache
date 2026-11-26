# D5 (N6): pinned pages and protection zones

**Conclusion: on this workload pinning is redundant and a protection zone is harmful, because the pages these mechanisms protect
are either selected anyway or not needed; a protection zone helps only when the eviction score is biased against new entries,
which reproduces CO2's finding in its own setting.** Page 0 (the sink) is selected at 89 % (0.6B) / 83 % (8B) of steps and the
newest completed page at 98 % / 94 %, so pinning them changes nothing (ΔG 0.000 ± 0.001); pages 1–3 are selected at only 5–10 % of
steps, so pinning 2–4 sink pages costs 0.03–0.08 G (8B up to 0.09). A FIFO protection zone of ρ·B slots never helps LRU or the EMA
at any budget, in any overlap group of layers, or in any regime of the synthetic model; it costs up to 0.7–0.9 G at ρ = 0.5.
With an H₂O-style *accumulated* attention share as the eviction score — the score CO2 starts from, which by construction is low
for new entries — the same protection zone helps strongly (0.6B B=96: G −0.77 without it, −0.39 at ρ = 0.2–0.3; 8B −0.98 → −0.47),
with the best ρ near CO2's 0.25. The flip condition is therefore the score's bias against young pages, not the access pattern.

Sources: pins and protection `analysis_v3.run d5` (`kvpolicy_v3.Protect`; tests: no protection equals LRU / the EMA, protection on
the EMA equals `kvpolicy_v2.CO2` with N=0), per-layer groups `analysis_v3/d5_summary.py` (`results_v3/d5/summary_{dev,8b}/`), the
accumulated-score control `analysis_v3.run d5b` (`results_v3/d5/d5b_{dev,8b}/`), selection shares `analysis_v3/instrument.py`
(`results_v3/inst/*/pins.csv`), synthetic regimes `results_v3/b2/synthetic/curves_summary.csv`. Audits: d5 3 136 / 4 032, d5b
1 568 / 2 016 streams, 0 mismatches.

## How often the protected pages are selected anyway (share of steps; mean over layers and documents)
| page | 0.6B dev | 8B dev |
|---|---|---|
| page 0 (sink) | 0.892 | 0.833 |
| pages 1 / 2 / 3 | 0.045 / 0.055 / 0.049 | 0.085 / 0.104 / 0.107 |
| newest completed page / 2nd / 3rd / 4th | 0.982 / 0.945 / 0.917 / 0.886 | 0.942 / 0.862 / 0.804 / 0.761 |

## Effect (ΔG against the same policy without protection; 14 dev documents)
| Config | 0.6B B=96 | 0.6B B=128 | 8B B=96 | 8B B=128 |
|---|---|---|---|---|
| EMA + pin sink 1, newest 1 | 0.000 | 0.000 | −0.001 | 0.000 |
| EMA + pin newest 4 | −0.002 | +0.002 | −0.006 | +0.004 |
| EMA + pin sink 2 | −0.027 | −0.014 | −0.030 | −0.014 |
| EMA + pin sink 4, newest 4 | −0.080 | −0.038 | −0.090 | −0.035 |
| LRU + protection ρ = 0.1 / 0.3 | −0.003 / −0.183 | 0.000 / −0.051 | −0.006 / −0.217 | 0.000 / −0.071 |
| EMA + protection ρ = 0.05 / 0.1 / 0.3 | −0.008 / −0.033 / −0.319 | −0.001 / −0.008 / −0.153 | −0.010 / −0.038 / −0.351 | 0.000 / −0.009 / −0.174 |
| **accumulated share, ρ = 0 → 0.2 → 0.3 (G)** | **−0.771 → −0.394 → −0.393** | −0.992 → −0.394 → −0.362 | **−0.983 → −0.479 → −0.469** | −1.256 → −0.542 → −0.487 |

Per layer group (terciles of the layers' adjacent-step overlap, 0.6B: ≤ 0.819, 0.819–0.862, > 0.862): the protection zone hurts
in every group, most in high-overlap layers for LRU (ρ = 0.2: −0.041 / −0.053 / −0.101). Layers where a config beats its base on
≥ 12 of 14 documents: 0 for every protection and pin config at B=96 on 0.6B; at most 3 (pin newest 4, 8B B=128).

## Comparison with CO2 (`docs/workshop/papers/co2.md`)
| | CO2 | here |
|---|---|---|
| access pattern | dense attention over every cached token | Quest selects K = 64 pages per step |
| unit | token | 16-token page |
| eviction | permanent (no recall) | recall from the host tier |
| score | H₂O accumulated attention (no normalisation): new tokens start near 0 | EMA of the selector's current score: a new page starts at its admission score |
| per-step selection score for every entry | no | yes |
| protection zone share | 25–50 % of the budget | 0–50 % (swept) |
| baseline | H₂O (no LRU) | LRU and the EMA |
| result | ρ 0.5 → 0.25 is the main gain | protection never helps the EMA or LRU; helps the accumulated score as in CO2 |

Synthetic model: protection hurts in every regime tried (overlap 0.59–0.94, recency boost 0–3, fast persistence 0.3–0.99), because
the model's score is unbiased; it does not contain an accumulated score, so the flip was established on the traces (d5b) instead.

## Paper wording
"Pinning sink and recent pages is redundant under a Quest selector (they are selected anyway), and a FIFO protection zone helps
only an eviction score that is biased against new entries, such as accumulated attention; with a score-EMA it only takes slots."
This is also why ArkVale's forced pages hardly change the access stream (`docs/rerun/s4/09_arkvale_full.md`).
