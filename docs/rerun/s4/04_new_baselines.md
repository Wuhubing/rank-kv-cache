# S4.4 New baselines (section 6.2), dev, LODO-tuned where a family has parameters

**Conclusion.** No classic cache policy helps: Random (−0.63), SIEVE (−0.02) and S3-FIFO (−0.10) at B=96 are all at or below
LRU. Every published score-based rule we could reproduce is weaker than the score EMA: ArkVale's 2+2 pinned pages
cost gap (0.215 vs 0.240 for plain current score), and the InfLLM rules (paper Eq. 1, code `lru-s`) give attention mass only to
selected pages and reach G ≈ 0.01–0.03 (with the *true* attention mass they are slightly worse than LRU; S6). Holt's trend term
and the CO2 factors add nothing. A power-law age kernel instead of the exponential decay is **better** than the EMA:
G 0.310 vs 0.279 at B=96, paired ΔG 0.032 [0.027, 0.036], better on 14/14 documents. Dev only, so this is exploratory.

Source: `analysis_v2/s4_dev.py replay/summarize` (`results_v2/dev/s4/`); policies in `kvpolicy_v2.py` (tests:
`tests/test_kvpolicy_v2.py`, including independent reference implementations of SIEVE and S3-FIFO). Every replay's LRU
and Bélády counts were re-checked with the independent simulator in `replay.py`: 3 920 streams, 0 mismatches
(`results_v2/dev/s4/audit.txt`).

## B = 96 and 128 (G, 95 % document-bootstrap CI; documents better than LRU of 14)
| Policy (exact rule) | G @96 | G @128 | wins @96 |
|---|---|---|---|
| Random among unselected residents (5 seeds; mean) | −0.633 [−0.646, −0.620] | −0.820 | 0 |
| SIEVE (NSDI'24; hand skips current selection) | −0.019 [−0.024, −0.014] | −0.014 | 1 |
| S3-FIFO (SOSP'23; 10 % small queue, ghost, 2-bit freq) | −0.095 [−0.106, −0.084] | −0.151 | 0 |
| Current score = FlashOffloading "Score" = ArkVale without pins | 0.240 [0.235, 0.246] | 0.216 | 14 |
| ArkVale, 2 sink + 2 recent pages pinned (32-token pages in ArkVale ≈ 4+4 here) | 0.215 [0.210, 0.220] | 0.202 | 14 |
| ArkVale, 4 sink + 4 recent pages pinned | 0.164 [0.159, 0.169] | 0.178 | 14 |
| InfLLM paper Eq. 1, d=0.1 (softmax proxy over selected pages) | 0.016 [0.014, 0.017] | 0.006 | 14 |
| InfLLM paper Eq. 1, LODO d | 0.026 | 0.012 | 14 |
| InfLLM code `lru-s` (selected: reset to this step's mass), LODO d | 0.010 [0.009, 0.011] | 0.003 | 14 |
| Lazy EMA refresh every r steps, r=2 / 4 / 8 (InertiaKV) | 0.198 / 0.072 / −0.097 | 0.223 / 0.136 / −0.012 | 14 / 14 / 0 |
| Frozen ranking (value at admission, never updated) | −1.443 | −1.977 | 0 |
| CO2 factorial α × ρ × N, LODO | 0.279 (picks ρ=0, N=0 = EMA) | 0.268 | 14 |
| Holt level + trend (PRR grid 4×5×4), LODO | 0.281 [0.277, 0.285] | 0.274 | 14 |
| **Power-law age kernel, γ=1 (LODO picks γ=1)** | **0.310 [0.305, 0.315]** | **0.295** | **14** |
| Score EMA z (reference) | 0.279 [0.274, 0.283] | 0.267 | 14 |

Paired contrasts (`stats/contrasts.csv`): EMA − current score +0.038 [0.036, 0.040]; EMA − ArkVale 2+2 +0.064; EMA − Holt
−0.003 [−0.003, −0.002]; EMA − power law −0.032 [−0.036, −0.027]; EMA − decayed softmax sum +0.013 [0.009, 0.016].

## Notes on the rules
- ArkVale (deviation from its code, stated in `kvpolicy_v2.ArkVale`): pinned pages are protected while resident but not
  force-loaded, because under the Quest selector only selected pages are fetched. **The full version is in
  [09_arkvale_full.md](09_arkvale_full.md)**: forced pages, ArkVale's window rule, unbudgeted layers 0–1 and the
  cuboid-mean digest (new trace read-out). It is weaker than this simplified version (G −0.219 at B=96).
- Holt follows PRR's convention (α weights the new observation). The trend term never helps: the best γ is 0.25–0.5 and
  the mean G falls monotonically with γ (`factorials/main_effects.csv`).
- CO2: the FIFO protection zone hurts (mean G over the other factors: ρ=0 0.038, ρ=0.25 −0.090, ρ=0.5 −0.559), and so does the
  minimum residency (N=0 −0.118, N=4 −0.185, N=16 −0.307). All the benefit comes from the decayed score, none from
  CO2's recency/deferral factors. This is the opposite of CO2's own finding (its gain came almost entirely from ρ).
- Power law: the kernel (age+1)^−γ is computed with a non-negative exponential mixture (max relative error 2.7 % for γ=1,
  `kvpolicy_v2.powerlaw_mixture`). γ=0.5 is worse (0.211) and γ=2 is close to the EMA (0.288). A heavy-tailed memory of past
  scores helps; this was not in the original study and has no held-out confirmation.
