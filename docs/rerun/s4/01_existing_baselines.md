# S4.1 Existing baselines re-run on the new dev traces (section 6.1)

**Conclusion.** On the 14 RTX 6000 Ada dev traces, every 6.1 baseline reproduces its L4 value, and the ranking is
unchanged: linear DAgger (G 0.300) > score EMA (0.279) > InfLLM-style decayed sum over all residents (0.266) > hybrid
(0.250) > current score (0.240) ≫ decayed count (0.040), rank policy (0.029) ≈ LRU (0). FIFO, LFU, LRU-2 and ARC are
worse than LRU at B=96. Note that the **existing `infllm` class**, which gives attention share only to the
*selected* pages (the closest reading of InfLLM's Eq. 1 with a softmax proxy), reaches only G 0.026. The 0.269 "InfLLM-style"
number in the plan is the decayed sum over **all residents** (`dsum_softmax`), which is closer to our EMA than to InfLLM.

Source: `results_v2/dev/efficient_dev/` (`efficient_dev.py` and `checks.py`, unchanged; `checks_report.txt` has every row).

## B = 96 (B/K = 1.5), pooled over 14 documents and 28 layers; document bootstrap CI (2000)
| Policy | miss % | G [95 % CI] | repeat-miss drop vs LRU | docs better than LRU |
|---|---|---|---|---|
| LRU (miss-first ties, the replay convention) | 6.04 | 0 | 0 | – |
| LRU, hit-first ties (HiSparse) | 6.13 | −0.044 [−0.047, −0.042] | −1.8 % | 0/14 |
| FIFO | 8.20 | −0.995 [−1.018, −0.969] | −41.3 % | 0/14 |
| LFU | 7.89 | −0.855 [−0.894, −0.810] | −35.5 % | 0/14 |
| LRU-2 | 6.15 | −0.052 [−0.062, −0.044] | −2.2 % | 0/14 |
| ARC | 6.06 | −0.012 [−0.014, −0.009] | −0.5 % | 0/14 |
| Current score (ArkVale-style, no pins) | 5.52 | 0.240 [0.235, 0.246] | 10.0 % | 14/14 |
| InfLLM class (selected pages only, softmax proxy, LODO d) | 5.98 | 0.026 [0.022, 0.030] | 1.1 % | 14/14 |
| Decayed sum, softmax over all residents (LODO d=0.5) | 5.46 | 0.266 [0.261, 0.271] | 11.0 % | 14/14 |
| Decayed count, no score (LODO d=0.8) | 5.95 | 0.040 [0.030, 0.051] | 1.7 % | 14/14 |
| Rank policy, stage 1–2 (LODO) | 5.97 | 0.029 [0.023, 0.035] | 1.2 % | 14/14 |
| Hybrid LRU + score (LODO) | 5.49 | 0.250 [0.247, 0.253] | 10.4 % | 14/14 |
| **Score EMA (z, all residents; LODO α=0.5)** | **5.43** | **0.279 [0.274, 0.283]** | **11.6 %** | **14/14** |
| Score EMA, raw score | 5.46 | 0.267 [0.262, 0.272] | 11.1 % | 14/14 |
| Score EMA, selected pages only | 9.57 | −1.627 | −67.6 % | 0/14 |
| Linear next-use regression (DAgger) | 5.39 | 0.300 [0.295, 0.304] | – | – |
| Bélády | 3.87 | 1 | 41.5 % | 14/14 |

Other budgets (B = 72, 80, 128, 256) are in `summary.csv` and `checks_stats.csv`, and side by side with L4 in
`docs/rerun/s3_dev_vs_l4.md`. At B=256 the score family's G shrinks (current score 0.116, EMA 0.236), as on L4.

## What this adds
- Updating **every resident** every step is essential: the "selected only" EMA and InfLLM's selected-only decayed sum are
  far worse (the selected-only EMA is much worse than LRU). The signal is in the scores of pages that are *not* selected.
- By task (B=96): gov_report EMA 0.280 / score 0.244; qmsum is in `checks_report.txt` (A3).
- Use-once evictions (C10): Bélády evicts a page admitted on the previous step and never hit in 18 % of its evictions at B=96,
  LRU in 0.4 %. This is the one form of "bypass" available under the HiSparse semantics (every selected page must be loaded).

New baselines (section 6.2), the budget sweep and the α curve: `04_new_baselines.md` to `06_alpha_curve.md`.
