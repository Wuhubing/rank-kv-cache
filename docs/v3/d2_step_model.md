# D2 (N2): why saved bytes do not become latency — a step-time model fitted to the measurements

**Conclusion: step time = fixed work + (missed pages × per-page cost) + a page-completion term, and on this machine the
per-page term is either hidden (batch ≤ 8: the step is launch-bound and a miss costs nothing measurable) or small (batch 32: 2.5 µs
per page, 12.5 % of LRU's step).** The upper bound of any policy's throughput gain is (transfer share) × (its share of saved
misses): for the EMA (−9.4 % misses) that is ≈ 1.2 % at batch 32, for Bélády (−33.5 %) ≈ 4.2 %, which Bélády reaches. The same
model, fitted to the simulated slow-tier runs, predicts the measured speedups to within 0.5 percentage points as the transfer share
grows to 75 % (E5). N2 holds on a PCIe GPU because the transfer share is small; it fails once transfer is a large share (slow tier,
or, by the same arithmetic, larger KV per page or more pages per step).

Model: least squares over every timed step of the LRU, current-score, EMA, allocation + EMA and Bélády runs of a configuration,
step_ms = a + b · (misses in the step, all requests and layers) + c · [page completed in the step] (`analysis_v3/e_summarize.py`,
`results_v3/e/*/summary/d2_model.csv`). Transfer share = b · (LRU misses per step) / (LRU step).

| config | batch | a (ms) | b (µs per missed page) | c (ms, page-completion step) | LRU transfer share |
|---|---|---|---|---|---|
| measured (E2) | 1 | 33.3 | −1.9 (≈ 0, not resolvable) | 4.1 | ≈ 0 |
| | 4 | 34.4 | −1.0 (≈ 0) | 4.2 | ≈ 0 |
| | 8 | 38.7 | −0.1 (≈ 0) | 4.7 | ≈ 0 |
| | 16 | 39.7 | 1.6 | 7.6 | 7.7 % |
| | 32 | 68.6 | 2.5 (26 GB/s for 64 KiB pages) | 13.9 | 12.5 % |
| simulated slow tier, d = 0.04 / 0.12 / 0.28 / 0.6 ns/B | 32 | 68.5 / 68.2 / 68.2 / 68.0 | 5.1 / 10.4 / 20.9 / 42.0 | 14.0 | 22.9 / 38.6 / 57.2 / 75.4 % |
| | 8 | 42.0 / 38.6 / 33.9 / 25.9 | −2.9 / 1.1 / 11.6 / 36.9 | 3.5–5.4 | ≈ 0 / 2.7 / 27.3 / 63.4 % |

(At batch ≤ 8 the fit's per-miss coefficient is within noise of zero, sometimes negative: the fetch is hidden behind the CPU's
launch time, as the all-miss runs confirm — fetching every page costs nothing at batch 1–4.)

Consequences:
- The fixed term a is the same for every policy (the policy kernels cost ~1 ms per step, E6), so a policy can only act through b.
- With this machine's b, 1 % throughput needs ~400 fewer misses per step at batch 32 (≈ 10 % of LRU's misses): only Bélády-like
  reductions show up clearly.
- The page-completion term c (4–14 ms once every 16 steps) is larger than the EMA's whole transfer saving and explains the tails of
  every policy (E4).
- Comparison with an independent decomposition: Xiaoyang Wu's L4 transfer-only benchmark (PR #9) gives 5.3 µs per page and a
  ~14 % transfer share from batch 8 on (transfer and compute timed separately and added); our end-to-end fit gives a smaller,
  partly hidden transfer at small batch and 12.5 % at batch 32 (`external_xiaoyang.md`).
