# E: system measurements, redone (fixed batched path, randomised order, ≥ 5 rounds)

**Conclusion (0.6B, PCIe RTX 6000 Ada): fewer misses become throughput only in proportion to transfer's share of the step, and
that share is small.** With batch ≤ 8 the decode step is bound by kernel launches (~33 ms for every policy; even fetching all K pages
per request and layer costs nothing at batch 1–4). At batch 32 transfer is 12 % of LRU's step (9.3 ms of 77 ms): Bélády, with 34 %
fewer misses, gains +4.2 % [4.2, 4.3] throughput, the EMA +0.1 % [0.1, 0.1], allocation + EMA +0.6 % [0.6, 0.7], the current score
+0.1 %. Bélády's gain equals the E3 upper bound (saved misses × 2.42 µs per missed page), the EMA's is below its bound (0.9 ms,
1.2 %). Replication-v2's batch-32 "+3.4 %" is gone with the fixed path and the randomised order: LRU no longer has a long tail; the
slow steps of every policy are the page-completion steps. E_MORE_PLACEHOLDER

Setup: `gpu_bench_batch2.py` (E1) with `overlap=False` (see "E1 status"), driver `analysis_v3/e_bench.py`, summary
`analysis_v3/e_summarize.py`. wulab2 card 0 (GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3), no other process on the card in any run
(checked and recorded before and after every run); host load average 3.4 at the start of the configuration (the precondition),
2–16 during runs (this process's CPU threads plus other users; recorded per run). GPU 64–73 °C, SM 2.5–2.7 GHz. Requests: the
longctx_v1 prompts with ≥ 16 384 tokens in rid order, cut to 16 384 tokens, nested across batch sizes. B=96, frozen α (0.5) and
allocation (`splits/heldout_frozen_v1.json`). Per batch: a preparation LRU run (simulator check, Bélády's next uses), then 5 rounds,
each a discarded warm-up run and every policy once in a random order (seed = round + 1000·batch); 264 steps per run, 8 warm-up
steps excluded (256 timed). Ratios: mean throughput of the policy / LRU, paired by round, bootstrap CI over rounds. Every LRU
run's per-request misses equal the simulator; ranks and tokens are identical across policies (except "nofetch").
Results: `results_v3/e/batch_06b/` (`summary/throughput.csv`, `tails.csv`, `d2_model.csv`; raw per-step data `raw/*.json.gz`).

## E2: throughput relative to LRU (0.6B, 16K-token prompts, B=96)
| batch | LRU median step (ms) | LRU tokens/s | current score | EMA | allocation + EMA | Bélády | LRU misses / step |
|---|---|---|---|---|---|---|---|
| 1 | 33.1 | 29.8 | 1.009 [0.997, 1.029] | 1.008 [0.998, 1.026] | 1.010 [1.000, 1.029] | 1.005 [0.995, 1.023] | 140 |
| 4 | 33.7 | 117.7 | 1.000 [0.996, 1.004] | 0.998 [0.994, 1.002] | 0.983 [0.948, 1.002] | 0.980 [0.966, 0.988] | 531 |
| 8 | 33.7 | 214.8 | 0.988 [0.965, 1.003] | 1.001 [0.997, 1.009] | 1.048 [0.997, 1.183] | 0.986 [0.983, 0.988] | 1 055 |
| 16 | 37.3 | 391.4 | 0.996 [0.992, 0.998] | 1.000 [0.998, 1.004] | 0.947 [0.840, 1.004] | **1.034 [1.025, 1.039]** | 2 016 |
| 32 | 76.8 | 410.8 | 1.001 [0.999, 1.005] | **1.001 [1.001, 1.001]** | **1.006 [1.006, 1.007]** | **1.042 [1.042, 1.043]** | 3 931 |

Median step times of the policies at batch 32: LRU 76.8, score 76.9, EMA 76.7, allocation + EMA 76.3, Bélády 73.6 ms. The wide CIs of
allocation + EMA at batch 8 and 16 come from one slow round each.

## E1 status
Batch-1 step 33.1 ms against 30.3 ms for the fused batch-1 path of S7 (replication-v2), gap 2.8 ms (9 %); the old batched path
needed 58 ms. Pinned host tier, fused batched policy kernel with no host synchronisation, zero-copy fetch and write-back: done and
tested (per-request misses equal the simulator for LRU and Bélády, EMA within ±2; ranks and tokens equal the old path).
**Not met: overlap of the transfer with compute.** The side-stream version (`overlap=True`) passed the CUDA tests but produced
nondeterministic "illegal memory access" errors in long multi-run benchmarks, never under `CUDA_LAUNCH_BLOCKING=1`; the race was
not found, so every measurement here uses `overlap=False` (fetch on the main stream). Without overlap the transfer is fully
visible in the step; with a working overlap the transfer share, and with it any policy gain, could only be smaller.

## E3: upper bound from all-miss vs no-fetch (same pipeline)
| batch | no-fetch step | all-miss step (K pages fetched per request and layer) | transfer at most (ms) | ms per missed page | LRU transfer (misses × per-page) | share of LRU step |
|---|---|---|---|---|---|---|
| 1 | 32.1 | 32.9 | 0.8 | — (hidden behind launches) | — | ≈ 0 |
| 4 | 33.1 | 34.0 | 0.9 | — | — | ≈ 0 |
| 8 | 33.1 | 53.9 | 20.8 | 1.4 µs | 1.5 ms | 4 % |
| 16 | 32.8 | 103.7 | 71.0 | 2.40 µs | 4.8 ms | 11 % |
| 32 | 67.5 | 210.7 | 143.2 | 2.42 µs (26 GB/s for 64 KiB pages) | 9.5 ms | 12 % |

Upper bound of a policy's gain at batch 32 = (LRU misses − its misses) × 2.42 µs: current score 0.79 ms (1.0 %), EMA 0.90 ms
(1.2 %), allocation + EMA 1.33 ms (1.7 %), Bélády 3.19 ms (4.2 %). Measured: +0.1 %, +0.1 %, +0.6 %, +4.2 %. Bélády reaches its
bound; the score-based policies reach less, i.e. part of their saved transfer is hidden behind other work.

## E4: where the slow steps come from
Per-step tags (page-completion step: a 16-token page is finished and written back; miss-count terciles), `tails.csv`:

| | LRU median / p99 (ms) | EMA | Bélády |
|---|---|---|---|
| batch 1, steps without a page completion | 33.5 / 34.8 | 33.1 / 33.9 | 33.2 / 33.6 |
| batch 1, page-completion steps | 37.6 / 38.7 | 37.1 / 37.4 | 37.1 / 37.6 |
| batch 32, steps without a page completion | 76.7 / 80.5 | 76.7 / 80.1 | 73.7 / 76.2 |
| batch 32, page-completion steps | 92.9 / 95.2 | 92.8 / 95.2 | 89.8 / 91.4 |

Every policy has the same tail: one step in 16 completes a page (bounds, write-back of R pages per layer) and costs +4 ms at batch 1
and +16 ms at batch 32. LRU has no extra tail of its own. Replication-v2's LRU tail at batch 32 (std 23 ms vs 7 ms) came from the old
batched path (pageable host tier, per-layer synchronising copies) and the fixed run order, not from the policy (R4).

## E5, E6
E5_PLACEHOLDER

## Comparison with Xiaoyang Wu's L4 benchmark (PR #9; `external_xiaoyang.md`)
His transfer-only decomposition on an L4 (5.3 µs per page, LRU transfer share ~14 % at B=96 from batch 8 on, ceilings EMA +1.5 %,
allocation + EMA +2.1 %, Bélády +5.4 %; transfer and compute timed separately and added) and our end-to-end measurement (2.42 µs per
page with the zero-copy gather on a PCIe 4 card, transfer 12 % of LRU's step at batch 32, Bélády +4.2 %, EMA +0.1 %) agree on the
mechanism: the ceiling is the transfer share times the policy's share of the miss gap. End to end, the score policies realise
less than their ceiling.
