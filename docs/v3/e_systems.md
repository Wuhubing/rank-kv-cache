# E: system measurements, redone (fixed batched path, randomised order, ≥ 5 rounds)

**Conclusion (0.6B, PCIe RTX 6000 Ada): fewer misses become throughput only in proportion to transfer's share of the step, and
that share is small.** With batch ≤ 8 the decode step is bound by kernel launches (~33 ms for every policy; even fetching all K pages
per request and layer costs nothing at batch 1–4). At batch 32 transfer is 11 % of LRU's step (9.0 ms of 77 ms): Bélády, with 37 %
fewer misses, gains +4.2 % [4.2, 4.3] throughput, the EMA +0.1 % [0.1, 0.1], allocation + EMA +0.6 % [0.6, 0.7], the current score
+0.1 %. Bélády reaches ≈ 97 % of the E3 upper bound (saved misses × 2.50 µs per missed page = 3.28 ms, 4.3 % of LRU's median
step; measured 3.17 ms), the EMA stays far below its bound (0.92 ms, 1.2 %; measured 0.06 ms). Replication-v2's batch-32 "+3.4 %" is gone with the fixed path and the randomised order: LRU no longer has a long tail; the
slow steps of every policy are the page-completion steps. Larger pages (32, 64 tokens), K = 128 and Qwen3-8B (16K / 12K
prompts) keep transfer at 5–10 % of the step at the largest batch, so Bélády gains +1.5 to +3.7 % and the EMA at most +0.8 % (8B:
−0.5 % at batch 16, −0.2 % at 12K batch 32). Only in the simulated slow tier (E5) does transfer become a large share (21–69 %), and
there every policy's speedup follows its miss reduction (EMA up to +7.2 %, allocation + EMA +11.3 %, Bélády +33.6 %).
(Numbers revised 2026-11-27 after Xiaoyang Wu's review of PR #11, `review_response.md`: misses per step now count the timed steps
only — the earlier summary divided the total over all 264 steps, warm-up included, by the 256 timed steps.)

Setup: `gpu_bench_batch2.py` (E1) with `overlap=False` (see "E1 status"), driver `analysis_v3/e_bench.py`, summary
`analysis_v3/e_summarize.py`. wulab2 card 0 (GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3). No other process was on the card in any
run of the 0.6B configurations, 8B batch 16 and 8B 12K batch 32 (checked before and after every run). In the 8B batch 1 and 4 runs
(all rounds) and two batch-8 runs, other processes were present on card 0 (PIDs in `raw/*.json.gz`); these rows are marked
(`clean=False` in `throughput.csv`) and not used for conclusions. Host load average 3.4 at the start of the configuration (the precondition),
2–16 during runs (this process's CPU threads plus other users; recorded per run). GPU 64–73 °C, SM 2.5–2.7 GHz. Requests: the
longctx_v1 prompts with ≥ 16 384 tokens in rid order, cut to 16 384 tokens, nested across batch sizes. B=96, frozen α (0.5) and
allocation (`splits/heldout_frozen_v1.json`). Per batch: a preparation LRU run (simulator check, Bélády's next uses), then 5 rounds,
each a discarded warm-up run and every policy once in a random order (seed = round + 1000·batch); 264 steps per run, 8 warm-up
steps excluded (256 timed). Ratios: mean throughput of the policy / LRU, paired by round, bootstrap CI over rounds. Every LRU
run's per-request misses equal the simulator; ranks and tokens are identical across policies (not checked for "nofetch": `n/a`).
Misses per step = mean over the 256 timed steps of a run, then over rounds.
Results: `results_v3/e/batch_06b/` (`summary/throughput.csv`, `tails.csv`, `d2_model.csv`; raw per-step data `raw/*.json.gz`).

## E2: throughput relative to LRU (0.6B, 16K-token prompts, B=96)
| batch | LRU median step (ms) | LRU tokens/s | current score | EMA | allocation + EMA | Bélády | LRU misses / step |
|---|---|---|---|---|---|---|---|
| 1 | 33.1 | 29.8 | 1.009 [0.997, 1.029] | 1.008 [0.998, 1.026] | 1.010 [1.000, 1.029] | 1.005 [0.995, 1.023] | 128 |
| 4 | 33.7 | 117.7 | 1.000 [0.996, 1.004] | 0.998 [0.994, 1.002] | 0.983 [0.948, 1.002] | 0.980 [0.966, 0.988] | 487 |
| 8 | 33.7 | 214.8 | 0.988 [0.965, 1.003] | 1.001 [0.997, 1.009] | 1.048 [0.997, 1.183] | 0.986 [0.983, 0.988] | 968 |
| 16 | 37.3 | 391.4 | 0.996 [0.992, 0.998] | 1.000 [0.998, 1.004] | 0.947 [0.840, 1.004] | **1.034 [1.025, 1.039]** | 1 841 |
| 32 | 76.8 | 410.8 | 1.001 [0.999, 1.005] | **1.001 [1.001, 1.001]** | **1.006 [1.006, 1.007]** | **1.042 [1.042, 1.043]** | 3 583 |

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
| 8 | 33.1 | 53.9 | 20.8 | 1.45 µs | 1.4 ms | 4 % |
| 16 | 32.8 | 103.7 | 71.0 | 2.48 µs | 4.6 ms | 12 % |
| 32 | 67.5 | 210.7 | 143.2 | 2.50 µs (26 GB/s for 64 KiB pages) | 9.0 ms | 12 % |

ms per missed page = (transfer at most) / (all-miss misses per timed step, = K·R·L). The no-fetch run counts LRU's misses but
fetches and writes back nothing, so its counted misses are not subtracted from the denominator. (Before the R11-1 fix the
denominator was the all-step total / 256: 2.42 µs at batch 32.)

Upper bound of a policy's gain at batch 32 = (LRU misses − its misses) × 2.50 µs: current score 0.80 ms (1.0 % of LRU's median
step), EMA 0.92 ms (1.2 %), allocation + EMA 1.37 ms (1.8 %), Bélády 3.28 ms (4.3 %). Measured median-step saving −0.08, 0.06,
0.48 and 3.17 ms (throughput +0.1 %, +0.1 %, +0.6 %, +4.2 %). Bélády reaches ≈ 97 % of its bound; the score-based policies reach
much less (the policy's own cost explains part of the gap: "Net gain" below).

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

## GPU memory (R11-5; from `raw/*.json.gz`: `peak_decode` = steady-state decode peak allocated, `host_bytes` = pinned host tier)
LRU, max over rounds (GiB; the other policies differ by ≤ 0.25 GiB, Bélády's next-use table being the largest addition):

| config | batch | decode peak GPU memory | host tier |
|---|---|---|---|
| 0.6B page 16 | 32 | 14.17 | 56.93 |
| 0.6B page 32 | 32 | 16.41 | 56.98 |
| 0.6B page 64 | 32 | 26.18 | 57.09 |
| 0.6B K 128 | 32 | 19.86 | 56.93 |
| 8B 16K | 16 | 23.76 | 36.60 |
| 8B 12K | 32 | 29.88 | 55.20 |

(`peak_decode_gib`, `host_tier_gib` in `throughput.csv`.) Memory by phase (prefill, migration, decode warm-up, steady state) was
not recorded in these runs; it is measured with the R11-7 timing runs ("Net gain" below).

## E5 (simulated slow tier): speedup against the transfer share — **simulation, not a measurement of a real slower link**
A GPU-side spin on the fetch path adds (missed bytes of the layer) × d ns per byte before each fetch, read on the GPU without a
host sync (`gpu_bench_batch2._spin`). Effective bandwidth = 64 KiB / (D2 per-miss cost). The d values bracket the published link
speeds: PCIe 4.0 ×16 ≈ 25–32 GB/s (our gather reaches ≈ 26 GB/s), a PCIe 4.0 NVMe SSD ≈ 7 GB/s sequential (vendor read
specifications, e.g. ~7.4 GB/s), 100 GbE ≈ 12.5 GB/s and 25 GbE ≈ 3.1 GB/s line rate. `results_v3/e/slow_06b/`; 5 rounds,
randomised order, batch 8 and 32.

| batch 32, d (ns/B) | effective bandwidth (D2) | LRU transfer share (D2) | LRU median step | EMA | allocation + EMA | Bélády |
|---|---|---|---|---|---|---|
| 0 (measured, E2) | 26.4 GB/s | 11.4 % | 76.8 ms | 1.001 [1.001, 1.001] | 1.006 [1.006, 1.007] | 1.042 [1.042, 1.043] |
| 0.04 | 12.9 GB/s | 20.9 % | 86.2 ms | 1.013 [1.012, 1.013] | 1.023 [1.022, 1.023] | 1.081 [1.080, 1.082] |
| 0.12 | 6.3 GB/s (≈ NVMe) | 35.2 % | 104.8 ms | 1.029 [1.028, 1.030] | 1.047 [1.047, 1.048] | 1.146 [1.146, 1.147] |
| 0.28 | 3.1 GB/s (≈ 25 GbE) | 52.2 % | 142.6 ms | 1.050 [1.050, 1.051] | 1.080 [1.080, 1.081] | 1.235 [1.234, 1.236] |
| 0.6 | 1.6 GB/s | 68.7 % | 218.5 ms | 1.072 [1.071, 1.072] | 1.113 [1.113, 1.114] | 1.336 [1.335, 1.337] |

The speedup rises monotonically with the transfer share for every policy, and it is what the miss reductions predict: with misses
cut by 10.3 % (EMA), 15.3 % (allocation + EMA) and 36.7 % (Bélády) and transfer share s, the D2 model gives 1 / (1 − s·r), e.g.
at s = 0.687: 1.076 / 1.118 / 1.338 against measured 1.072 / 1.113 / 1.336 (`summary/speedup_curve.csv`). At batch 32 the D2
prediction matches the simulated slow-tier speedups to within ~1 percentage point for the score-based policies (measured
consistently 0.4–1.0 points below the prediction, the largest gaps at 21–35 % transfer share: EMA −0.95 / −0.87 / −0.64 / −0.45,
allocation + EMA −1.03 / −0.96 / −0.66 / −0.41 points for d = 0.04 / 0.12 / 0.28 / 0.6, consistent with the policy's own cost) and
to within 0.25 points for Bélády (also below: −0.22 / −0.23 / −0.25 / −0.17). Break-even: the EMA's throughput gain has a CI above
0 already at the measured 11.4 % share (+0.1 %), but it exceeds 1 % only from a ~19 % share and 5 % from a ~52 % share (linear
interpolation of the measured batch-32 EMA ratios against the D2 share). At batch 8 the results are noisier (launch-bound; e.g. EMA 1.051 [1.039, 1.061] and Bélády 1.229 at d = 0.6).
(In the slow-tier runs the "nofetch" reference still pays the simulated delay for its bookkeeping misses, so it is not a
zero-transfer reference there; the transfer share comes from the D2 fit.) No unsimulated slowdown (concurrent H2D traffic on the
same link) was measured.

## E6: the policy's own GPU time (CUDA events, `results_v3/e/timing_06b/`, 2 rounds, 64 timed steps)
Per decode step, summed over 28 layers (ms):

| batch, policy | append | selection (Quest scores + top-K) | **policy kernels** | fetch + index | attention |
|---|---|---|---|---|---|
| 1, LRU / EMA / Bélády | 1.11 / 1.11 / 1.13 | 5.2 | **2.94 / 2.99 / 3.16** | 5.2–5.3 | 3.9 |
| 16, LRU / EMA / Bélády | 0.68 | 6.0 | **0.95 / 1.40 / 1.07** | 5.9 / 5.3 / 5.0 | 18.5–18.6 |

At batch 16 the fused policy step costs 34 µs per layer for LRU and 50 µs for the EMA (+0.45 ms per decode step, 1.4 % of the
GPU time); at batch 1 the phase is inflated by launch gaps (the GPU waits for the CPU). The arithmetic is tiny (the EMA update is
≈ 3·B operations per layer and request, `results_v2/gpu/flops.csv`: 10⁻⁵ of the model's), so the measured cost is kernel latency, not
FLOPs. (Event timing adds overhead; these runs are separate from the throughput runs.)

## E5 (measured): larger pages, larger K, the 8B model (5 rounds each, randomised order)
`results_v3/e/{ps32_06b,ps64_06b,k128_06b,batch_8b,batch_8b_12k}/`, `results_v3/e/rest/summary/`. Throughput relative to LRU at
the largest batch, with the D2 transfer share:

| config (B = 96 pages unless stated) | batch | LRU median step | LRU transfer share | EMA | allocation + EMA | Bélády |
|---|---|---|---|---|---|---|
| 0.6B, page 16, K 64 (E2) | 32 | 76.8 ms | 11.4 % | 1.001 [1.001, 1.001] | 1.006 [1.006, 1.007] | 1.042 [1.042, 1.043] |
| 0.6B, page 32 (α 0.5, no allocation) | 32 | 125.9 ms | 9.7 % | 1.005 [1.004, 1.005] | – | 1.037 [1.037, 1.037] |
| 0.6B, page 64 | 32 | 223.7 ms | 5.7 % | 1.004 [1.004, 1.004] | – | 1.024 [1.023, 1.024] |
| 0.6B, K 128, B 192 | 32 | 133.9 ms | 9.3 % | 1.008 [1.008, 1.008] | – | 1.037 [1.037, 1.037] |
| 8B, 16K prompts (frozen 8B α and allocation) | 16 | 77.9 ms | 5.2 % | 0.995 [0.994, 0.996] | 1.000 [0.996, 1.003] | 1.015 [1.014, 1.016] |
| 8B, 12K prompts | 32 | 134.7 ms | 6.2 % | 0.998 [0.998, 0.998] | 1.004 [1.003, 1.004] | 1.020 [1.020, 1.020] |

At batch 1 every configuration is launch-bound (0.6B 33 ms, 8B 59 ms for every policy). K 128 at batch 1 had one slow round that
widens the CIs; medians are unaffected (`throughput.csv`). The 8B batch 4 anomaly (all non-LRU policies 0.86–0.87 with CIs reaching
1.0) is not a slow round of the policies: in every 8B batch-1 and batch-4 run other processes were on card 0 (`clean=False`), so
these rows are interference and not used (clean re-run below: "8B batch 1 and 4, re-run on a clean card"). Larger pages move more bytes per miss but also make attention more expensive, so the share does not
rise; on 8B the larger compute dilutes transfer further.

## 8B batch 1 and 4, re-run on a clean card (R11-6)
`results_v3/e/batch_8b_rerun/` (`summary/throughput.csv`); same settings as `batch_8b` (`heldout_frozen_8b.json`, 16K prompts,
5 rounds, random order, `overlap=False`), 2026-11-28 00:01–00:44Z on card 3 (GPU-850490f2-9baf-0ef9-bf4e-a84a10d01f50; card 0 was
taken by another user's job when the re-run started; same card model). Every run waited for a card without other processes
(`--wait-clean`); one batch-4 LRU run had another process at its end and was repeated (kept as `discarded_runs`); all 70 kept runs
are clean, and LRU's misses equal the simulator.

| batch | LRU median step | LRU tokens/s | current score | EMA | allocation + EMA | Bélády | D2 transfer share |
|---|---|---|---|---|---|---|---|
| 1 (re-run) | 44.8 ms | 21.2 | 1.007 [0.977, 1.047] | 1.021 [0.998, 1.058] | 1.005 [0.968, 1.052] | 1.014 [0.989, 1.055] | ≈ 0 |
| 4 (re-run) | 48.0 ms | 81.2 | 0.982 [0.946, 1.002] | 0.984 [0.943, 1.012] | 1.015 [0.989, 1.054] | 0.989 [0.980, 0.999] | ≈ 0 |
| 1 (old, interfered) | 59.1 ms | 16.9 | 1.023 | 0.991 | 0.990 | 1.007 | – |
| 4 (old, interfered) | 69.7 ms | 65.5 | 0.870 | 0.865 | 0.864 | 0.871 | – |

The interference had slowed every step by 25–45 % (LRU 59.1 → 44.8 ms at batch 1, 69.7 → 48.0 ms at batch 4) and produced the
batch-4 anomaly; on a clean card no policy is at 0.86–0.87. At batch 1 and 4 the 8B step is launch-bound like the 0.6B one (D2
per-miss coefficient ≈ 0; fetching all K pages every step costs 1.0 ms at batch 1 and 13.0 ms at batch 4, while LRU fetches
4.6 % of that), so the ratios are noise around 1
(CIs of ±2–5 %). The old `batch_8b` batch-1/4 rows stay in `results_v3/e/rest/summary/throughput.csv` with `clean=False` and are
not used.

## Break-even and the P5 verdict
- Upper bound: holds in every measured (unsimulated) configuration within noise — the median-step saving of no policy exceeds
  (saved misses) × (per-page cost) by more than 0.5 % (largest: page 64 Bélády, 5.31 ms against a 5.29 ms bound); Bélády reaches
  ≈ 97 % of it at batch 32 (0.6B, page 16).
- Speedup rises monotonically with the transfer share (E5 simulated, every policy) and is proportional to the byte reduction: at
  batch 32 the D2 prediction is within ~1 percentage point for the score-based policies (measured 0.4–1.0 points below) and within
  0.25 points for Bélády.
- Break-even for the EMA (throughput CI lower bound > 0): already at the measured 11.4 % share at batch 32 (+0.1 %), but a gain of
  ≥ 1 % needs a share of ~19 %, ≥ 5 % about 52 % — reached here only by simulation (≈ 3 GB/s effective bandwidth).
- On this PCIe 4 single GPU, with 0.6B or 8B, pages 16–64 and K 64–128: transfer is not on the critical path at batch ≤ 8 and is
  6–11 % of the step at batch 32, so the policies' byte savings (10–20 % for EMA and allocation + EMA) become ≤ 1 % throughput.

## Comparison with Xiaoyang Wu's L4 benchmark (PR #9; `external_xiaoyang.md`)
His transfer-only decomposition on an L4 (5.3 µs per page, LRU transfer share ~14 % at B=96 from batch 8 on, ceilings EMA +1.5 %,
allocation + EMA +2.1 %, Bélády +5.4 %; transfer and compute timed separately and added) and our end-to-end measurement (2.50 µs per
page with the zero-copy gather on a PCIe 4 card, transfer 11 % of LRU's step at batch 32, Bélády +4.2 %, EMA +0.1 %) agree on the
mechanism: the ceiling is the transfer share times the policy's share of the miss gap. End to end, the score policies realise
less than their ceiling.
