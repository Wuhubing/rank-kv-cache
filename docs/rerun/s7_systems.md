# S7: system measurements (one RTX 6000 Ada, card 0)

**Conclusion (thesis point 2).** In a real two-tier cache the replacement policy behaves exactly as simulated. LRU and Bélády
miss counts match the simulator exactly; EMA is within 0.1 %, from float32 vs float16 scores. Score EMA moves **8–10 % fewer
host→GPU bytes** than LRU (allocation + EMA 11–20 %). It does **not** turn into lower latency or higher throughput on this machine:
- at batch 1 (fused Triton policy kernels) every tiered policy runs at 30.3–30.7 ms/token, and the differences are within the
  repeat-to-repeat spread;
- at batch 4–16 (new batched implementation, `gpu_bench_batch.py`, one merged transfer per layer) throughput scales almost
  linearly with the batch (17 → 231 tokens/s), but EMA reaches only 0.93–0.98× LRU's throughput. Its extra unfused GPU work costs more
  than the saved transfer. Even Bélády, with 30 % fewer bytes, gains only 2–3 %.

The bytes are not the bottleneck on this machine, even at batch 16, so the earlier "batch 32 saves 2.3 ms/step" extrapolation does
not hold as stated. A policy gain only matters when transfer is a large share of the step (slower links, larger KV per page or
step, more pages fetched per step), and the policy's own GPU cost must be fused to near zero.

Setup: wulab2 card 0 (`GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3`, driver 550.90.12), no other process on the card during the
measurements (checked and logged before each run in `logs/s7.log` on wulab2), torch 2.6.0+cu124, bf16, greedy. Page 16, K=64,
64 KiB per page per layer for both models. Frozen α and allocation from `splits/heldout_frozen_v1.json` (0.6B) and
`splits/heldout_frozen_8b.json` (8B). Code: `gpu_bench.py` (batch 1, unchanged since #7; needs `setuptools` for Triton),
`gpu_bench_batch.py` (new), driver `analysis_v2/s7_bench.py`, summaries `analysis_v2/s7_summarize.py`.

## 1. Batch 1, Qwen3-0.6B, 3 held-out documents (20–30K tokens), 384 decode steps, 8 warm-up steps excluded, 3 repeats
`results_v2/gpu/single_06b/` (`summary.csv`, `checks.csv`, `runs.json`).

| B | Policy | TPOT median (ms) | p90 | p99 | repeat-to-repeat std of the median | misses / step | PCIe MiB / token |
|---|---|---|---|---|---|---|---|
| – | dense (full attention, all KV on GPU) | 23.44 | 23.63 | 23.88 | 0.02 | 0 | 0 |
| – | resident Quest (all KV on GPU = no host fetch) | 26.17 | 26.89 | 28.85 | 0.02 | 0 | 0 |
| 96 | LRU | 30.33 | 31.55 | 33.95 | 0.07 | 130.2 | 8.14 |
| 96 | current score (EMA α=0) | 30.41 | 31.29 | 34.11 | 0.04 | 118.7 | 7.42 |
| 96 | score EMA (α=0.5) | 30.31 | 31.18 | 33.92 | 0.04 | 117.2 | 7.33 |
| 96 | allocation + EMA | 30.35 | 31.00 | 33.76 | 0.03 | 112.1 | 7.01 |
| 96 | Bélády | 30.56 | 31.21 | 33.97 | 0.04 | 83.2 | 5.20 |
| 128 | LRU | 30.42 | 30.95 | 33.75 | 0.04 | 77.7 | 4.86 |
| 128 | current score | 30.47 | 31.18 | 33.88 | 0.03 | 71.0 | 4.44 |
| 128 | score EMA (α=0.7) | 30.72 | 31.27 | 34.16 | 0.37 | 69.5 | 4.34 |
| 128 | allocation + EMA | 30.60 | 32.33 | 35.09 | 0.16 | 62.4 | 3.90 |
| 128 | Bélády | 30.67 | 31.84 | 34.21 | 0.04 | 47.1 | 2.94 |

Bytes vs LRU at B=96: current score −8.8 %, EMA −10.0 %, allocation + EMA −13.9 %, Bélády −36.1 %. Latency: all within ±0.4 ms of
LRU, no ordering by bytes.

**Latency breakdown** (sync-free path with CUDA events, B=96, mean GPU ms per token summed over 28 layers; events add
overhead, so these runs take ~46–50 ms/token and are only used for the split):
| Policy | append page | Quest select | policy | host fetch | attention |
|---|---|---|---|---|---|
| LRU | 1.27 | 4.97 | 12.11 | 4.69 | 5.07 |
| EMA | 1.27 | 4.97 | 14.48 | 5.58 | 5.04 |
| Bélády | 1.27 | 4.98 | 12.24 | 4.95 | 5.10 |

In the unfused path the EMA policy step costs 2.4 ms per token more than LRU's (about 85 µs per layer). That overhead is what the fused
Triton kernel removes (the latency table uses the fused path). "Host fetch" includes the one sync per layer, so it barely moves with the
number of missed pages. The no-host-fetch control (resident Quest, 26.2 ms) vs the tiered cache (30.3 ms) puts the whole cost of
offloading at about 4 ms per token at batch 1.

**Memory by phase** (B=96; GiB, `torch.cuda.max_memory_allocated` with a reset at each phase boundary): prefill 4.07; migration of
completed pages to the host tier 4.03; decode warm-up 1.84; steady decode 1.84 (tiered) vs 3.57 (resident Quest) and 3.69 (dense).
The tiered cache halves decode memory at this context length; the prefill peak is unchanged (the full prefill KV is still built
on the GPU first). Policy metadata: one float per slot (EMA) or one int (LRU/Bélády): 28 × 96 × 4 B = 10.5 KiB.

**Correctness** (`checks.csv`): in all 6 (document, budget) pairs the measured misses equal the simulator on the run's own ranks for
LRU and Bélády. EMA, current score and allocation + EMA differ from the simulator replaying the collected trace's float16 scores by
1–43 misses out of 26–48 K (≤ 0.1 %), because the live run uses float32 scores. Every policy run reproduced the
resident run's ranks and tokens exactly, and the resident run reproduced the first 384 steps of the collected held-out trace exactly
(`trace_prefix_equal`). **Generation is identical across policies**, as it must be (the policy only decides where pages live).

## 2. Batch > 1, Qwen3-0.6B (`gpu_bench_batch.py`)
Requests: the longctx_v1 prompts with ≥ 16 384 tokens, cut to exactly 16 384 tokens (first and last 8 192, so the instruction
stays). 128 decode steps, 8 warm-up excluded, 2 repeats, B=96 per request and layer. `results_v2/gpu/batch_06b/`
(`summary.csv`, `checks.csv`, `batch.png`).

| Batch | LRU tokens/s | EMA / LRU throughput | current score / LRU | alloc+EMA / LRU | Bélády / LRU | EMA / LRU bytes | Bélády / LRU bytes | LRU PCIe MiB/step | decode peak GiB | host tier GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 17.1 | 0.934 | 0.917 | 0.945 | 1.032 | 0.918 | 0.698 | 8.6 | 1.6 | 1.8 |
| 4 | 63.9 | 0.913 | 0.937 | 0.936 | 0.959 | 0.919 | 0.695 | 37.1 | 2.8 | 7.1 |
| 8 | 119.8 | 0.977 | 0.953 | 0.951 | 1.027 | 0.918 | 0.701 | 69.5 | 4.4 | 14.1 |
| 16 | 231.3 | 0.935 | 0.971 | 0.973 | 1.020 | 0.916 | 0.700 | 137.6 | 7.6 | 28.2 |
| 24, 32 | (running) | | | | | | | | | |

- The batched implementation is **not fused**: the policy is ~30 vectorised PyTorch ops per layer with one host sync, and the
  missed pages are gathered on the CPU. Its step time at batch 1 is 58 ms, against 30 ms for the fused batch-1 path. So the
  absolute numbers are a lower bound on what an optimised system gets. The comparison between policies inside it is fair for transfer
  (same code path) and pessimistic for EMA (more unfused ops than LRU). Bélády has the same op count as LRU, so its column isolates
  the transfer effect: 30 % fewer bytes give +2–3 % throughput at batch 8–16.
- Step time grows only from 58 ms (batch 1) to 68 ms (batch 16) while bytes per step grow 16×. Transfer is a small part of the step
  at every batch size we could run.
- **Correctness:** every request's LRU and Bélády misses inside the batch equal the simulator on that request's ranks (all batch
  sizes); ranks and tokens are identical across policies within a batch. **Batch vs alone:** the 4 requests of batch 4, run
  alone, generate the same first 19–105 tokens and then diverge for 3 of 4 (55 % of tokens equal over 128 steps; the ranks differ from
  the first steps). Batched kernels change the bf16 numerics, which changes near-tie selections and, later, the greedy token. This is
  a property of batched inference with a data-dependent selector, not of the cache.

## 3. Where a policy could pay off (pending: B=80/256, 8B, FLOPs)
- Context length (0.6B, prompts cut to 8 192 tokens, B=96): dense 19.2 ms, resident Quest 26.1 ms, tiered 30.4 ms (all policies
  within 0.5 ms; EMA −11 % bytes). At 8K on a 0.6B model, sparse selection costs more than it saves; offloading only buys memory.
