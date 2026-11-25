# S7: system measurements (one RTX 6000 Ada, card 0)

**Conclusion (thesis point 2).** In a real two-tier cache the replacement policy behaves exactly as simulated. LRU and Bélády
miss counts match the simulator exactly; EMA is within 0.1 %, from float32 vs float16 scores. Score EMA moves **8–10 % fewer
host→GPU bytes** than LRU (allocation + EMA 11–20 %). On this machine that **mostly does not** turn into lower latency or higher
throughput:
- at batch 1 (fused Triton policy kernels) every tiered policy runs at 30.3–30.7 ms/token (0.6B) or 40.6 ms (8B), and the
  differences are within the repeat-to-repeat spread;
- at batch 4–24 (new batched implementation, `gpu_bench_batch.py`, one merged transfer per layer) throughput scales almost linearly
  with the batch (17 → 313 tokens/s on 0.6B), but EMA reaches only 0.90–1.00× LRU's throughput. Its extra unfused GPU work costs about
  as much as the saved transfer. Even Bélády, with 30 % fewer bytes, gains only 0–7 %;
- **at batch 32 on 0.6B a small, reproducible gain appears**: EMA +3.4 %, allocation + EMA +4.1 %, Bélády +4.8 % tokens/s over LRU
  (two repeats agree within 0.3 tokens/s). It comes from fewer slow steps (mean step 94.1 vs 97.3 ms; the median step is the same);
- at batch 32 on 8B (12K-token prompts) the step is compute-bound (148 ms) and every policy is within the ±2.5 % noise.

The earlier "batch 32 saves 2.3 ms/step" extrapolation is withdrawn: the measured effect at batch 32 is about 3 ms per step on 0.6B
(+3.4 % throughput), and zero on 8B. A policy gain needs transfer to be a large share of the step (slower links, larger KV per page,
more pages per step, larger batch on a small model), and the policy's own GPU cost must be fused to near zero.

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
| 24 | 312.9 | 0.997 | 1.008 | 1.016 | 1.066 | 0.915 | 0.699 | 206.2 | 10.8 | 42.4 |
| 32 | 329.0 | **1.034** | 1.033 | **1.041** | **1.048** | 0.915 | 0.698 | 271.7 | 14.1 | 56.5 |

(Batch 24 and 32 come from a separate run, `batch_06b_r32/`, over the first 32 of the 33 longctx_v1 prompts with ≥ 16 384 tokens;
the batch-1–16 rows use the first 31 such prompts among the first 32 documents. At batch 24 the two LRU repeats disagree, 330 vs 296
tokens/s, so that row is not conclusive. At batch 32 every policy's two repeats agree within 1.5 tokens/s.)

**Where the batched step goes** (`batch_06b_timed/`, CUDA events, 64 steps, 1 repeat; GPU ms per step summed over 28 layers):
| Batch | Policy | append | select | policy | fetch (gather + H2D + scatter) | attention | step (ms) |
|---|---|---|---|---|---|---|---|
| 1 | LRU | 1.6 | 6.8 | 15.5 | 6.5 | 7.5 | 62.0 |
| 1 | EMA | 1.5 | 6.7 | 18.4 | 7.5 | 7.4 | 65.5 |
| 16 | LRU | 1.9 | 8.1 | 15.1 | 12.4 | 21.1 | 69.5 |
| 16 | EMA | 1.8 | 8.1 | 18.4 | 13.0 | 21.2 | 73.5 |
| 16 | Bélády | 1.9 | 8.1 | 15.3 | 10.9 | 21.1 | 68.3 |

At batch 16 the fetch phase is ~18 % of the step, and Bélády's 24 % fewer bytes save 1.5 ms of it. The EMA's unfused policy step costs
3.3 ms more than LRU's, and its fetch phase is not measurably shorter (13.0 vs 12.4 ms in this single repeat, for 7 % fewer bytes). (The allocation + EMA timed run at batch 16 had one slow
outlier, with fetch 30.6 ms in a single repeat, and is not used.)

- The batched implementation is **not fused**: the policy is ~30 vectorised PyTorch ops per layer with one host sync, and the
  missed pages are gathered on the CPU. Its step time at batch 1 is 58 ms, against 30 ms for the fused batch-1 path. So the
  absolute numbers are a lower bound on what an optimised system gets. The comparison between policies inside it is fair for transfer
  (same code path) and pessimistic for EMA (more unfused ops than LRU). Bélády has the same op count as LRU, so its column isolates
  the transfer effect: 30 % fewer bytes give +2–3 % throughput at batch 8–16.
- Step time grows only from 58 ms (batch 1) to 68 ms (batch 16) while bytes per step grow 16×; at batch 32 it reaches 93 ms and the
  byte saving starts to show in throughput (+3–5 %).
- **Correctness:** every request's LRU and Bélády misses inside the batch equal the simulator on that request's ranks (all batch
  sizes); ranks and tokens are identical across policies within a batch. **Batch vs alone:** the 4 requests of batch 4, run
  alone, generate the same first 19–105 tokens and then diverge for 3 of 4 (55 % of tokens equal over 128 steps; the ranks differ from
  the first steps). Batched kernels change the bf16 numerics, which changes near-tie selections and, later, the greedy token. This is
  a property of batched inference with a data-dependent selector, not of the cache.

## 3. Where could a policy pay off? (B/K, context length, 0.6B vs 8B)
**Budget (0.6B, batch 1, 3 documents, 2 repeats; `single_06b_b80_256/`):**
| B (B/K) | LRU ms/token | EMA ms/token | Bélády ms/token | LRU MiB/token | EMA bytes vs LRU | Bélády bytes vs LRU |
|---|---|---|---|---|---|---|
| 80 (1.25) | 30.10 | 30.13 | 30.92 | 11.85 | −8.2 % | −30.0 % |
| 96 (1.5) | 30.33 | 30.31 | 30.56 | 8.14 | −10.0 % | −36.1 % |
| 128 (2) | 30.42 | 30.72 | 30.67 | 4.86 | −10.6 % | −39.5 % |
| 256 (4) | 30.45 | 31.00 | 30.60 | 1.84 | −7.6 % | −28.8 % |

Latency does not move with bytes across a 6× range of PCIe traffic (1.8–11.9 MiB/token). LRU and Bélády misses equal the
simulator in all 12 (document, budget, policy) pairs.

**Context length (0.6B, prompts cut to 8 192 tokens, B=96; `single_06b_8k/`):** dense 19.2 ms, resident Quest 26.1 ms, tiered
30.4 ms (all policies within 0.5 ms; EMA −11 % bytes). At 8K on a 0.6B model the sparse selection costs more than it saves, and
offloading only buys memory.

**Model size (Qwen3-8B, 2 held-out documents, 256 steps, 2 repeats, B=96; `single_8b/`):**
| Policy | ms/token | misses/step | MiB/token | decode peak GiB |
|---|---|---|---|---|
| dense | 48.89 | – | – | 17.70 |
| resident Quest | 34.60 | – | – | 17.51 |
| LRU | 40.56 | 96.5 | 6.03 | 16.02 |
| current score | 40.57 | 88.9 | 5.56 | 16.02 |
| score EMA (α=0.6, frozen on 8B dev) | 40.55 | 86.9 | 5.43 | 16.02 |
| allocation + EMA | 40.64 | 78.3 | 4.89 | 16.03 |
| Bélády | 40.85 | 61.4 | 3.84 | 16.02 |

On 8B, sparse attention clearly pays off (resident Quest 29 % faster than dense), and offloading costs 6 ms per token. The
replacement policy is again a tie (all within 0.3 ms), and the 8B weights (16 GB) dominate decode memory. Breakdown (unfused
timed path, ms per token): LRU policy 15.9 / fetch 6.1; EMA 19.0 / 7.3; Bélády 16.1 / 6.5. LRU, Bélády and current score match the
simulator exactly in both documents, EMA within 11 misses of ~20–29 K, and every run reproduced the collected 8B held-out trace prefix.

**Batch on 8B (prompts cut to 16 384 tokens, 64 steps, 2 repeats; `batch_8b/`):**
| Batch | LRU tokens/s | EMA / LRU | current score / LRU | alloc+EMA / LRU | Bélády / LRU | EMA bytes / LRU | Bélády bytes / LRU | LRU MiB/step | decode peak GiB | host tier GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 14.1 | 0.904 | 0.943 | 0.939 | 1.034 | 0.937 | 0.800 | 7.4 | 15.9 | 2.3 |
| 4 | 49.4 | 0.963 | 0.960 | 0.907 | 1.001 | 0.946 | 0.818 | 28.4 | 17.4 | 9.0 |
| 8 | 100.0 | 0.899 | 0.887 | 0.932 | 1.000 | 0.945 | 0.819 | 57.3 | 19.5 | 18.1 |
| 16 | 173.9 | 0.975 | 0.934 | 0.978 | 1.002 | 0.948 | 0.821 | 114.7 | 23.6 | 36.2 |

Only 25 of the 35 8B documents reach 16 384 tokens, which limits this table to batch 16. With prompts cut to 12 288 tokens
(`batch_8b_12k/`, all 35 documents qualify): batch 1 / 16 / 32 LRU 13.8 / 176.4 / 209.0 tokens/s; EMA 0.946 / 0.955 / 0.984×,
allocation + EMA 0.914 / 0.959 / 0.996×, Bélády 1.026 / 1.005 / 1.014× (repeat spread at batch 32 ≈ ±2.5 %); step 72 / 89 / 148 ms;
decode peak 15.8 / 22.5 / 29.7 GiB; host tier 1.7 / 27.2 / 54.4 GiB. At batch 32 the 8B step is compute-bound, and no policy differs
from LRU beyond the noise. In this 64-step window the 8B per-request saving is smaller than on 0.6B (EMA −5 %, Bélády −18 % bytes:
a short decode, so a larger share of misses is compulsory). Every request's LRU/Bélády misses equal the simulator. Batch vs alone (batch 4):
3 of 4 requests generate identical tokens over 64 steps (94 % of tokens equal), although the ranks differ.

**FLOPs/MACs per generated token and sequence** (`flops.csv`, analytic from the model configs):
| Model, context | linear layers + lm_head | Quest selector | sparse attention (K pages + tail) | dense attention | EMA update | eviction sort |
|---|---|---|---|---|---|---|
| 0.6B, 16K | 0.60 G | 0.12 G (20 %) | 0.12 G (20 %) | 1.88 G | 8.1 K ops | 35 K compares |
| 0.6B, 32K | 0.60 G | 0.23 G (39 %) | 0.12 G | 3.76 G | 8.1 K | 35 K |
| 8B, 16K | 7.57 G | 0.30 G (4 %) | 0.30 G (4 %) | 4.83 G | 10.4 K | 46 K |
| 8B, 32K | 7.57 G | 0.60 G (8 %) | 0.30 G | 9.66 G | 10.4 K | 46 K |

The replacement policy's arithmetic is about 10⁻⁵ of the model's. Its cost in practice is kernel launches and synchronisation (unfused:
+2.4 ms/token on 0.6B, +3.1 ms on 8B; fused: ≈ 0), not FLOPs. The selector, not the policy, is the expensive part of sparse attention
on a small model.

## 4. Characterisation (thesis point 2)
The miss and byte savings of a better replacement policy turn into latency or throughput only when **host→GPU transfer is a large
share of the decode step**. On a PCIe 4.0 RTX 6000 Ada, with 0.6B and 8B Qwen3, 16–32K contexts, B/K = 1.25–4 and batch 1–24, it
is not. At batch 1 the step is bound by kernel dispatch (the whole offloading path costs 4–6 ms per token, of which transfer is a
fraction), and with batching the step time grows far more slowly than the bytes; even the optimal policy (Bélády, 18–40 % fewer
bytes) is within a few % of LRU. The first measurable gain appears at **batch 32 on the 0.6B model** (EMA +3.4 %, allocation + EMA
+4.1 %, Bélády +4.8 % throughput), where 272 MiB move per step; on 8B the larger compute hides it again. Conditions under which this could change, none of which we measured: a slower or
contended host link, much larger KV per page (MHA instead of 8-head GQA, larger head dim), many more pages fetched per step (larger
K, smaller B/K), or a fully fused/asynchronous offload path whose remaining cost *is* the transfer.
