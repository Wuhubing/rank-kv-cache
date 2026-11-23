# GPU latency benchmark: two-tier Quest KV cache (Colab L4, 2026-11-23)

**Result in one line:** in a real host-offloaded cache the replacement policy reaches the
simulator's miss counts exactly. It does not reduce decode latency, though: PCIe transfer is not
the bottleneck here. Score EMA cuts host→GPU traffic by 9–10 % but adds about 6.5 ms per token of
policy work, so it is slower than LRU in this implementation. Even Bélády, with 34 % fewer misses
than LRU, is no faster than LRU.

## Setup
- Model: Qwen3-0.6B (bf16, 28 layers, 8 KV heads, head dim 128), HF transformers 4.53.3, PyTorch
  SDPA, greedy decoding, on an NVIDIA L4 with 52 GB host RAM.
- Inputs: 3 held-out documents (2 gov_report, 1 qmsum; prompts 20–30K tokens), with the same prompt
  as the traces. 384 decode steps after the prefill; the first 8 steps are warm-up. Latency is the
  wall time of one full model step (`torch.cuda.synchronize` → next token on the CPU).
- Two-tier cache (`gpu_bench.py`):
  - After the prefill, every completed 16-token page of every layer is moved to pinned host memory
    and dropped from the GPU (a page is 64 KiB of K+V).
  - Each layer has B GPU page slots, filled on demand from a cold start (the simulator's semantics).
  - Quest min/max bounds stay on the GPU.
  - Each step and layer: score all pages, take the top-K (K=64), the policy picks victims among
    unselected residents (vectorised on the GPU, with the same keys as `kvpolicy.py`), missing pages
    are gathered on the host into a pinned staging buffer and copied to the GPU, then attention runs
    over the K pages plus the incomplete tail page.
  - Copies are synchronous: one sync per layer to learn the miss IDs, with no prefetch or overlap.
- Baselines:
  - "resident": the same Quest attention with the whole KV on the GPU (no offload).
  - "dense": full attention with the whole KV on the GPU.
  - Bélády: the tiered cache using next-use times from the trace.
- Policies: LRU, score EMA (α frozen per budget: 0.5 at B=96, 0.7 at B=128), and allocation + EMA
  (the frozen dev allocation in `splits/heldout_frozen_v1.json`).

## Correctness
- **Rankings and tokens:** the tiered runs reproduce the captured trace exactly: the same top-K
  pages at every step and layer (rank_eq = 1.0 on all 24 tiered runs) and the same generated tokens.
  Offloading copies the bits exactly and keeps the page order.
- **Miss counts vs the simulator** (`kvpolicy.simulate` on the same 384 steps):
  - LRU and Bélády match exactly on every run.
  - EMA and allocation + EMA differ by at most 39 of 45K misses (<0.1 %). The trace stores scores
    as fp16, while the live run uses fp32, so a few near-ties break differently.

## Latency (median of 3 documents; p90 within +4 to +8 ms of the median)
| Config | ms/token, B=96 | ms/token, B=128 | PCIe MB/token, B=96 | PCIe MB/token, B=128 |
|---|---|---|---|---|
| Resident Quest (no offload) | 57.1 | – | 0 | – |
| Dense attention (whole KV on GPU) | 68–97 (depends on length) | – | 0 | – |
| Tiered LRU | 86.6 | 84.5 | 8.50 | 5.34 |
| Tiered score EMA | 93.1 | 89.5 | 7.71 | 4.81 |
| Tiered allocation + EMA | 93.0 | 90.3 | 7.38 | 4.31 |
| Tiered Bélády (oracle) | 87.3 | – | 5.61 | – |

Peak GPU memory during decode is 3.5–5.1 GB in every config. The prefill and the 0.6B model
dominate it, so the offload's memory saving (about 2–3 GB of KV) does not show in this peak.

## Where the time goes (CUDA events, attention backend summed over 28 layers, B=96, mean ms/step)
| Phase | LRU | EMA | What it is |
|---|---|---|---|
| append + page completion | 2.4 | 2.4 | tail update, bounds of completed pages, write-back to host |
| Quest scoring + top-K | 11.9 | 11.9 | same work as the resident path |
| policy (victim choice, incl. 1 sync/layer) | 20.2 | 25.1 | about 15 small kernels per layer + `mp.cpu()` |
| transfer (host gather + H2D) | 11.4 | 12.3 | CPU `index_select` into staging + copy |
| attention | 10.3 | 10.3 | SDPA over K·16 + tail tokens |

(Event timing adds about 7 ms per step, so these rows sum to more than the gap in the latency
table.)

**Transfer is not bandwidth-bound.** The measured pinned host→GPU bandwidth is 12.2 GB/s. LRU's
8.5 MB per token is therefore about 0.7 ms of PCIe time, yet the transfer phase takes 11.4 ms. Over
90 % of it is fixed per-layer cost: the CPU gather, kernel launches and the sync. The same holds for
the policy phase: its cost is kernel-launch and sync latency (28 layers × about 15 launches), not
arithmetic. Score EMA saves about 0.8 MB per token (about 0.07 ms of PCIe) but adds about 5 ms of
launches (the EMA update over B slots, the z normalisation, a second sort key). Bélády saves 2.9 MB
per token (about 0.24 ms) and its latency stays within 1 ms of LRU, which confirms that misses
hardly move latency in this regime.

## What this means
- The policy's benefit is real in bytes: 9–10 % less PCIe traffic for EMA and 13–19 % for
  allocation + EMA on these documents, consistent with the held-out miss reductions. At this scale
  it does not turn into latency, because the per-layer fixed overheads of an unfused PyTorch
  implementation (about 30 ms per token over resident) dwarf the 0.5–0.7 ms of actual PCIe time.
- For miss reduction to matter for latency, transfer has to be a large share of the step:
  - bigger KV per page or step (a larger model, batching, or longer K);
  - slower links (shared PCIe, or CPU memory on another NUMA node);
  - or a fused implementation where the fixed overheads are gone.

  With fused kernels the EMA update is one elementwise pass over B ≤ 256 floats per layer
  (microseconds), so its overhead should disappear and EMA's byte saving should come through. This
  is not measured here.
- "Near-zero overhead" holds for the algorithm (O(B) floats of state, one multiply-add per resident
  per step) but **not** for this unfused PyTorch implementation, where EMA costs about 6.5 ms per
  token more than LRU (+7.5 %).

## Limitations
- One small model (0.6B) on one GPU type; batch size 1; 3 documents × 384 steps.
- No compute/transfer overlap or prefetch. HiSparse-style systems overlap transfers, which would
  hide even more of the PCIe time.
- The Python/HF per-layer dispatch dominates; a CUDA-graph or fused-kernel version was not built.
- Held-out documents were used only for timing. No policy parameter was tuned on them.

## Files
- `gpu_bench.py`: the tiered cache, the baselines and the timing harness.
- On Drive, `MyDrive/rank-kv-cache/results_gpu_benchmark/`: `bench.pkl` (per-step latencies,
  per-layer misses and the breakdown for every run), `bench_summary.csv`, and a copy of
  `gpu_bench.py`.
- `distill_kvrl.py` remains untracked and unused.

## Follow-up: sync-free zero-copy path and launch-free GPU cost (same day)
To check whether the per-layer sync and the CPU gather were hiding the policy effect,
`gpu_bench.run(..., zc=True)` adds a second tiered implementation:
- No host sync: victims are assigned with fixed-size (K) tensor ops.
- A Triton kernel reads only the missed pages directly from mapped pinned host memory over PCIe
  (zero-copy, about 12 GB/s in a microbenchmark), with no staging copy.

**Correctness** (doc gov_report/00dfb, 128 steps): identical tokens and rankings. LRU 19,538 and
Bélády 13,604 misses equal the simulator exactly; EMA is 17,929 vs 17,937 in the simulator, and
allocation + EMA is 17,380 vs 17,377.

**Latency is unchanged** (LRU 89.1, EMA 95.5, allocation + EMA 95.5, Bélády 89.4 ms/token). The
sync was not the bottleneck: the step is bound by **CPU dispatch**, meaning Python and HF overhead
plus about 60 small kernel launches per layer.

**GPU-busy time per step** (sum of all kernel and copy durations from `torch.profiler`, prefill
subtracted, 40 steps, B=96). This approximates the cost of a launch-free (CUDA-graph or fused)
implementation:

| Config | GPU ms/step | Misses/step |
|---|---|---|
| Resident Quest (HF DynamicCache) | 35.0 | 0 |
| Tiered LRU | 16.7 | 194 |
| Tiered score EMA | 17.4 | 184 |
| Tiered allocation + EMA | 17.8 | 182 |
| Tiered Bélády | 16.3 | 155 |

What these numbers show:
- **The resident baseline is inflated by HF itself.** It does more GPU work than the tiered cache
  because HF's `DynamicCache` concatenates the whole KV every step (about 3.4 GB copied per step).
  Its 57 ms wall time is still lower only because it launches fewer kernels. A static-cache
  resident baseline would be the fair comparison; it was not built.
- **A miss is cheap on the GPU.** Bélády removes 39 misses per step and saves 0.45 ms, about 11.5 µs
  per 64 KiB page. Score EMA removes about 10 misses per step (about 0.12 ms) but its extra
  unfused elementwise kernels cost more (net +0.7 ms GPU). A single fused EMA kernel per layer
  (28 × a few µs) would put EMA at roughly break-even at this scale.
- **Conclusion at batch 1 / 0.6B / L4:** even launch-free, the best possible policy (Bélády) is
  worth only about 3 % of GPU time, and EMA's share of that is under 1 %. Misses only matter where
  transfer is a large share of the step. KV bytes per page and layer do not grow with model width
  under 8-head GQA, so the relevant lever is **batching** (each sequence brings its own misses). At
  batch 32, LRU would move about 400 MB per step (about 33 ms at 12 GB/s), and EMA's 9–10 % byte
  saving would be about 3 ms per step. This is an extrapolation; it was not measured.

Latency wall times above are wall-clock with `torch.cuda.synchronize`; GPU-busy times come from one
profiled run per config on one document, so treat them as indicative (±0.2 ms).
