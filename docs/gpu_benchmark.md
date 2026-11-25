# GPU latency benchmark: two-tier Quest KV cache (Colab L4, 2026-11-23)

**Summary.** In a real host-offloaded two-tier cache the replacement policy reproduces the
simulator: LRU and Bélády miss counts match exactly, EMA within 39 of about 45K misses (fp16 trace
scores). It does not reduce decode latency:
- **Final (fused Triton kernels, last section):** all four tiered policies run at about 62 ms per
  token at B=96 (LRU 62.3, EMA 62.3, allocation + EMA 61.6, Bélády 62.1), within the run-to-run
  noise of about 1.5 ms. EMA's policy kernel costs about 0.08 ms per token more GPU time than LRU's,
  which its transfer saving roughly cancels. **EMA ties LRU in latency; it is neither slower nor
  faster.**
- **Bytes:** score EMA moves 9.3–9.8 % fewer host→GPU bytes than LRU (B=96 / B=128), and
  allocation + EMA 13–19 %. At batch 1 on a 0.6B model this is worth about 0.07 ms per token.
- **Why no latency win:** the step is bound by CPU dispatch of the HF model, not by PCIe. Even
  Bélády, with 34 % fewer misses than LRU, is not measurably faster.
- **Earlier unfused implementation** (the sections up to "Follow-up", kept for the record): there
  EMA was about 6.5 ms per token slower than LRU. That overhead was kernel launches and syncs of
  the unfused PyTorch policy step, and it is gone in the fused version.
- Savings at larger batch (for example batch 32) are **linear extrapolations** from single-sequence
  measurements. Batching is not implemented in `gpu_bench.py`, so they are not measurements.

**Measurement caveats (these numbers have not been re-measured since the fixes):**
- *Warm-up.* The setup below excludes the first 8 decode steps as warm-up, but when these runs were
  made `gpu_bench.run` ignored its `warmup` argument and returned every step's latency and the miss
  total over all steps. Whether the outer Colab harness sliced them off is not recorded, so the
  latencies and miss counts may include the warm-up steps. `run` now validates `warmup` and returns
  raw and steady-state (warm-up excluded) metrics side by side (`metrics`, `warmup_mask`,
  `step_miss`).
- *Memory.* The reported decode peak (3.5–5.1 GB) was read after `reset_peak_memory_stats`, but at
  that point the prefill model output still referenced the full prefill cache, so for the tiered
  modes the peak includes the prefill KV. `run` now releases it first and reports the prefill,
  migration, warm-up decode and steady decode peaks separately.
- The two-tier cache still materialises the full prefill cache on the GPU before offloading, so it
  does not show support for prompts whose prefill does not fit on the GPU.

## Setup
- Model: Qwen3-0.6B (bf16, 28 layers, 8 KV heads, head dim 128), HF transformers 4.53.3, PyTorch
  SDPA, greedy decoding, on an NVIDIA L4 with 52 GB host RAM.
- Inputs: 3 held-out documents (2 gov_report, 1 qmsum; prompts 20–30K tokens), with the same prompt
  as the traces. 384 decode steps after the prefill; the first 8 steps are meant as warm-up (see the
  caveats above: the benchmark function did not apply this itself). Latency is the
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

## Latency, unfused implementation (median of 3 documents; p90 within +4 to +8 ms of the median)
| Config | ms/token, B=96 | ms/token, B=128 | PCIe MB/token, B=96 | PCIe MB/token, B=128 |
|---|---|---|---|---|
| Resident Quest (no offload) | 57.1 | – | 0 | – |
| Dense attention (whole KV on GPU) | 68–97 (depends on length) | – | 0 | – |
| Tiered LRU | 86.6 | 84.5 | 8.50 | 5.34 |
| Tiered score EMA | 93.1 | 89.5 | 7.71 | 4.81 |
| Tiered allocation + EMA | 93.0 | 90.3 | 7.38 | 4.31 |
| Tiered Bélády (oracle) | 87.3 | – | 5.61 | – |

Peak GPU memory during decode was reported as 3.5–5.1 GB in every config, so the offload's memory
saving (about 2–3 GB of KV) did not show in this peak. For the tiered modes this peak still included
the prefill cache, which the benchmark kept alive into decode (see the caveats at the top); it needs
re-measuring with the fixed harness before any memory comparison.

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

## What this means (unfused implementation; see the fused section for the final numbers)
- The policy's benefit is real in bytes: 9.3–9.8 % less PCIe traffic for EMA and 13–19 % for
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
- `bench_metrics.py`: warm-up handling and raw vs steady-state metrics used by `gpu_bench.run`.
- On Drive, `MyDrive/rank-kv-cache/results_gpu_benchmark/`: `bench.pkl` (per-step latencies,
  per-layer misses and the breakdown for every run), `bench_summary.csv`, and a copy of
  `gpu_bench.py`.

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
  batch 32, LRU would move about 400 MB per step (about 33 ms at 12 GB/s), and EMA's 9.3–9.8 %
  byte saving would be about 3 ms per step. This is a linear extrapolation from one sequence;
  batching is not implemented in `gpu_bench.py`, and contention or transfer overlap could change it.

Latency wall times above are wall-clock with `torch.cuda.synchronize`; GPU-busy times come from one
profiled run per config on one document, so treat them as indicative (±0.2 ms).

## Fused kernels (same day): the policy overhead is gone, and the policy still does not change latency
`gpu_bench.run(..., zc="fused")` replaces the per-layer policy step with two Triton kernels plus one
gather kernel:
- **`_policy`** (one CTA per layer):
  - EMA update over all slots;
  - victim ranking by pairwise comparison of the (prim, last, hit, −page) keys over B ≤ 256 slots;
  - assignment of misses to victims;
  - next per-slot state.
- **`_slotup`**: a write-only update of the page→slot table.
- **`_gather2`**: K and V pages read zero-copy from pinned host memory, missed pages only.

Bugs found and fixed while building it:
- **Race in the first version.** It updated the slot tables in place inside the kernel. Triton may
  re-issue a load after a store, so about 15 % of replays of one captured state produced duplicate
  target slots. On allocation layers (B=88/112) this caused an illegal memory access (Xid 43) that
  killed the Colab kernel.
- **Fix.** The per-slot state is now ping-ponged between two buffers: the kernel reads set A and
  writes set B. The page→slot update moved into a separate write-only kernel whose two address sets
  are disjoint.
- **Check.** After the fix, two full passes of all four policies are bit-identical, and a 300-replay
  stress test of the failing state passed.

**Correctness (3 held-out docs × 384 steps, all configs):**
- Rankings and tokens are identical to the trace (rank_eq = 1.0 on every run).
- LRU and Bélády miss counts equal the simulator exactly.
- EMA and allocation + EMA are within 39 misses of about 45K (the trace stores fp16 scores).

**Wall-clock latency** (median ms/token, mean over 3 documents; LRU and EMA at B=96 run twice):

| Config | B=96 | B=128 | vs unfused (B=96) |
|---|---|---|---|
| Resident Quest (no offload, HF DynamicCache) | 57.3 | – | – |
| Tiered LRU | 62.3 | 62.0 | 86.6 → 62.3 |
| Tiered score EMA | 62.3 | 61.7 | 93.1 → 62.3 |
| Tiered allocation + EMA | 61.6 | 61.7 | 93.0 → 61.6 |
| Tiered Bélády (oracle) | 62.1 | – | 87.3 → 62.1 |

p90 is 1.5–2.5 ms above the median, and repeat runs differ by up to about 1.5 ms. **All four tiered
policies are within noise of one another**, so EMA is no longer slower than LRU. Even the oracle is
not measurably faster, because the step is still bound by CPU dispatch of the HF model (about 60
ms per token of Python and launches for a 0.6B model).

**GPU-busy time per decode step** (profiler started after the prefill and 16 warm-up steps;
32 profiled steps; doc gov_report/00dfb; B=96):

| Config | GPU ms/step | `_policy` | `_slotup` | `_gather2` (H2D) | Misses/step |
|---|---|---|---|---|---|
| Resident Quest | 36.32 | – | – | – | 0 |
| Tiered LRU, fused | 14.86 | 0.806 | 0.038 | 0.901 | 194 |
| Tiered score EMA, fused | 14.87 | 0.890 | 0.038 | 0.835 | 184 |
| Tiered allocation + EMA, fused | 14.90 | 0.948 | 0.038 | 0.813 | 182 |
| Tiered Bélády, fused | 14.72 | 0.970 | 0.038 | 0.613 | 155 |
| Tiered LRU, unfused (zc) | 16.94 | | | | 194 |
| Tiered score EMA, unfused (zc) | 17.60 | | | | 184 |

What the GPU-busy numbers show:
- **EMA overhead is now near zero.** EMA's policy kernel costs +0.084 ms per token over LRU
  (28 layers, about 3 µs per layer), against +0.66 ms unfused. **"Near-zero overhead" now holds in
  the implementation**, not just on paper.
- **Transfer time scales with misses** at about 7.3 µs per 64 KiB page (zero-copy gather). EMA saves
  0.066 ms per token of gather time, allocation + EMA 0.088 ms, and Bélády 0.29 ms.
- **Net GPU effect at batch 1:** EMA ≈ LRU (−0.07 transfer + 0.08 policy, within ±0.01 ms);
  allocation + EMA is +0.04 ms; Bélády is −0.14 ms. All are ≤ 1 % of the 14.9 ms GPU step.
- **The offloaded cache uses much less GPU time than the resident path** (14.9 vs 36.3 ms), mostly
  because HF's DynamicCache copies the whole KV every step. It still has a higher wall time, because
  both paths are bound by CPU dispatch.

**Verdict.** With fused kernels the score-EMA policy costs the same as LRU on the GPU and moves
9.3–9.8 % fewer bytes (13–19 % with allocation). At batch 1 on a 0.6B model, that byte saving is worth only about 0.07 ms per
token, so the policy is a tie with LRU in latency, not a win. A latency win needs transfer to be a
real fraction of the step: large batch or long K, a slower host link, or no prefetch overlap. The
measured per-page cost (7.3 µs) and the measured miss reductions let that be estimated directly:
saving per step ≈ 7.3 µs × misses saved × batch. For EMA that is about 10 misses × batch; a
linear extrapolation to batch 32 gives about 2.3 ms per step. Batching is not implemented, so this
is an estimate, not a measurement.

Data, all in `MyDrive/rank-kv-cache/results_gpu_benchmark/`:
- `bench_fused.pkl`: per-step latencies, per-layer misses, simulator counts and tokens for every
  run;
- `bench_fused_summary.csv`;
- `gpu_busy_fused.json`;
- `gpu_bench.py` (copy).

A local copy of the summary files is in `results_gpu/` (the per-step pickle is on Drive only).
