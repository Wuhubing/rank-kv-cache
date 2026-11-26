# Batch scaling of the page-transfer cost (L4, Colab)

**What kind of number this is.** A GPU-time model: decode, scoring and page transfer are each timed separately on the
L4 and added. It excludes CPU launch gaps, the cost of the policy's own resolver kernel (measured separately at batch 1:
+0.084 ms per token for EMA over LRU, docs/gpu_benchmark.md), and any overlap of transfer with compute. It is one run;
run-to-run variance was not measured. Use it for the batch trend and the ceiling, not as an end-to-end token/s claim.

Notebook [`notebooks/batch_transfer_bench.ipynb`](../notebooks/batch_transfer_bench.ipynb), run once on a Colab
NVIDIA L4 (torch 2.14.1+cu130, Triton 3.8.0). Raw output: [`results/batch_bench_l4/`](results/batch_bench_l4/).

**Setup.** Per-step, per-layer miss counts of the frozen held-out policies on the 21 MPS held-out documents
(`miss_schedule.npz`, exported by `local/export_miss_schedule.py`, totals equal to `results/mps_heldout/misses.csv`).
Sequence j of a batch replays document j mod 21 from a staggered start; 256 measured steps after 16 warm-up steps.
Each step and layer copies its missed 64 KiB pages from pinned host memory into GPU slots with the zero-copy Triton
gather of `gpu_bench.py`. Compute side at the same batch: one Qwen3-0.6B decode step attending to 1,088 tokens
(K = 64 pages + tail) and the Quest scoring step over 1,600 pages per layer. All times are GPU kernel time summed
with `torch.profiler` (launch gaps excluded); transfer is not overlapped with compute.

**Transfer is bandwidth-bound and linear in misses.** Reference memcpy 12.35 GB/s; the gather moves a page in
5.32–5.48 µs from batch 16 up (about 12 GB/s), 5.32–5.86 µs from batch 4 up and 6.0–6.7 µs at batch 1. So every saved
miss saves about 5.3 µs per step at large batch.

## Results (GPU time per decode step)

| B | batch | decode + scoring | LRU transfer (share of step) | saved vs LRU: EMA | alloc + EMA | Bélády | step speedup: EMA / alloc + EMA / Bélády |
|---|---|---|---|---|---|---|---|
| 96 | 1 | 9.2 + 1.2 ms | 0.77 ms (6.8%) | 0.08 ms | 0.09 ms | 0.26 ms | +0.7% / +0.9% / +2.4% |
| 96 | 8 | 19.6 + 12.9 ms | 5.0 ms (13.4%) | 0.52 ms | 0.72 ms | 1.84 ms | +1.4% / +2.0% / +5.1% |
| 96 | 32 | 70.9 + 48.8 ms | 19.5 ms (14.0%) | 2.00 ms | 2.81 ms | 7.10 ms | +1.5% / +2.1% / +5.4% |
| 96 | 64 | 132.9 + 97.7 ms | 38.2 ms (14.2%) | 3.87 ms | 5.48 ms | 13.77 ms | +1.5% / +2.1% / +5.4% |
| 128 | 1 | 9.2 + 1.2 ms | 0.47 ms (4.3%) | 0.05 ms | 0.08 ms | 0.18 ms | +0.5% / +0.8% / +1.7% |
| 128 | 32 | 70.9 + 48.8 ms | 11.5 ms (8.8%) | 1.27 ms | 2.43 ms | 4.66 ms | +1.0% / +1.9% / +3.7% |
| 128 | 64 | 132.9 + 97.7 ms | 22.6 ms (8.9%) | 2.41 ms | 4.66 ms | 8.89 ms | +1.0% / +1.9% / +3.6% |

Full table (batch 1, 4, 8, 16, 32, 64): `results/batch_bench_l4/batch_bench.csv`.

## What this shows

1. **Fewer misses do become time, linearly.** Saved transfer time equals saved misses × ~5.3 µs at every batch.
   At batch 32, B=96, EMA saves 2.0 ms per step and allocation + EMA 2.8 ms; the earlier estimate from the batch-1
   benchmark (7.3 µs × saved misses × batch, docs/STATUS_REPORT.md §9.6) predicted about 2.3 ms for EMA.
2. **But the transfer share saturates.** Decode and scoring also grow linearly with batch, so the LRU transfer share
   rises from 7% (batch 1) to a plateau of about 14% at B=96 (9% at B=128) from batch 8 on. The step speedup therefore
   also plateaus: EMA +1.5%, allocation + EMA +2.1%, and even the Bélády oracle only +5.4% (B=96).
3. **The ceiling is set by the transfer share, not by the policy.** Allocation + EMA reaches 39% of Bélády's time
   saving at B=96 (2.1 of 5.4 points) and 52% at B=128, close to its share of the miss gap (G 0.40 / 0.53).
4. **Sensitivity to the scoring cost.** Our scoring step is a plain PyTorch einsum over fp32 page bounds and is
   memory-bound (49 ms at batch 32). If scoring were free, the LRU transfer share at batch 64 would be 22% (B=96) and
   the speedups EMA +2.3%, allocation + EMA +3.3%, Bélády +8.8%; at B=128: +1.6% / +3.1% / +6.1%.
5. **Where it would matter more:** a larger transfer share — slower host links, larger pages or K, no overlap with
   compute, or smaller models per KV byte. With a prefetch that fully overlaps transfer and compute (transfer < compute
   here at every batch), the saving would vanish. We did not measure overlap.

**Scope.** One GPU (L4, PCIe gen4 x16 as provisioned by Colab), Qwen3-0.6B, page and K of the trace study, synthetic
compute inputs of the right shape; transfers and compute are timed separately and added, not run as one fused decoder.
The CPU launch cost of 28 gather launches per step (wall 3–43 ms) is excluded; at batch 1 it dominates, as the
full-model batch-1 benchmark found (docs/gpu_benchmark.md).
