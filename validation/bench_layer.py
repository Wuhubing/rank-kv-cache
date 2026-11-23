"""Paired CUDA layer microbenchmark with explicit warmup and raw samples.

python -m validation.bench_layer --out validation/results/layer_benchmark.json
Measures the shipped synchronous backend on synthetic correlated queries.
This is one attention layer, not end-to-end Qwen latency or a batching result.
"""
import argparse
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import gpu_bench as gb


@torch.inference_mode()
def bandwidth():
    host = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    device = torch.empty_like(host, device="cuda")
    for _ in range(3):
        device.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    durations = []
    for _ in range(10):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        device.copy_(host, non_blocking=True)
        stop.record()
        stop.synchronize()
        durations.append(start.elapsed_time(stop))
    return {"bytes_per_copy": host.numel(), "event_ms": durations,
            "median_GB_per_s": host.numel() / (np.median(durations) * 1e6)}


@torch.inference_mode()
def benchmark(steps, repeats, seed, pages, warmup):
    if not 0 <= warmup < steps or pages < 128 or repeats < 2:
        raise ValueError("require 0 <= warmup < steps, pages >= 128 and at least two repeats")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dtype, kvh, dim, heads = torch.bfloat16, 8, 128, 16
    module = SimpleNamespace(layer_idx=0)
    initial = pages * gb.PS
    # Reference resident storage is preallocated; no full-history concatenation.
    full_k = torch.randn(1, kvh, initial + steps, dim, device="cuda", dtype=dtype)
    full_v = torch.randn_like(full_k)
    q = torch.randn(steps, 1, heads, 1, dim, device="cuda", dtype=torch.float32)
    for t in range(1, steps):
        q[t] = 0.98 * q[t - 1] + (1 - 0.98**2)**0.5 * q[t]
    q = q.to(dtype)
    pk = full_k[0, :, :initial].reshape(kvh, pages, gb.PS, dim).transpose(0, 1).cpu()
    pv = full_v[0, :, :initial].reshape(kvh, pages, gb.PS, dim).transpose(0, 1).cpu()
    reference = gb.Bench("resident")
    reference.rb, reference.rec = {}, [None]
    ranks = []
    for t in range(steps):
        reference.quest_resident(module, q[t], full_k[:, :, :initial+t+1],
                                 full_v[:, :, :initial+t+1], None, dim**-0.5)
        ranks.append(reference.rec[0].cpu().numpy())
    future = gb.next_use(np.array(ranks)[:, None])
    cases = [(mode, b) for b in (96, 128) for mode in ("lru", "ema", "belady")]
    cases += [("static_resident", None)]
    rows = []
    for repeat in range(repeats):
        order = rng.permutation(len(cases))
        for case in order:
            mode, budget = cases[case]
            bench = gb.Bench(mode)
            if budget is None:
                bench.rb = {}
            else:
                pmax = pages + steps // gb.PS + 2
                tier = gb.Tier(pk, pv, full_k[0, :, initial:initial].clone(),
                               full_v[0, :, initial:initial].clone(), budget, pmax, mode,
                               0.5 if budget == 96 else 0.7, None)
                bench.tiers[0] = tier
                bench.nxt_all = future
            latencies, misses = [], []
            for t in range(steps):
                bench.step, bench.rec, bench.evs = t, [None], []
                prior = 0 if budget is None else tier.misses
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                if budget is None:
                    bench.quest_resident(module, q[t], full_k[:, :, :initial+t+1],
                                         full_v[:, :, :initial+t+1], None, dim**-0.5)
                else:
                    bench.quest_tiered(module, q[t], full_k[:, :, initial+t:initial+t+1],
                                       full_v[:, :, initial+t:initial+t+1], None, dim**-0.5)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter_ns() - start) / 1e6)
                misses.append(0 if budget is None else tier.misses - prior)
                assert np.array_equal(bench.rec[0].cpu().numpy(), ranks[t])
            steady = np.array(latencies[warmup:])
            rows.append(dict(repeat=repeat, policy=mode, B=budget,
                             median_ms=float(np.median(steady)), p95_ms=float(np.percentile(steady, 95)),
                             mean_ms=float(steady.mean()),
                             steady_misses=sum(misses[warmup:]),
                             steady_h2d_bytes=sum(misses[warmup:]) * 65536,
                             step_latency_ms=latencies, step_misses=misses))
            bench.tiers.clear()
    summary = []
    for mode, budget in cases:
        subset = [r for r in rows if (r["policy"], r["B"]) == (mode, budget)]
        summary.append(dict(policy=mode, B=budget,
                            median_of_repeat_medians_ms=float(np.median([r["median_ms"] for r in subset])),
                            min_repeat_median_ms=min(r["median_ms"] for r in subset),
                            max_repeat_median_ms=max(r["median_ms"] for r in subset),
                            median_steady_misses=float(np.median([r["steady_misses"] for r in subset]))))
    return dict(steps=steps, repeats=repeats, warmup_excluded=warmup, seed=seed,
                pages=pages, page_bytes=65536, query_ar1_rho=0.98, summary=summary, runs=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=192)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--pages", type=int, default=512)
    ap.add_argument("--seed", type=int, default=614)
    ap.add_argument("--warmup", type=int, default=16)
    args = ap.parse_args()
    result = {"environment": {"platform": platform.platform(), "torch": torch.__version__,
                               "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda},
              "scope": "Single-layer synthetic wall-time microbenchmark, serialized CPU/GPU, BF16; excludes model and prefill.",
              "bandwidth": bandwidth(),
              "benchmark": benchmark(args.steps, args.repeats, args.seed, args.pages, args.warmup)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"bandwidth_GB_per_s": result["bandwidth"]["median_GB_per_s"],
                      "summary": result["benchmark"]["summary"]}, indent=2))


if __name__ == "__main__":
    main()
