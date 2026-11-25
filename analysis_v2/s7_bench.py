"""S7 system measurements on one GPU (run with CUDA_VISIBLE_DEVICES set to the fixed card).

  python -m analysis_v2.s7_bench single --split splits/longctx_v1.json --subset heldout --ndocs 3 \
      --budgets 96 128 --steps 384 --repeats 3 --out out/results_v2/gpu/single_06b [--trace-dir data/heldout]
  python -m analysis_v2.s7_bench batch --split splits/longctx_v1.json --ndocs 32 --tokens 16384 \
      --batches 1 4 8 16 32 --budget 96 --steps 128 --repeats 2 --out out/results_v2/gpu/batch_06b
  python -m analysis_v2.s7_bench flops --out out/results_v2/gpu/flops.csv

single: batch 1, gpu_bench.run (fused Triton policy kernels for latency; the sync-free path with CUDA
events for the latency breakdown). Policies: dense, resident (Quest with all KV on GPU = no host
fetch), LRU, current score (EMA alpha=0), score EMA (frozen alpha), allocation + EMA (frozen
allocation), Belady (next use from the resident run). Checks: LRU and Belady misses equal the
simulator on the run's own ranks; every policy reproduces the resident run's ranks and tokens.
batch: gpu_bench_batch.run on R equal-length requests (prompts cut to --tokens by keeping the first
and last halves of the chat prompt). Checks: per-request LRU/Belady misses equal the simulator;
ranks and tokens identical across policies; for R=4, tokens vs. the same requests run alone.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def env_info():
    import torch
    import transformers
    q = subprocess.run(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                        "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True)
    load = os.getloadavg()
    others = subprocess.run(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                             "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip()
    return dict(gpu=q.stdout.strip(), torch=torch.__version__, transformers=transformers.__version__,
                host=platform.node(), loadavg=load, compute_apps_before=others,
                time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def load_prompts(split_path, subset, ndocs):
    from collect.long import build_turns, load_model, load_rows, row_file
    split = json.loads(Path(split_path).read_text())
    ps = [p for p in split["prompts"] if subset in (None, "all") or p["split"] == subset]
    ps = sorted(ps, key=lambda p: p["rid"])[:ndocs]
    tok, model = load_model(split)
    rows = load_rows(split["data"], ps, split.get("source", "longbench"))
    prompts = [(p["rid"], build_turns(tok, split, p, rows[row_file(p), int(p["row"])])[0]) for p in ps]
    return split, tok, model, prompts


def frozen(path):
    f = json.loads(Path(path).read_text())
    return {int(k): v for k, v in f["alpha"].items()}, {int(k): v for k, v in f["alloc"].items()}


def lat_summary(lat, warmup):
    x = np.asarray(lat[warmup:]) * 1e3
    return dict(n=int(x.size), mean_ms=float(x.mean()), std_ms=float(x.std()), median_ms=float(np.median(x)),
                p10_ms=float(np.percentile(x, 10)), p90_ms=float(np.percentile(x, 90)),
                p99_ms=float(np.percentile(x, 99)))


def cmd_single(a):
    import torch
    import gpu_bench as gb
    import kvpolicy as kp
    split, tok, model, prompts = load_prompts(a.split, a.subset, a.ndocs)
    if a.cut:
        prompts = [(rid, cut(p, a.cut)) for rid, p in prompts]
        a.trace_dir = None  # a cut prompt no longer matches the collected trace
    alpha, alloc = frozen(a.frozen)
    L = model.config.num_hidden_layers
    a.out.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    info = env_info()
    for rid, prompt in prompts:
        res = gb.run(model, prompt, "resident", a.steps, warmup=a.warmup)
        ref_ranks, ref_gen = res["ranks"], res["gen"]
        trace_ok = None
        if a.trace_dir:
            from collect.long import trace_name
            p = Path(a.trace_dir) / trace_name(rid)
            if p.exists():
                z = np.load(p)
                n = min(a.steps, z["ranked"].shape[0])
                trace_ok = bool(np.array_equal(z["ranked"][:n].astype(np.int64), ref_ranks[:n].astype(np.int64)))
        for mode in ("dense", "resident"):
            for rep in range(a.repeats):
                r = gb.run(model, prompt, mode, a.steps, warmup=a.warmup)
                rows.append(dict(doc=rid, B=None, policy=mode, rep=rep, misses=0, **lat_summary(r["lat"], a.warmup),
                                 peak_prefill=r["peak_prefill"], peak_decode=r["peak_mem"], mem_start=r["mem_start"],
                                 tokens_equal_resident=(r["gen"] == ref_gen) if mode == "resident" else None))
        for B in a.budgets:
            pols = [("lru", "lru", [B] * L, None), ("score", "ema", [B] * L, 0.0),
                    ("ema", "ema", [B] * L, alpha[B]), ("belady", "belady", [B] * L, None)]
            if B in alloc and len(alloc[B]) == L:
                pols.append(("alloc_ema", "ema", alloc[B], alpha[B]))
            for name, mode, budgets, al in pols:
                for rep in range(a.repeats):
                    r = gb.run(model, prompt, mode, a.steps, budgets=budgets, alpha=al, trace=ref_ranks,
                               warmup=a.warmup, zc="fused")
                    m = int(np.sum(r["misses"]))
                    steady = r["metrics"].get("misses_per_step_steady")
                    rows.append(dict(doc=rid, B=B, policy=name, rep=rep, misses=m, misses_per_step_steady=steady,
                                     pcie_bytes_per_token=(steady or 0) * 65536, **lat_summary(r["lat"], a.warmup),
                                     peak_prefill=r["peak_prefill"], peak_migration=r["peak_migration"],
                                     peak_decode_warmup=r["peak_decode_warmup"], peak_decode=r["peak_decode_steady"],
                                     mem_start=r["mem_start"], tokens_equal_resident=r["gen"] == ref_gen,
                                     ranks_equal_resident=bool(np.array_equal(r["ranks"], ref_ranks))))
                    if rep == 0 and name in ("lru", "belady", "score", "ema", "alloc_ema"):
                        sim = sum(kp.simulate(ref_ranks[:, l], None if mode != "ema" else None, budgets[l],
                                              mode)["misses"] for l in range(L)) if mode != "ema" else None
                        checks.append(dict(doc=rid, B=B, policy=name, measured=m, simulated=sim,
                                           equal=(sim == m) if sim is not None else None, trace_prefix_equal=trace_ok))
                # breakdown (sync-free path with CUDA events; separate run, events add overhead)
                if name in ("lru", "ema", "belady"):
                    r = gb.run(model, prompt, mode, a.steps, budgets=budgets, alpha=al, trace=ref_ranks,
                               warmup=a.warmup, zc=True, timing=True)
                    bd = r["metrics"].get("breakdown_steady_mean")
                    rows.append(dict(doc=rid, B=B, policy=name + "_timed", rep=0, misses=int(np.sum(r["misses"])),
                                     **lat_summary(r["lat"], a.warmup),
                                     **({f"bd_{k}_ms": v for k, v in zip(
                                         ("append", "select", "policy", "fetch", "attention"), bd)} if bd else {})))
            print(rid, B, "done", flush=True)
        torch.cuda.empty_cache()
    (a.out / "runs.json").write_text(json.dumps(dict(env=info, args={k: str(v) for k, v in vars(a).items()},
                                                     rows=rows, checks=checks), indent=1))
    print(json.dumps(checks, indent=1))


def cut(prompt, n):
    return prompt if len(prompt) <= n else prompt[: n // 2] + prompt[-(n - n // 2):]


def cmd_batch(a):
    import torch
    import gpu_bench_batch as gbb
    import kvpolicy as kp
    split, tok, model, prompts = load_prompts(a.split, "all", a.ndocs)
    alpha, alloc = frozen(a.frozen)
    L = model.config.num_hidden_layers
    P = [cut(p, a.tokens) for _, p in prompts if len(p) >= a.tokens]
    a.out.mkdir(parents=True, exist_ok=True)
    info = env_info()
    rows, checks = [], []
    page_bytes = None
    for R in a.batches:
        if R > len(P):
            break
        reqs = P[:R]
        pmax = (a.tokens + a.steps + 2) // 16 + 1
        cfg = model.config
        page_bytes = 2 * cfg.num_key_value_heads * 16 * cfg.head_dim * 2
        host = R * L * pmax * page_bytes
        if host > a.host_cap_gb * 2**30:
            rows.append(dict(R=R, skipped=f"host tier {host/2**30:.1f} GB > cap {a.host_cap_gb} GB"))
            continue
        try:
            B = a.budget
            pols = [("lru", "lru", B, 0.0), ("score", "ema", B, 0.0), ("ema", "ema", B, alpha.get(B, 0.5))]
            if B in alloc and len(alloc[B]) == L:
                pols.append(("alloc_ema", "ema", alloc[B], alpha.get(B, 0.5)))
            ref = None
            for name, mode, budgets, al in pols + [("belady", "belady", B, 0.0)]:
                for rep in range(a.repeats):
                    nxt = gbb.next_use_batch(ref["ranks"]) if mode == "belady" else None
                    r = gbb.run(model, reqs, mode, a.steps, budgets, alpha=al, nxt=nxt, warmup=a.warmup)
                    if ref is None:
                        ref = r
                    lat = r["lat"][a.warmup:]
                    m_steady = None
                    rows.append(dict(R=R, B=B, policy=name, rep=rep, tokens_per_s=R * len(lat) / lat.sum(),
                                     **lat_summary(r["lat"], a.warmup), misses=int(r["misses"].sum()),
                                     pcie_bytes_per_step=r["bytes"] / a.steps, host_tier_bytes=r["host_bytes"],
                                     peak_prefill=r["peak_prefill"], mem_start=r["mem_start"],
                                     peak_decode=r["peak_decode_steady"],
                                     ranks_equal_lru=bool(np.array_equal(r["ranks"], ref["ranks"])),
                                     tokens_equal_lru=r["gen"] == ref["gen"]))
                    if rep == 0 and mode in ("lru", "belady"):
                        bud = budgets if isinstance(budgets, list) else [budgets] * L
                        ok = all(int(r["misses"][q, l]) == kp.simulate(r["ranks"][:, q, l], None, bud[l], mode)["misses"]
                                 for q in range(R) for l in range(L))
                        checks.append(dict(R=R, policy=name, per_request_misses_equal_simulator=ok))
                    del r
                    torch.cuda.empty_cache()
            if R == 4:  # batch vs alone
                same, tot, rk = 0, 0, 0
                for q in range(R):
                    alone = gbb.run(model, [reqs[q]], "lru", a.steps, B)
                    g1, g2 = alone["gen"][0], ref["gen"][q]
                    same += sum(x == y for x, y in zip(g1, g2))
                    tot += len(g1)
                    first_div = next((i for i, (x, y) in enumerate(zip(g1, g2)) if x != y), None)
                    rk += int(np.array_equal(alone["ranks"][:, 0], ref["ranks"][:, q]))
                    checks.append(dict(R=4, request=q, tokens_equal_alone=g1 == g2, first_divergence=first_div,
                                       ranks_equal_alone=bool(np.array_equal(alone["ranks"][:, 0], ref["ranks"][:, q]))))
                checks.append(dict(R=4, share_tokens_equal_alone=same / tot, requests_ranks_equal=rk))
            print("R", R, "done", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            rows.append(dict(R=R, skipped=f"CUDA OOM: {str(e)[:200]}"))
            torch.cuda.empty_cache()
            break
    (a.out / "runs.json").write_text(json.dumps(dict(env=info, args={k: str(v) for k, v in vars(a).items()},
                                                     page_bytes=page_bytes, rows=rows, checks=checks), indent=1))
    print(json.dumps(checks, indent=1))


def cmd_flops(a):
    """Per generated token, per sequence: multiply-accumulates (MACs) of each component."""
    import csv
    models = {"Qwen3-0.6B": dict(L=28, H=16, kvh=8, d=128, hidden=1024, inter=3072, vocab=151936, params_nonemb=0.44e9),
              "Qwen3-8B": dict(L=36, H=32, kvh=8, d=128, hidden=4096, inter=12288, vocab=151936, params_nonemb=6.95e9)}
    rows = []
    for name, m in models.items():
        for N in (16384, 24576, 32768):
            for B in (96, 128):
                P = N // 16
                K = 64
                sel = m["L"] * m["H"] * P * m["d"] * 2                    # two einsums (clamped q vs hi/lo)
                attn = m["L"] * m["H"] * (K * 16 + 8) * m["d"] * 2         # QK^T and PV over K pages + tail
                dense_attn = m["L"] * m["H"] * N * m["d"] * 2
                lin = m["params_nonemb"] + m["hidden"] * m["vocab"]        # weights (1 MAC per weight) + lm_head
                ema = m["L"] * B * 3                                       # z, multiply-add per resident
                evict = m["L"] * B * np.log2(B) * 2                        # two stable sorts (comparisons)
                rows.append((name, N, B, int(lin), int(sel), int(attn), int(dense_attn), int(ema), int(evict),
                             sel / lin, attn / lin, ema / lin))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "context_tokens", "B", "linear_macs", "selector_macs", "sparse_attention_macs",
                    "dense_attention_macs", "ema_update_ops", "eviction_sort_compares", "selector_over_linear",
                    "attention_over_linear", "ema_over_linear"])
        w.writerows(rows)
    for r in rows:
        print(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["single", "batch", "flops"])
    ap.add_argument("--split", default="splits/longctx_v1.json")
    ap.add_argument("--subset", default="heldout")
    ap.add_argument("--ndocs", type=int, default=3)
    ap.add_argument("--budgets", type=int, nargs="+", default=[96, 128])
    ap.add_argument("--budget", type=int, default=96)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=384)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--frozen", default="splits/heldout_frozen_v1.json")
    ap.add_argument("--trace-dir")
    ap.add_argument("--host-cap-gb", type=float, default=80)
    ap.add_argument("--cut", type=int, help="single: cut prompts to this many tokens (context-length variation)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd != "flops":
        a.out = Path(a.out)
    dict(single=cmd_single, batch=cmd_batch, flops=cmd_flops)[a.cmd](a)


if __name__ == "__main__":
    main()
