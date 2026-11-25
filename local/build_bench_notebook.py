"""Generate notebooks/batch_transfer_bench.ipynb:  python local/build_bench_notebook.py notebooks/batch_transfer_bench.ipynb"""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Batch scaling of the KV page-transfer cost (layer-level benchmark)

**Question.** In a HiSparse-style two-tier cache (all KV pages in pinned host memory, B page slots per layer on the GPU), score EMA and per-layer allocation cut page misses by 12–17% against LRU. At batch 1 that saved no time (docs/gpu_benchmark.md). Does it at larger batch?

**What this measures, on the Colab GPU:**
1. **Transfer.** For batch b, every decode step and layer copies that step's missed 64 KiB pages (K and V of 16 tokens × 8 KV heads × 128 dims, bf16) from pinned host memory into GPU slots with the zero-copy Triton gather of `gpu_bench.py`. The number of misses per (sequence, step, layer) is replayed from the real held-out traces (`miss_schedule.npz`, 21 documents, frozen policies: LRU, score EMA, allocation + EMA, Bélády; B = 96 and 128). Sequence j of a batch replays document j mod 21 with a staggered start step.
2. **Compute.** GPU time of one Qwen3-0.6B decode step at batch b that attends to 1,088 tokens per sequence (K = 64 pages + a 64-token tail, i.e. what Quest attends to), plus the Quest page-scoring step over 1,600 pages per layer.

Both are **GPU kernel times** summed with `torch.profiler` (launch gaps excluded), i.e. what an implementation without launch overhead (CUDA graphs / fused kernels) would see. Transfers are not overlapped with compute (worst case for the cache, best case for a better policy).

**Run:** Runtime → Change runtime type → **L4** (A100/T4 also work; record which). Upload `miss_schedule.npz` (from `docs/results/mps_heldout/` on branch `xiaoyang/score-ema-final`) to `MyDrive/rank-kv-cache/` or when the upload prompt appears. Then **Runtime → Run all** (≈ 10–20 min). Results go to `MyDrive/rank-kv-cache/batch_bench/<GPU>-<time>/` (or a download if Drive is not mounted).
""")

code(r"""
import json, math, os, time, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import triton
import triton.language as tl
assert torch.cuda.is_available(), "Select a GPU runtime"
GPU = torch.cuda.get_device_name(0)
print(GPU, "| torch", torch.__version__, "| triton", triton.__version__, "| CUDA", torch.version.cuda)
BATCHES = [1, 4, 8, 16, 32, 64]
STEPS, WARMUP = 256, 16          # measured decode steps per configuration
PS, KVH, DIM, K, L = 16, 8, 128, 64, 28
N_EL = PS * KVH * DIM            # elements of one page's K (or V)
POOL = 8192                      # host pages in the pinned pool (2 x 256 MiB)
""")

code(r"""
# Miss schedule: upload or Drive
SCHED = Path("/content/miss_schedule.npz")
OUT_ROOT = None
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT_ROOT = Path("/content/drive/MyDrive/rank-kv-cache/batch_bench")
    if not SCHED.exists() and Path("/content/drive/MyDrive/rank-kv-cache/miss_schedule.npz").exists():
        SCHED = Path("/content/drive/MyDrive/rank-kv-cache/miss_schedule.npz")
except Exception as e:
    print("Drive not mounted:", e)
if not SCHED.exists():
    from google.colab import files
    up = files.upload()
    SCHED = Path("/content") / next(iter(up))
Z = np.load(SCHED)
RIDS = [str(r) for r in Z["rids"]]
BUDGETS = [int(b) for b in Z["budgets"]]
POLICIES = ["lru", "ema", "alloc_ema", "belady"]
SCHEDULE = {(p, B): [Z[f"{i}/{p}@{B}"].astype(np.int32) for i in range(len(RIDS))] for p in POLICIES for B in BUDGETS}
for B in BUDGETS:
    tot = {p: sum(int(m.sum()) for m in SCHEDULE[p, B]) for p in POLICIES}
    print(f"B={B}: misses over {len(RIDS)} documents", tot,
          {p: f"{1 - tot[p] / tot['lru']:+.1%}" for p in POLICIES})
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = (OUT_ROOT or Path("/content/batch_bench")) / f"{GPU.replace(' ', '_')}-{stamp}"
OUT.mkdir(parents=True, exist_ok=True)
print("results ->", OUT)
""")

code(r"""
# Zero-copy gather (gpu_bench.py _gather2): each program copies one 1024-element block of one page's K and V.
@triton.jit
def _gather2(ksrc, vsrc, kdst, vdst, page_ptr, slot_ptr, n_el: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    j = tl.program_id(1)
    p = tl.load(page_ptr + i).to(tl.int64)
    d = tl.load(slot_ptr + i).to(tl.int64)
    offs = j * BLOCK + tl.arange(0, BLOCK)
    tl.store(kdst + d * n_el + offs, tl.load(ksrc + p * n_el + offs))
    tl.store(vdst + d * n_el + offs, tl.load(vsrc + p * n_el + offs))

class HostPtr:
    # Under UVA a pinned host pointer is valid on the device, so Triton can read it directly.
    def __init__(self, t): self.t, self.dtype = t, t.dtype
    def data_ptr(self): return self.t.data_ptr()

kh = torch.randn(POOL, N_EL, dtype=torch.bfloat16).pin_memory()
vh = torch.randn(POOL, N_EL, dtype=torch.bfloat16).pin_memory()
KH, VH = HostPtr(kh), HostPtr(vh)
gen = torch.Generator(device="cuda").manual_seed(0)

def launch(src, dst, kd, vd):
    if len(src):
        _gather2[(len(src), N_EL // 1024)](KH, VH, kd, vd, src, dst, N_EL, 1024)

def gather(n, kd, vd, nslots):
    src = torch.randint(0, POOL, (n,), device="cuda", generator=gen)
    dst = torch.randperm(nslots, device="cuda", generator=gen)[:n]
    launch(src, dst, kd, vd)
    return src, dst

def kernel_ms(fn, n=1, name=None):
    # GPU kernel time per call, summed from torch.profiler (excludes launch gaps); optionally one kernel name
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n): fn()
        torch.cuda.synchronize()
    tot = 0.0
    for e in prof.key_averages():
        if name is None or name in e.key:
            tot += getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
    return tot / 1e3 / n

# correctness: copied pages equal the host rows
kd = torch.empty(512, N_EL, dtype=torch.bfloat16, device="cuda"); vd = torch.empty_like(kd)
src, dst = gather(300, kd, vd, 512); torch.cuda.synchronize()
assert torch.equal(kd[dst].cpu(), kh[src.cpu()]) and torch.equal(vd[dst].cpu(), vh[src.cpu()])
print("gather check OK")

# reference bandwidths: one large pinned->device memcpy, and the gather at 1,024 pages
big = torch.empty(256 << 20, dtype=torch.uint8).pin_memory(); dbig = torch.empty_like(big, device="cuda")
for _ in range(3): dbig.copy_(big, non_blocking=True)
e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
e0.record(); [dbig.copy_(big, non_blocking=True) for _ in range(10)]; e1.record(); torch.cuda.synchronize()
MEMCPY_GBS = 10 * big.numel() / (e0.elapsed_time(e1) / 1e3) / 1e9
kd = torch.empty(4096, N_EL, dtype=torch.bfloat16, device="cuda"); vd = torch.empty_like(kd)
src, dst = gather(1024, kd, vd, 4096)
GATHER_US_PER_PAGE = kernel_ms(lambda: launch(src, dst, kd, vd), 20, "_gather2") * 1e3 / 1024
print(f"memcpy {MEMCPY_GBS:.1f} GB/s | gather {GATHER_US_PER_PAGE:.2f} us per 64 KiB page "
      f"= {65536 / GATHER_US_PER_PAGE / 1e3:.1f} GB/s")
del big, dbig
""")

code(r"""
def batch_counts(policy, B, b, steps):
    # misses[t, l] summed over b sequences; sequence j replays document j % n from a staggered start
    docs = SCHEDULE[policy, B]
    n = len(docs)
    tot = np.zeros((steps, L), np.int64)
    for j in range(b):
        m = docs[j % n]
        start = (37 * j) % m.shape[0]
        idx = (start + np.arange(steps)) % m.shape[0]
        tot += m[idx]
    return tot

def time_transfer(counts, seed=0):
    # Indices are drawn before timing (random host pages; distinct, scattered GPU slots per launch),
    # so only the gather kernels run in the timed loop. Returns gather GPU kernel ms per step and wall ms per step.
    rng = np.random.default_rng(seed)
    b_slots = int(counts.max()) + K
    flat = counts.reshape(-1)
    src_all = torch.from_numpy(rng.integers(0, POOL, int(flat.sum()))).cuda()
    dst_all = torch.from_numpy(np.concatenate(
        [(np.arange(n) + rng.integers(0, b_slots)) % b_slots for n in flat] or [np.zeros(0, int)])).cuda()
    offs = np.concatenate(([0], np.cumsum(flat)))
    kd = torch.empty(b_slots, N_EL, dtype=torch.bfloat16, device="cuda"); vd = torch.empty_like(kd)
    def run(t0, t1):
        for t in range(t0, t1):
            for l in range(L):
                i = t * L + l
                launch(src_all[offs[i]:offs[i + 1]], dst_all[offs[i]:offs[i + 1]], kd, vd)
    run(0, WARMUP); torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu = kernel_ms(lambda: run(WARMUP, len(counts)), 1, "_gather2") / STEPS
    wall = (time.perf_counter() - t0) * 1e3 / STEPS
    return gpu, wall

rows = []
for B in BUDGETS:
    for b in BATCHES:
        for p in POLICIES:
            c = batch_counts(p, B, b, STEPS + WARMUP)
            mean, wall = time_transfer(c)
            pages = c[WARMUP:].sum(1).mean()
            rows.append(dict(B=B, batch=b, policy=p, pages_per_step=pages, transfer_ms_mean=mean,
                             launch_wall_ms=wall, us_per_page=1e3 * mean / max(pages, 1)))
            print(f"B={B:3d} b={b:2d} {p:9s} {pages:8.0f} pages/step  {mean:7.3f} ms/step  "
                  f"{1e3 * mean / max(pages, 1):5.2f} us/page")
""")

code(r"""
# Compute side: Qwen3-0.6B decode step at batch b over 1,088 attended tokens, plus Quest scoring.
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca",
                                             torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
CTX, PAGES = 1088, 1600

compute = {}
with torch.inference_mode():
    for b in BATCHES:
        ids = torch.randint(0, 150000, (b, CTX), device="cuda")
        out = model(input_ids=ids, use_cache=True)
        cache, tok = out.past_key_values, out.logits[:, -1:].argmax(-1)
        def step():
            global cache
            # crop back so every timed step attends to the same 1,088 + 1 tokens
            cache.crop(CTX)
            model(input_ids=tok, past_key_values=cache, use_cache=True)
        for _ in range(3): step()
        dec = kernel_ms(step, 10)
        q = torch.randn(b, KVH, 2, DIM, device="cuda")                  # 16 query heads = 8 KV heads x 2
        lo = torch.randn(b, KVH, PAGES, DIM, device="cuda"); hi = lo + 1
        def score():
            for _ in range(L):
                s = (torch.einsum("bhgd,bhpd->bhgp", q.clamp_min(0), hi)
                     + torch.einsum("bhgd,bhpd->bhgp", q.clamp_max(0), lo)).mean((1, 2))
                s.topk(K, dim=-1)
        for _ in range(3): score()
        sc = kernel_ms(score, 10)
        compute[b] = dict(decode_ms=dec, scoring_ms=sc)
        print(f"b={b:2d} decode {dec:7.3f} ms  scoring {sc:6.3f} ms (GPU kernel time per step)")
        del cache, out, lo, hi
        torch.cuda.empty_cache()
""")

code(r"""
import csv
import matplotlib.pyplot as plt
for r in rows:
    c = compute[r["batch"]]
    lru = next(x for x in rows if x["B"] == r["B"] and x["batch"] == r["batch"] and x["policy"] == "lru")
    r.update(decode_ms=c["decode_ms"], scoring_ms=c["scoring_ms"])
    r["step_ms"] = c["decode_ms"] + c["scoring_ms"] + r["transfer_ms_mean"]
    r["transfer_share"] = r["transfer_ms_mean"] / r["step_ms"]
    r["saved_vs_lru_ms"] = lru["transfer_ms_mean"] - r["transfer_ms_mean"]
    lru_step = c["decode_ms"] + c["scoring_ms"] + lru["transfer_ms_mean"]
    r["speedup_vs_lru"] = lru_step / r["step_ms"] - 1
with open(OUT / "batch_bench.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
meta = dict(gpu=GPU, torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
            memcpy_GBps=MEMCPY_GBS, gather_us_per_page=GATHER_US_PER_PAGE, steps=STEPS, warmup=WARMUP,
            batches=BATCHES, budgets=BUDGETS, schedule=str(SCHED), docs=RIDS, ctx=CTX, scoring_pages=PAGES,
            compute=compute)
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for B, ls in zip(BUDGETS, ["-", "--"]):
    for p, col in zip(POLICIES, ["#888", "#1f77b4", "#2ca02c", "#d62728"]):
        rr = [r for r in rows if r["B"] == B and r["policy"] == p]
        ax[0].plot([r["batch"] for r in rr], [r["transfer_share"] for r in rr], ls, color=col, marker="o", label=f"{p} B={B}")
        if p != "lru":
            ax[1].plot([r["batch"] for r in rr], [100 * r["speedup_vs_lru"] for r in rr], ls, color=col, marker="o", label=f"{p} B={B}")
for a in ax: a.set_xscale("log", base=2); a.set_xlabel("batch size")
ax[0].set_ylabel("transfer share of GPU step time"); ax[1].set_ylabel("step-time speedup vs LRU (%)")
ax[1].legend(fontsize=7); fig.suptitle(GPU); fig.tight_layout(); fig.savefig(OUT / "batch_bench.png", dpi=150)
print("\n B  batch  policy      pages/step  transfer ms  decode ms  share   saved ms  speedup")
for r in rows:
    print(f"{r['B']:3d} {r['batch']:5d}  {r['policy']:9s} {r['pages_per_step']:10.0f} {r['transfer_ms_mean']:11.3f} "
          f"{r['decode_ms']:10.3f} {r['transfer_share']:6.1%} {r['saved_vs_lru_ms']:9.3f} {r['speedup_vs_lru']:+7.2%}")
print("saved to", OUT)
if OUT_ROOT is None:
    import shutil; from google.colab import files
    shutil.make_archive("/content/batch_bench", "zip", OUT); files.download("/content/batch_bench.zip")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
