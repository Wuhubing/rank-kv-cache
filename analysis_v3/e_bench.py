"""Part E (followup-v3): batched system measurements with gpu_bench_batch2 on one GPU.

  CUDA_VISIBLE_DEVICES=<free card> python -m analysis_v3.e_bench --out out/v3/e/<name> \
      [--split splits/longctx_v1.json --frozen splits/heldout_frozen_v1.json] [--batches 1 4 8 16 32] [--rounds 5]
      [--steps 264 --warmup 8] [--tokens 16384] [--budget 96] [--page-size 16 --top-k 64]
      [--policies lru score ema alloc_ema belady allmiss nofetch] [--delay-ns-per-byte 0 ...] [--timing]
      [--wait-clean]

Design (task section 9, E2): requests are the split's prompts with >= --tokens tokens in rid order, cut to exactly
--tokens tokens (first and last halves), so the request set of a smaller batch is a prefix (subset) of a larger one.
Per batch: a preparation LRU run (not timed) gives the ranks for Belady's next uses and the per-request simulator
check; then --rounds rounds, each starting with one discarded warm-up run and then every policy once in a random
order (numpy seed = round index + 1000 * batch, recorded). Every run records load average, GPU temperature, SM/memory
clocks, power and the compute processes on the card before and after. Per run: per-step wall latency, per-step miss
counts and page-completion flags (for E4), misses, bytes. --delay-ns-per-byte repeats every round for each delay
value (slow-tier simulation, E5; reported as simulated).
--wait-clean (R11-6, added after PR #11): before every run, wait (60 s polls, at most --wait-max s) until no other
process is on the card; a run with another process on the card before or after is repeated (at most --retries times)
and the unclean attempt is kept in "discarded_runs" (the summary reads "runs" only). Per run also: GPU memory by phase
(peak of the prefill forwards, peak of the migration to the host tier, allocated at decode start, decode warm-up peak,
steady-state peak; bytes).
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from analysis_v2.s7_bench import cut, frozen, load_prompts


def gpu_state():
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    q = subprocess.run(["nvidia-smi", "-i", dev, "--query-gpu=index,uuid,temperature.gpu,clocks.sm,clocks.mem,power.draw",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "-i", dev, "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    idx, uuid, temp, sm, mem, pw = [x.strip() for x in q.split(",")]
    pids = [int(l.split(",")[0]) for l in apps.splitlines() if l.strip()]
    return dict(index=idx, uuid=uuid, temp_c=float(temp), sm_mhz=float(sm), mem_mhz=float(mem), power_w=float(pw),
                loadavg1=os.getloadavg()[0], other_pids=[p for p in pids if p != os.getpid()],
                time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def wait_clean(wait_max, poll=60):
    """Seconds waited until no other process is on the card (gives up after wait_max; the run is then unclean)."""
    t0 = time.time()
    while gpu_state()["other_pids"] and time.time() - t0 < wait_max:
        time.sleep(poll)
    return round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="splits/longctx_v1.json")
    ap.add_argument("--frozen", default="splits/heldout_frozen_v1.json")
    ap.add_argument("--ndocs", type=int, default=35)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=264)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--budget", type=int, default=96)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--policies", nargs="+", default=["lru", "score", "ema", "alloc_ema", "belady", "allmiss", "nofetch"])
    ap.add_argument("--delay-ns-per-byte", type=float, nargs="+", default=[0.0])
    ap.add_argument("--alpha", type=float, help="EMA alpha (default: frozen value for --budget)")
    ap.add_argument("--timing", action="store_true", help="CUDA-event breakdown (E6; adds overhead, separate runs)")
    ap.add_argument("--overlap", action="store_true", help="side-stream fetch (faulted in long runs; off by default)")
    ap.add_argument("--wait-clean", action="store_true", help="wait for a card without other processes before each run")
    ap.add_argument("--wait-max", type=int, default=1800)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--wait-total", type=float, default=float("inf"),
                    help="cap on the accumulated --wait-clean waiting of this process (s); afterwards runs do not wait")
    a = ap.parse_args()
    import torch
    import gpu_bench_batch as gbb
    import gpu_bench_batch2 as gb2
    import kvpolicy as kp
    gb2.PS = a.page_size
    split, tok, model, prompts = load_prompts(a.split, "all", a.ndocs)
    alpha_f, alloc_f = frozen(a.frozen)
    alpha = a.alpha if a.alpha is not None else alpha_f.get(a.budget, 0.5)
    L = model.config.num_hidden_layers
    alloc = alloc_f.get(a.budget) if a.page_size == 16 and a.top_k == 64 else None
    reqs_all = [(rid, cut(p, a.tokens)) for rid, p in prompts if len(p) >= a.tokens]
    a.out.mkdir(parents=True, exist_ok=True)
    meta = dict(args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, alpha=alpha,
                alloc=alloc, requests=[rid for rid, _ in reqs_all], start=gpu_state())
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    pols = [p for p in a.policies if not (p == "alloc_ema" and (alloc is None or len(alloc) != L))]
    waited_total = 0.0          # --wait-clean waiting accumulated over all batches of this process
    for R in a.batches:
        if R > len(reqs_all):
            print(f"batch {R}: only {len(reqs_all)} requests", flush=True)
            break
        P = [p for _, p in reqs_all[:R]]

        def one(pol, delay=0.0, steps=a.steps, timing=False, nxt=None):
            kw = dict(K=a.top_k, warmup=a.warmup, delay_ns_per_byte=delay, timing=timing, overlap=a.overlap)
            if pol == "belady":
                return gb2.run(model, P, "belady", steps, a.budget, nxt=nxt, **kw)
            if pol == "score":
                return gb2.run(model, P, "ema", steps, a.budget, alpha=0.0, **kw)
            if pol == "ema":
                return gb2.run(model, P, "ema", steps, a.budget, alpha=alpha, **kw)
            if pol == "alloc_ema":
                return gb2.run(model, P, "ema", steps, [alloc[l] for l in range(L)], alpha=alpha, **kw)
            return gb2.run(model, P, pol, steps, a.budget, **kw)

        prep = one("lru")
        nxt = gbb.next_use_batch(prep["ranks"])
        checks = []
        for r in range(R):
            for l in range(L):
                sim = kp.simulate(prep["ranks"][:, r, l], None, a.budget, "lru")["misses"]
                checks.append(int(prep["misses"][r, l]) == sim)
        runs, discarded = [], []
        for rnd in range(a.rounds):
            seed = rnd + 1000 * R
            rng = np.random.default_rng(seed)
            todo = [(p, d) for d in a.delay_ns_per_byte for p in pols]
            order = [todo[i] for i in rng.permutation(len(todo))]
            one("lru", steps=max(a.warmup + 8, 24))       # discarded warm-up run
            for pol, delay in order:
                for attempt in range(a.retries + 1 if a.wait_clean else 1):
                    waited = wait_clean(min(a.wait_max, max(0.0, a.wait_total - waited_total))) if a.wait_clean else 0
                    waited_total += waited
                    before = gpu_state()
                    res = one(pol, delay, timing=a.timing, nxt=nxt)
                    after = gpu_state()
                    ok_tokens = res["gen"] == prep["gen"] if pol != "nofetch" else None
                    ok_ranks = bool(np.array_equal(res["ranks"], prep["ranks"])) if pol != "nofetch" else None
                    lat = res["lat"]
                    n = len(lat) - a.warmup
                    runs.append(dict(R=R, round=rnd, seed=seed, policy=pol, delay_ns_per_byte=delay,
                                     position=order.index((pol, delay)), lat_ms=(lat * 1e3).tolist(),
                                     step_misses=res["step_misses"].tolist(), page_step=res["page_step"].tolist(),
                                     misses=int(res["misses"].sum()), bytes=int(res["bytes"]),
                                     tokens_per_s=R * n / float(lat[a.warmup:].sum()),
                                     median_ms=float(np.median(lat[a.warmup:]) * 1e3),
                                     ranks_equal_prep=ok_ranks, tokens_equal_prep=ok_tokens,
                                     breakdown_ms=res["breakdown"].tolist() if res["breakdown"] is not None else None,
                                     peak_decode=res["peak_decode_steady"], host_bytes=res["host_bytes"],
                                     mem_phases=dict(prefill_forward=res.get("peak_prefill_forward"),
                                                     migration=res.get("peak_migration"), decode_start=res["mem_start"],
                                                     decode_warmup=res["peak_decode_warmup"],
                                                     decode_steady=res["peak_decode_steady"]),
                                     waited_s=waited, attempt=attempt, before=before, after=after))
                    print(f"R={R} round={rnd} {pol} delay={delay} median {runs[-1]['median_ms']:.1f} ms "
                          f"{runs[-1]['tokens_per_s']:.1f} tok/s misses {runs[-1]['misses']} load {before['loadavg1']:.1f} "
                          f"others {before['other_pids']} {after['other_pids']}", flush=True)
                    del res
                    torch.cuda.empty_cache()
                    if not a.wait_clean or not (before["other_pids"] or after["other_pids"]):
                        break
                    if attempt < a.retries:
                        discarded.append(runs.pop())
        out = dict(R=R, budget=a.budget, alpha=alpha, page_bytes=int(prep["page_bytes"]),
                   lru_misses_equal_simulator=all(checks), checks=len(checks), runs=runs, discarded_runs=discarded)
        (a.out / f"batch{R}.json").write_text(json.dumps(out))
        del prep
        torch.cuda.empty_cache()
    meta["end"] = gpu_state()
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
