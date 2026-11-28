"""D1 / G1 (followup-v3): iso-miss capacity of policy + allocation, and what it means in GPU memory.

  python -m analysis_v3.d1_iso --lru-fine out/v3/lru_fine_dev/misses.csv --layers 28 \
      --policy ema=out/v3/b1_dev/misses.csv:ema_z:LODO --policy c1=out/v3/b1_dev/misses.csv:pow_norm_kth:1.0 \
      --c4 out/v3/c4_dev/rows.csv --out results_v3/d1/dev

For every budget B and policy: total misses over the documents; the LRU budget (uniform, linear interpolation on the
fine LRU grid 64..1024 step 8) with the same total misses; the iso-miss capacity ratio iso_B / B; the GPU memory that ratio
saves at the same miss count, (iso_B - B) x L x 64 KiB per sequence (a 16-token page of one layer holds K and V of 8 KV heads
x 128 dims in bf16 = 64 KiB for both Qwen3 models); and at fixed memory the miss and host->GPU byte reduction against LRU.
--policy name=FILE:CONFIG reads a miss table (doc, layer, config, B, ...); CONFIG "ema_z:LODO" picks, per budget, the EMA alpha
with the fewest pooled misses over the table's ema_z grid (the LODO choice converges to it). --c4 adds the offline allocation +
EMA (sum of curve entries at the LODO allocation, as analysis_v3.c4_alloc) and the LODO-tuned global percentile pool.
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PAGE_BYTES = 64 * 1024


def rd(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        yield from csv.DictReader(f)


def iso(grid, lru, m):
    if m > lru[0]:
        return float("nan")
    for (b0, m0), (b1, m1) in zip(zip(grid, lru), zip(grid[1:], lru[1:])):
        if m1 <= m <= m0:
            return b0 + (b1 - b0) * (m0 - m) / (m0 - m1) if m0 != m1 else b0
    return float("inf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lru-fine", type=Path, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--policy", action="append", default=[])
    ap.add_argument("--c4", type=Path)
    ap.add_argument("--budgets", type=int, nargs="+", default=[80, 96, 128, 256])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    lf = defaultdict(int)
    acc = defaultdict(int)
    docs = set()
    for r in rd(a.lru_fine):
        if r["config"] == "lru":
            lf[int(r["B"])] += int(r["misses"])
            acc[int(r["B"])] += int(r["accesses"])
            docs.add(r["doc"])
    grid = sorted(lf)
    lru = [lf[b] for b in grid]
    tot = {}                                   # (policy, B) -> total misses
    for spec in a.policy:
        name, rest = spec.split("=", 1)
        path, cfg = rest.split(":", 1)
        T = defaultdict(int)
        for r in rd(path):
            if r["doc"] in docs:
                T[r["config"], int(r["B"])] += int(r["misses"])
        for B in a.budgets:
            if cfg.endswith(":LODO"):
                fam = cfg[:-5]
                cands = {c: m for (c, b), m in T.items() if b == B and c.startswith(fam + ":")}
                if cands:
                    tot[name, B] = min(cands.values())
            elif (cfg, B) in T:
                tot[name, B] = T[cfg, B]
    if a.c4:
        curve = defaultdict(dict)
        other = defaultdict(dict)
        for r in rd(a.c4):
            if r["kind"] == "curve":
                curve[r["doc"], r["config"]][int(r["layer"]), int(r["B"])] = int(r["misses"])
            else:
                other[r["config"], int(r["B"])][r["doc"]] = int(r["misses"])
        import kvpolicy as kp
        from analysis_v3.c4_alloc import ALPHAS, GRID
        cdocs = sorted({d for d, _ in curve})
        L = a.layers
        for B in a.budgets:
            s_alloc = s_pool = 0
            for d in cdocs:
                tr = [x for x in cdocs if x != d]
                al = min(ALPHAS, key=lambda q: (sum(curve[x, f"ema:{q}"][l, B] for x in tr for l in range(L)), q))
                cur = {l: {g: sum(curve[x, f"ema:{al}"][l, g] for x in tr) for g in GRID} for l in range(L)}
                alloc = kp.allocate(cur, L, L * B, GRID)
                s_alloc += sum(curve[d, f"ema:{al}"][l, alloc[l]] for l in range(L))
                pc = sorted(c for (c, b) in other if b == B and c.startswith("pool_ema:allrank:"))
                best = min(pc, key=lambda c: (sum(other[c, B][x] for x in tr), c))
                s_pool += other[best, B][d]
            tot["alloc+ema", B] = s_alloc
            tot["pool_percentile", B] = s_pool
    rows = []
    L = a.layers
    for (name, B), m in sorted(tot.items(), key=lambda x: (x[0][1], x[0][0])):
        ib = iso(grid, lru, m)
        lb = lf[B]
        rows.append(dict(policy=name, B=B, B_over_K=B / 64, misses=m, lru_misses=lb, miss_rate=m / acc[B],
                         lru_miss_rate=lb / acc[B], iso_lru_B=ib, iso_ratio=ib / B,
                         MiB_saved_per_sequence=(ib - B) * L * PAGE_BYTES / 2 ** 20,
                         miss_reduction_vs_lru=1 - m / lb, bytes_saved_GiB_total=(lb - m) * PAGE_BYTES / 2 ** 30))
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "iso.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"B={r['B']:>3} {r['policy']:<16} iso {r['iso_lru_B']:7.1f} ratio {r['iso_ratio']:.3f} "
              f"saves {r['MiB_saved_per_sequence']:7.1f} MiB/seq; misses -{100*r['miss_reduction_vs_lru']:.1f} % vs LRU")


if __name__ == "__main__":
    main()
