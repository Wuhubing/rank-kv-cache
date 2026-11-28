"""C4 (followup-v3): allocation without offline curves. Dev only (0.6B and 8B dev traces).

  python -m analysis_v3.c4_alloc replay TRACES --out results_v3/c4/<model> [--procs 16]
  python -m analysis_v3.c4_alloc summarize --out results_v3/c4/<model>

curves   per (document, layer, grid budget): misses of LRU, Belady and score EMA (z; alpha 0.3 / 0.5 / 0.7). Offline
         allocation = kvpolicy.allocate (greedy on summed training curves, as efficient_dev / the frozen files) with the
         total L*B; LODO = curves of the other documents; oracle = the document's own curves. Because layers are
         independent streams, the misses at any per-layer budget vector are sums of curve entries (no extra replay),
         and so are LRU and Belady at the same allocation (validation/alloc_oracle.py's anchors).
online   per-layer score EMA with per-layer budgets that move during decoding: each layer keeps a ghost FIFO of its last
         g = s evicted pages; a miss that hits the ghost counts as "s more slots would have saved it". Every N steps the
         layer with the most ghost hits takes s slots from the layer with the fewest (never below K); counters are then
         multiplied by `decay` (0 = reset). No offline statistic is used. Grid N x s x decay, tuned LODO.
pool     one pool of L*B slots for all layers; at each (step, layer) the selected pages of that layer must be resident and
         the victim is the resident page (any layer) with the lowest value: value = EMA of the layer's signal (z, or the
         all-page percentile "allrank") + beta * (the layer's running miss rate, EMA 0.99 over steps). Ties: oldest last
         use. Also global LRU and global Belady (next use in (step, layer) order) on the same pool.
Bootstraps: 2000 resamples of documents, seed 0, paired.
"""
import argparse
import csv
import heapq
import math
from collections import OrderedDict, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import kvpolicy as kp

GRID = [64, 72, 80, 88, 96, 104, 112, 128, 160, 192, 256]
BUDGETS = [80, 96, 128, 256]
ALPHAS = (0.3, 0.5, 0.7)
ONLINE = [dict(N=N, s=s, decay=d) for N in (16, 64, 256) for s in (2, 4, 8) for d in (0.0, 0.5)]
POOLS = ([("lru", {}), ("belady", {})] +
         [("ema", dict(signal=sig, beta=b)) for sig in ("z", "allrank") for b in (0.0, 0.5, 2.0)])


def curves_job(args):
    path, = args
    R, S, rid = kp.load(path)
    rows = []
    for l in range(R.shape[1]):
        for B in GRID:
            for name, pol, kw in ([("lru", "lru", {}), ("belady", "belady", {})] +
                                  [(f"ema:{a}", "ema", dict(alpha=a)) for a in ALPHAS]):
                c = kp.simulate(R[:, l], S[:, l], B, pol, **kw)
                rows.append(("curve", rid, name, l, B, c["accesses"], c["misses"], c["compulsory"]))
    return rows


def online_run(R, S, B, alpha, N, s, decay):
    """Per-layer ScoreEMA with ghost-driven reallocation. Returns total misses and the final budgets."""
    T, L, _ = R.shape
    rows = [[tuple(int(p) for p in R[t, l] if p >= 0) for t in range(T)] for l in range(L)]
    K = max(len(r) for lr in rows for r in lr)
    budget = [B] * L
    pols = [kp.ScoreEMA(K, alpha=alpha) for _ in range(L)]
    res = [set() for _ in range(L)]
    ghost = [OrderedDict() for _ in range(L)]
    gh = [0.0] * L
    miss = 0
    for t in range(T):
        for l in range(L):
            row, pol = rows[l][t], pols[l]
            pol.begin(t, row, S[t, l].tolist())
            sel = set(row)
            hits = sel & res[l]
            misses = sel - hits
            for p in misses:
                if p in ghost[l]:
                    gh[l] += 1
                    del ghost[l][p]
            n = len(res[l]) + len(misses) - budget[l]
            if n > 0:
                for v in pol.evict(res[l] - sel, n):
                    res[l].discard(v)
                    pol.dropped(v)
                    ghost[l][v] = None
                    while len(ghost[l]) > s:
                        ghost[l].popitem(last=False)
            res[l] |= sel
            pol.admit(hits, misses)
            miss += len(misses)
        if (t + 1) % N == 0:
            recv = max(range(L), key=lambda l: (gh[l], -budget[l], -l))
            cand = [l for l in range(L) if l != recv and budget[l] - s >= K]
            if cand:
                donor = min(cand, key=lambda l: (gh[l], -budget[l], l))
                if gh[recv] > gh[donor]:
                    budget[donor] -= s
                    budget[recv] += s
            gh = [g * decay for g in gh]
    return miss, budget


def pool_run(R, S, total, kind, alpha=0.5, signal="z", beta=0.0):
    """Global pool of `total` slots over all layers. kind: lru | belady | ema."""
    T, L, _ = R.shape
    rows = [[tuple(int(p) for p in R[t, l] if p >= 0) for l in range(L)] for t in range(T)]
    res = set()                   # (l, p)
    last = {}
    val = [dict() for _ in range(L)]   # per layer: page -> EMA value
    mr = [0.0] * L
    fut = None
    if kind == "belady":
        fut = defaultdict(list)
        for t in range(T - 1, -1, -1):
            for l in range(L - 1, -1, -1):
                for p in rows[t][l]:
                    fut[l, p].append(t * L + l)   # popped from the end = next use first
    miss = acc = 0
    for t in range(T):
        for l in range(L):
            row = rows[t][l]
            if kind == "ema":
                srow = S[t, l]
                kth, top = srow[row[-1]], srow[row[0]]
                if signal == "allrank":
                    table = np.sort(srow[np.isfinite(srow)].astype(np.float64))
                vl = val[l]
                for p in vl:
                    x = ((srow[p] - kth) / (top - kth + 1e-6) if signal == "z"
                         else kp._allrank(table, float(srow[p])))
                    vl[p] = alpha * vl[p] + (1 - alpha) * x
            sel = {(l, p) for p in row}
            hits = sel & res
            misses = sel - hits
            n = len(res) + len(misses) - total
            if n > 0:
                cands = res - sel
                if kind == "lru":
                    key = lambda q: (last[q], q[0], q[1])  # noqa: E731
                elif kind == "belady":
                    key = lambda q: (-(fut[q][-1] if fut[q] else math.inf), q[0], q[1])  # noqa: E731
                else:
                    key = lambda q: (val[q[0]][q[1]] + beta * mr[q[0]], last[q], q[0], q[1])  # noqa: E731
                for v in heapq.nsmallest(n, cands, key=key):
                    res.discard(v)
                    last.pop(v, None)
                    val[v[0]].pop(v[1], None)
            res |= sel
            for q in sel:
                last[q] = t * L + l
                if kind == "belady":
                    fut[q].pop()
            if kind == "ema":
                srow = S[t, l]
                kth, top = srow[row[-1]], srow[row[0]]
                for q in misses:
                    p = q[1]
                    val[l][p] = ((srow[p] - kth) / (top - kth + 1e-6) if signal == "z"
                                 else kp._allrank(table, float(srow[p])))
            mr[l] = 0.99 * mr[l] + 0.01 * (len(misses) / len(row))
            miss += len(misses)
            acc += len(row)
    return miss, acc


def online_job(args):
    path, B, alpha, cfg = args
    R, S, rid = kp.load(path)
    m, budget = online_run(R, S, B, alpha, **cfg)
    return [("online", rid, f"online:{alpha}:{cfg['N']}:{cfg['s']}:{cfg['decay']}", -1, B, 0, m, 0,
             " ".join(map(str, budget)))]


def pool_job(args):
    path, B, kind, kw = args
    R, S, rid = kp.load(path)
    m, acc = pool_run(R, S, B * R.shape[1], kind, **kw)
    name = f"pool_{kind}" + ("" if kind != "ema" else f":{kw['signal']}:{kw['beta']}")
    return [("pool", rid, name, -1, B, acc, m, 0, "")]


def run_any(job):
    kind, args = job
    return {"curves": curves_job, "online": online_job, "pool": pool_job}[kind](args)


def cmd_replay(a):
    paths = sorted(str(p) for p in a.traces.glob("*.npz"))
    jobs = [("curves", (p,)) for p in paths]
    jobs += [("online", (p, B, 0.5, cfg)) for p in paths for B in BUDGETS for cfg in ONLINE]
    jobs += [("pool", (p, B, kind, kw)) for p in paths for B in BUDGETS for kind, kw in POOLS]
    a.out.mkdir(parents=True, exist_ok=True)
    with Pool(a.procs) as pool, open(a.out / "rows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "doc", "config", "layer", "B", "accesses", "misses", "compulsory", "extra"])
        for n, rows in enumerate(pool.imap_unordered(run_any, jobs), 1):
            for r in rows:
                w.writerow(list(r) + [""] * (9 - len(r)))
            if n % 50 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
                f.flush()


def g_of(l, p, o):
    return (l - p) / (l - o) if l != o else float("nan")


def cmd_summarize(a):
    curve = defaultdict(dict)        # (doc, name) -> {(layer, B): misses}
    other = defaultdict(dict)        # (config, B) -> {doc: misses}
    budgets_of = {}
    with open(a.out / "rows.csv") as f:
        for r in csv.DictReader(f):
            if r["kind"] == "curve":
                curve[r["doc"], r["config"]][int(r["layer"]), int(r["B"])] = int(r["misses"])
            else:
                other[r["config"], int(r["B"])][r["doc"]] = int(r["misses"])
                if r["extra"]:
                    budgets_of[r["config"], int(r["B"]), r["doc"]] = r["extra"]
    docs = sorted({d for d, _ in curve})
    L = 1 + max(l for (l, _) in curve[docs[0], "lru"])
    idx = np.random.default_rng(0).integers(0, len(docs), size=(2000, len(docs)))

    def tot(doc, name, alloc):
        return sum(curve[doc, name][l, alloc[l]] for l in range(L))

    def lodo_alpha(B, train):
        return min(ALPHAS, key=lambda al: (sum(tot(d, f"ema:{al}", [B] * L) for d in train), al))

    rows, cons = [], []
    for B in BUDGETS:
        per = defaultdict(list)
        for d in docs:
            train = [x for x in docs if x != d]
            al = lodo_alpha(B, train)
            ema = f"ema:{al}"
            uni = [B] * L
            per["lru_uniform"].append(tot(d, "lru", uni))
            per["belady_uniform"].append(tot(d, "belady", uni))
            per["ema_uniform"].append(tot(d, ema, uni))
            for tag, src in (("offline_lodo", train), ("oracle", [d])):
                cur = {l: {g: sum(curve[x, ema][l, g] for x in src) for g in GRID} for l in range(L)}
                alloc = kp.allocate(cur, L, L * B, GRID)
                al_v = [alloc[l] for l in range(L)]
                per[f"alloc_{tag}+ema"].append(tot(d, ema, al_v))
                per[f"lru_same_alloc_{tag}"].append(tot(d, "lru", al_v))
                per[f"belady_same_alloc_{tag}"].append(tot(d, "belady", al_v))
            for fam, prefix in (("online(lodo)", "online:"), ("pool_ema_z(lodo)", "pool_ema:z:"),
                                ("pool_ema_allrank(lodo)", "pool_ema:allrank:")):
                cfgs = sorted(c for (c, b) in other if b == B and c.startswith(prefix))
                if cfgs:
                    best = min(cfgs, key=lambda c: (sum(other[c, B][x] for x in train), c))
                    per[fam].append(other[best, B][d])
            for c in ("pool_lru", "pool_belady"):
                if (c, B) in other:
                    per[c].append(other[c, B][d])
        per = {k: np.array(v, float) for k, v in per.items() if len(v) == len(docs)}
        lu, bu = per["lru_uniform"], per["belady_uniform"]
        for k, v in per.items():
            gs = [g_of(lu[i].sum(), v[i].sum(), bu[i].sum()) for i in idx]
            lo, hi = np.nanpercentile(gs, [2.5, 97.5])
            rows.append(dict(B=B, policy=k, misses=int(v.sum()), G_uniform_anchors=g_of(lu.sum(), v.sum(), bu.sum()),
                             lo=lo, hi=hi, docs_better_than_uniform_ema=int((v < per["ema_uniform"]).sum())))
        # share of the offline allocation gain reached without offline curves, and paired contrasts
        eu, eo = per["ema_uniform"], per["alloc_offline_lodo+ema"]
        for k in ("online(lodo)", "pool_ema_z(lodo)", "pool_ema_allrank(lodo)", "alloc_oracle+ema"):
            if k not in per:
                continue
            v = per[k]
            share = (eu.sum() - v.sum()) / (eu.sum() - eo.sum()) if eu.sum() != eo.sum() else float("nan")
            sh = [(eu[i].sum() - v[i].sum()) / (eu[i].sum() - eo[i].sum()) for i in idx]
            dG = [g_of(lu[i].sum(), v[i].sum(), bu[i].sum()) - g_of(lu[i].sum(), eu[i].sum(), bu[i].sum()) for i in idx]
            dGo = [g_of(lu[i].sum(), v[i].sum(), bu[i].sum()) - g_of(lu[i].sum(), eo[i].sum(), bu[i].sum()) for i in idx]
            cons.append(dict(B=B, policy=k, share_of_offline_gain=share,
                             share_lo=np.nanpercentile(sh, 2.5), share_hi=np.nanpercentile(sh, 97.5),
                             dG_vs_uniform_ema=g_of(lu.sum(), v.sum(), bu.sum()) - g_of(lu.sum(), eu.sum(), bu.sum()),
                             dG_vs_uniform_ema_lo=np.nanpercentile(dG, 2.5),
                             dG_vs_uniform_ema_hi=np.nanpercentile(dG, 97.5),
                             dG_vs_offline=g_of(lu.sum(), v.sum(), bu.sum()) - g_of(lu.sum(), eo.sum(), bu.sum()),
                             dG_vs_offline_lo=np.nanpercentile(dGo, 2.5), dG_vs_offline_hi=np.nanpercentile(dGo, 97.5),
                             docs_better_than_uniform_ema=int((v < eu).sum()), n=len(docs)))
        if "pool_belady" in per:
            pb, pl = per["pool_belady"], per["pool_lru"]
            for k in ("pool_ema_z(lodo)", "pool_ema_allrank(lodo)"):
                if k in per:
                    rows.append(dict(B=B, policy=k + " vs global anchors", misses=int(per[k].sum()),
                                     G_uniform_anchors=g_of(pl.sum(), per[k].sum(), pb.sum()), lo=float("nan"),
                                     hi=float("nan"), docs_better_than_uniform_ema=-1))
    for name, data in (("summary.csv", rows), ("shares.csv", cons)):
        with open(a.out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    for r in rows:
        print(f"B={r['B']:>3} {r['policy']:<34} G {r['G_uniform_anchors']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]")
    for c in cons:
        print(f"B={c['B']:>3} {c['policy']:<24} share {c['share_of_offline_gain']:.2f} [{c['share_lo']:.2f},"
              f"{c['share_hi']:.2f}] dG vs uniform EMA {c['dG_vs_uniform_ema']:+.3f} "
              f"[{c['dG_vs_uniform_ema_lo']:+.3f},{c['dG_vs_uniform_ema_hi']:+.3f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["replay", "summarize"])
    ap.add_argument("traces", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    dict(replay=cmd_replay, summarize=cmd_summarize)[a.cmd](a)


if __name__ == "__main__":
    main()
