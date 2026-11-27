"""Exploratory per-setting replay for S6 (dev only): budgets as multiples of K, LODO-tuned families.

  python -m analysis_v2.explore TRACES --out results_v2/ext/<setting> [--ratios 1.25 1.5 2 4] [--procs 32]
      [--min-steps 0] [--glob '*.npz']

K is read from the traces. If the traces carry attn_mass, the two InfLLM baselines are also run with
the true attention mass. Writes misses.csv, summary.csv (pooled G with document bootstrap CIs, wins
vs LRU, repeat-miss drop), lodo.csv, short_docs.csv (documents with fewer than --min-steps decode
steps are excluded from the summary and listed there), and rho.csv when true_rho is present.
"""
import argparse
import csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import kvpolicy as kp

ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9, 0.95)
DECAYS = (0.1, 0.5, 0.9)


def configs(has_attn):
    C = {"lru": ("lru", {}, False), "belady": ("belady", {}, False), "score": ("score", {}, False),
         "arc": ("arc", {}, False), "dsum_softmax:0.5": ("dsum", dict(decay=0.5, v="softmax"), False),
         "infllm_eq1:0.1": ("infllm_eq1", dict(decay=0.1), False)}
    for a in ALPHAS:
        C[f"ema:{a}"] = ("ema", dict(alpha=a), False)
        C[f"ema_allrank:{a}"] = ("ema", dict(alpha=a, signal="allrank"), False)
    for d in DECAYS:
        C[f"infllm_lrus:{d}"] = ("infllm_lrus", dict(decay=d), False)
    if has_attn:
        C["infllm_eq1_true:0.1"] = ("infllm_eq1", dict(decay=0.1), True)
        for d in DECAYS:
            C[f"infllm_lrus_true:{d}"] = ("infllm_lrus", dict(decay=d), True)
            C[f"infllm_eq1_true:{d}"] = ("infllm_eq1", dict(decay=d), True)
    return C


FAMILIES = {"ema(lodo)": "ema:", "ema_allrank(lodo)": "ema_allrank:", "infllm_lrus(lodo)": "infllm_lrus:",
            "infllm_lrus_true(lodo)": "infllm_lrus_true:", "infllm_eq1_true(lodo)": "infllm_eq1_true:"}


def run(job):
    path, budgets, only = job
    with np.load(path, allow_pickle=False) as z:
        A = z["attn_mass"].astype(np.float32) if "attn_mass" in z.files else None
    R, S, rid = kp.load(path)
    out = []
    for name, (pol, kw, true) in configs(A is not None).items():
        if only and name not in only:
            continue
        for B in budgets:
            for l in range(R.shape[1]):
                c = kp.simulate(R[:, l], S[:, l], B, pol, attn=(A[:, l] if true else None), **kw)
                out.append((rid, l, name, B, c["accesses"], c["misses"], c["compulsory"]))
    return out


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def g_of(l, p, o):
    return (l - p) / (l - o) if l != o else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ratios", type=float, nargs="+", default=[1.25, 1.5, 2, 4])
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--min-steps", type=int, default=0)
    ap.add_argument("--glob", default="*.npz")
    ap.add_argument("--only", nargs="*", help="run only these fixed configs (no LODO tuning), e.g. on held-out data")
    a = ap.parse_args()
    paths = sorted(str(p) for p in a.traces.glob(a.glob))
    with np.load(paths[0]) as z:
        K = z["ranked"].shape[2]
    budgets = sorted({max(K, int(round(r * K))) for r in a.ratios})
    short, keep = [], []
    rho_rows = []
    for p in paths:
        with np.load(p) as z:
            T = z["ranked"].shape[0]
            rid = str(z["rid"])
            if "true_rho" in z.files:
                r = z["true_rho"]
                rho_rows += [(rid, l, float(np.nanmean(r[:, l])), float(np.nanmedian(r[:, l])))
                             for l in range(r.shape[1])]
        (short if T < a.min_steps else keep).append((p, rid, T))
    write(a.out / "short_docs.csv", ["doc", "steps"], [(rid, T) for _, rid, T in short])
    if rho_rows:
        write(a.out / "rho.csv", ["doc", "layer", "mean_rho", "median_rho"], rho_rows)
    with Pool(a.procs) as pool:
        rows = [r for part in pool.imap_unordered(run, [(p, budgets, a.only) for p, _, _ in keep]) for r in part]
    write(a.out / "misses.csv", ["doc", "layer", "config", "B", "accesses", "misses", "compulsory"], rows)
    M = defaultdict(lambda: defaultdict(lambda: np.zeros(3, np.int64)))
    for d, l, c, B, acc, m, comp in rows:
        M[c, B][d] += (acc, m, comp)
    docs = sorted({d for d, *_ in rows})
    per, lodo_rows = {}, []
    for B in budgets:
        for c in {c for c, b in M if b == B}:
            per[c, B] = np.array([M[c, B][d][1] for d in docs])
        for fam, prefix in FAMILIES.items():
            cfgs = sorted(c for c, b in M if b == B and c.startswith(prefix))
            if not cfgs:
                continue
            v = []
            for d in docs:
                tr = [x for x in docs if x != d]
                best = min(cfgs, key=lambda c: (sum(M[c, B][x][1] for x in tr), c))
                v.append(M[best, B][d][1])
                lodo_rows.append((fam, B, d, best))
            per[fam, B] = np.array(v)
    write(a.out / "lodo.csv", ["family", "B", "doc", "chosen"], lodo_rows)
    idx = np.random.default_rng(0).integers(0, len(docs), size=(2000, len(docs)))
    out = []
    for (label, B), pol in sorted(per.items(), key=lambda x: (x[0][1], x[0][0])):
        lru, opt = per["lru", B], per["belady", B]
        acc = sum(M["lru", B][d][0] for d in docs)
        comp = sum(M["lru", B][d][2] for d in docs)
        gs = [g_of(lru[i].sum(), pol[i].sum(), opt[i].sum()) for i in idx]
        lo, hi = np.nanpercentile(gs, [2.5, 97.5])
        out.append((label, B, B / K, int(pol.sum()), acc, pol.sum() / acc, g_of(lru.sum(), pol.sum(), opt.sum()), lo, hi,
                    1 - (pol.sum() - comp) / (lru.sum() - comp), int((pol < lru).sum()), len(docs)))
    write(a.out / "summary.csv", ["policy", "B", "B_over_K", "misses", "accesses", "miss_rate", "G", "G_lo", "G_hi",
                                  "repeat_miss_drop_vs_lru", "docs_better_than_lru", "n_docs"], out)
    print(f"K={K} budgets={budgets} docs={len(docs)} short={len(short)}")
    for r in out:
        if r[0] in ("lru", "belady", "score", "ema(lodo)", "ema_allrank(lodo)", "infllm_eq1:0.1",
                    "infllm_lrus(lodo)", "infllm_eq1_true:0.1", "infllm_lrus_true(lodo)"):
            print(f"B={r[1]:>4} {r[0]:<24} miss {100*r[5]:.2f}%  G {r[6]:+.3f} [{r[7]:+.3f},{r[8]:+.3f}] "
                  f"wins {r[10]}/{r[11]}")


if __name__ == "__main__":
    main()
