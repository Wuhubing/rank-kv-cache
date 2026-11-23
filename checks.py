"""Robustness, ablation and mechanism checks on top of efficient_dev.py outputs (dev only).

  python checks.py TRACES --res results/efficient_dev [--budgets 96 128 256] [--procs 11]

Adds misses_ext.csv (ablation configs) and writes checks_*.csv plus checks_report.txt.
"""
import argparse
import csv
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import kvpolicy as kp
from efficient_dev import GRID, write

EXTRA = {}
for a in (0.1, 0.2, 0.4, 0.6, 0.8, 0.95):
    EXTRA[f"ema:{a}"] = ("ema", dict(alpha=a))
for a in (0.0, 0.3, 0.5, 0.7, 0.9):
    EXTRA[f"ema_sel:{a}"] = ("ema", dict(alpha=a, mode="selected"))
for a in (0.3, 0.5, 0.7, 0.9):
    EXTRA[f"ema_rank:{a}"] = ("ema", dict(alpha=a, signal="rank"))
for a in (0.0, 0.5):
    EXTRA[f"ema_rank_sel:{a}"] = ("ema", dict(alpha=a, signal="rank", mode="selected"))
for a in (0.3, 0.5, 0.7):
    EXTRA[f"ema_raw:{a}"] = ("ema", dict(alpha=a, signal="raw"))
for d in (0.5, 0.8, 0.9, 0.95, 0.99):
    EXTRA[f"infllm:{d}"] = ("infllm", dict(decay=d))

FAMILIES = {
    "ema(z,all)": ["ema:0.1", "ema:0.2", "ema:0.3", "ema:0.4", "ema:0.5", "ema:0.6", "ema:0.7",
                   "ema:0.8", "ema:0.9", "ema:0.95"],
    "ema(z,selected)": [c for c in EXTRA if c.startswith("ema_sel:")],
    "ema(rank,all)": [c for c in EXTRA if c.startswith("ema_rank:")],
    "ema(rank,selected)": [c for c in EXTRA if c.startswith("ema_rank_sel:")],
    "ema(raw,all)": [c for c in EXTRA if c.startswith("ema_raw:")],
    "infllm": [c for c in EXTRA if c.startswith("infllm:")],
    "rank(stage1-2)": [f"rank:{d}:{b}" for d in (0, 4, 8) for b in (0, 2, 4)],
    "hybrid": [f"hybrid:{w}:{a}" for w in (1, 4, 16, 64) for a in (0, 0.5, 0.8)],
}
FIXED = ["lru", "fifo", "lfu", "lru2", "arc", "score", "belady"]


def run_doc(args):
    path, config, budgets = args
    R, S, rid = kp.load(path)
    name, kw = EXTRA[config]
    rows = []
    for B in budgets:
        for l in range(R.shape[1]):
            c = kp.simulate(R[:, l], S[:, l], B, name, **kw)
            rows.append((rid, l, config, B, c["accesses"], c["misses"], c["compulsory"]))
    return rows


def doc_meta(paths):
    meta = {}
    for p in paths:
        z = np.load(p)
        rid = str(z["rid"])
        meta[rid] = dict(task=rid.split("/")[0],
                         band="24-32K" if int(z["prompt_tokens"]) >= 24000 else "16-24K",
                         steps=int(z["ranked"].shape[0]))
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", type=Path)
    ap.add_argument("--res", type=Path, default=Path("results/efficient_dev"))
    ap.add_argument("--budgets", type=int, nargs="+", default=[96, 128, 256])
    ap.add_argument("--procs", type=int, default=11)
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()
    paths = sorted(args.traces.glob("*.npz"))
    ext = args.res / "misses_ext.csv"
    if not args.skip_run:
        have = set()
        if ext.exists():
            with open(ext) as f:
                have = {(r["config"], int(r["B"])) for r in csv.DictReader(f)}
        todo = [(str(p), c, [B for B in args.budgets if (c, B) not in have]) for p in paths for c in EXTRA]
        todo = [t for t in todo if t[2]]
        new = not ext.exists()
        with Pool(args.procs) as pool, open(ext, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["doc", "layer", "config", "B", "accesses", "misses", "compulsory"])
            for part in pool.imap_unordered(run_doc, todo):
                w.writerows(part)
                f.flush()

    # Load everything into M[config][B][doc] = (acc, miss, comp) summed over layers, plus per layer.
    M = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0])))
    ML = defaultdict(lambda: defaultdict(dict))
    for fn in ("misses.csv", "misses_ext.csv"):
        with open(args.res / fn) as f:
            for r in csv.DictReader(f):
                c, B, d, l = r["config"], int(r["B"]), r["doc"], int(r["layer"])
                v = (int(r["accesses"]), int(r["misses"]), int(r["compulsory"]))
                cell = M[c][B][d]
                for i in range(3):
                    cell[i] += v[i]
                ML[c][B][d, l] = v[1]
    meta = doc_meta(paths)
    docs = sorted(meta)
    L = 1 + max(l for _, l in ML["lru"][96])
    budgets = sorted(set(args.budgets) | {72, 80})
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(docs), (2000, len(docs)))
    report = []

    def vec(c, B, i=1):
        return np.array([M[c][B][d][i] for d in docs], float)

    def stats(pol, B):
        """pol: per-doc misses vector. Returns G, CI, repeat-miss drop, CI, docs better, worst."""
        lru, opt, comp = vec("lru", B), vec("belady", B), vec("lru", B, 2)
        g = lambda i: (lru[i].sum() - pol[i].sum()) / (lru[i].sum() - opt[i].sum())
        rd = lambda i: 1 - (pol[i].sum() - comp[i].sum()) / (lru[i].sum() - comp[i].sum())
        allidx = np.arange(len(docs))
        G, R = g(allidx), rd(allidx)
        Gs = np.array([g(b) for b in boot])
        Rs = np.array([rd(b) for b in boot])
        rel = (pol - lru) / lru
        return dict(G=G, G_lo=np.percentile(Gs, 2.5), G_hi=np.percentile(Gs, 97.5), rep_drop=R,
                    rep_lo=np.percentile(Rs, 2.5), rep_hi=np.percentile(Rs, 97.5),
                    better=int((pol < lru).sum()), worst_rel=float(rel.max()),
                    miss_rate=pol.sum() / vec("lru", B, 0).sum())

    def lodo(family, B, configs=None):
        cfgs = [c for c in (configs or FAMILIES[family]) if B in M[c]]
        out, chosen = np.zeros(len(docs)), []
        for i, held in enumerate(docs):
            tr = [d for d in docs if d != held]
            best = min(cfgs, key=lambda c: (sum(M[c][B][d][1] for d in tr), c))
            chosen.append(best)
            out[i] = M[best][B][held][1]
        return out, chosen

    # A1/A2/A4: every fixed config and every LODO family at every budget.
    rows = []
    for B in budgets:
        if B not in M["lru"]:
            continue
        for c in FIXED:
            rows.append(dict(policy=c, B=B, **stats(vec(c, B), B)))
        for fam in FAMILIES:
            if not all(B in M[c] for c in FAMILIES[fam]):
                continue
            pol, chosen = lodo(fam, B)
            s = stats(pol, B)
            rows.append(dict(policy=f"{fam} [lodo]", B=B, **s,
                             chosen=" ".join(sorted(set(chosen)))))
    keys = ["policy", "B", "miss_rate", "G", "G_lo", "G_hi", "rep_drop", "rep_lo", "rep_hi",
            "better", "worst_rel", "chosen"]
    write(args.res / "checks_stats.csv", keys, [[r.get(k, "") for k in keys] for r in rows])
    report.append("== A1/A2/A4  G and repeat-miss drop vs LRU (doc bootstrap 95% CI), docs better of 14, worst doc rel. change")
    for r in rows:
        report.append(f"B={r['B']:>3} {r['policy']:<26} miss {100*r['miss_rate']:5.2f}%  G {r['G']:+.3f} "
                      f"[{r['G_lo']:+.3f},{r['G_hi']:+.3f}]  repdrop {100*r['rep_drop']:+5.1f}% "
                      f"[{100*r['rep_lo']:+.1f},{100*r['rep_hi']:+.1f}]  better {r['better']}/{len(docs)}  "
                      f"worst {100*r['worst_rel']:+.1f}%  {r.get('chosen', '')}")

    # A3: by task and by length band at the primary budget for the main contenders.
    report.append("\n== A3  by task / length band (B=96)")
    for key in ("task", "band"):
        for grp in sorted({meta[d][key] for d in docs}):
            idx = [i for i, d in enumerate(docs) if meta[d][key] == grp]
            for c in ("ema:0.5", "score", "infllm:0.9", "rank:4:2", "arc", "lru2"):
                if 96 not in M[c]:
                    continue
                lru, pol, opt = vec("lru", 96)[idx], vec(c, 96)[idx], vec("belady", 96)[idx]
                report.append(f"{key}={grp:<11} n={len(idx):>2} {c:<12} LRU {100*lru.sum()/vec('lru',96,0)[idx].sum():.2f}% "
                              f"G {(lru.sum()-pol.sum())/(lru.sum()-opt.sum()):+.3f} better {(pol<lru).sum()}/{len(idx)}")

    # B5: alpha sensitivity and LODO stability.
    report.append("\n== B5  EMA decay sensitivity (G, all docs) and LODO choices")
    for B in budgets:
        fam = [c for c in FAMILIES["ema(z,all)"] if B in M[c]]
        if not fam:
            continue
        lru, opt = vec("lru", B), vec("belady", B)
        curve = " ".join(f"{c.split(':')[1]}:{(lru.sum()-vec(c,B).sum())/(lru.sum()-opt.sum()):+.3f}" for c in fam)
        _, chosen = lodo("ema(z,all)", B, fam)
        report.append(f"B={B:>3} {curve}   lodo picks {dict((x, chosen.count(x)) for x in sorted(set(chosen)))}")

    # B6 + C10: per-layer allocation (LODO vs oracle), base LRU and base EMA, and additivity.
    report.append("\n== B6/C10  per-layer allocation: LODO vs oracle; additivity with the score rule")
    alloc_rows = []
    ema_grid = [c for c in ("ema:0.3", "ema:0.5", "ema:0.7", "ema:0.9") if all(g in ML[c] for g in GRID)]
    for B in (72, 80, 96, 128, 256):
        lru_tot = vec("lru", B).sum()
        opt_tot = vec("belady", B).sum()
        res = {}
        for base in ("lru", "ema"):
            lodo_m, oracle_m, allocs = 0, 0, []
            for held in docs:
                tr = [d for d in docs if d != held]
                if base == "lru":
                    cfg = "lru"
                else:
                    cfg = min(ema_grid, key=lambda c: (sum(M[c][B][d][1] for d in tr), c))
                curves = {l: {g: sum(ML[cfg][g][d, l] for d in tr) for g in GRID} for l in range(L)}
                a = kp.allocate(curves, L, L * B, GRID)
                allocs.append([a[l] for l in range(L)])
                lodo_m += sum(ML[cfg][a[l]][held, l] for l in range(L))
                oc = {l: {g: ML[cfg][g][held, l] for g in GRID} for l in range(L)}
                ao = kp.allocate(oc, L, L * B, GRID)
                oracle_m += sum(ML[cfg][ao[l]][held, l] for l in range(L))
            uni = lru_tot if base == "lru" else lodo("ema(z,all)", B, ema_grid)[0].sum()
            A = np.array(allocs)
            res[base] = (uni, lodo_m, oracle_m)
            share = (uni - lodo_m) / (uni - oracle_m) if uni != oracle_m else float("nan")
            alloc_rows.append((B, base, uni, lodo_m, oracle_m, share, float(A.std(0).mean()),
                               " ".join(str(int(x)) for x in np.median(A, 0))))
            report.append(f"B={B:>3} base={base:<4} uniform {uni:.0f}  alloc(lodo) {lodo_m:.0f}  alloc(oracle) {oracle_m:.0f}  "
                          f"lodo gets {100*share:.0f}% of oracle gain; per-layer std across folds {A.std(0).mean():.1f} slots")
        g = lambda m: (lru_tot - m) / (lru_tot - opt_tot)
        e_only, a_only, both = g(res["ema"][0]), g(res["lru"][1]), g(res["ema"][1])
        report.append(f"B={B:>3} G: ema {e_only:+.3f}  alloc+lru {a_only:+.3f}  alloc+ema {both:+.3f}  "
                      f"(sum of separate gains {e_only + a_only:+.3f})")
        alloc_rows.append((B, "G_ema|alloc_lru|alloc_ema", e_only, a_only, both, e_only + a_only, "", ""))
    write(args.res / "checks_alloc.csv", ["B", "base", "uniform", "alloc_lodo", "alloc_oracle",
                                          "share_of_oracle", "fold_std", "median_alloc"], alloc_rows)

    # C10: use-once (no-admission) evictions.
    by = args.res / "bypass.csv"
    if by.exists():
        report.append("\n== C10  evictions of pages admitted on the previous step and never hit (use-once)")
        agg = defaultdict(lambda: [0, 0])
        with open(by) as f:
            for r in csv.DictReader(f):
                a = agg[r["policy"], int(r["B"])]
                a[0] += int(r["evictions"])
                a[1] += int(r["use_once_evictions"])
        for (pol, B), (n, k) in sorted(agg.items()):
            report.append(f"B={B:>3} {pol:<7} use-once share {100*k/n:.1f}% of {n} evictions")

    learned = args.res / "learned.csv"
    if learned.exists():
        report.append("\n== learned next-use regression (LODO)")
        per = defaultdict(dict)
        with open(learned) as f:
            for r in csv.DictReader(f):
                per[r["model"], int(r["B"])][r["rid"] if "rid" in r else r["doc"]] = int(r["misses"])
        for (model, B), dm in sorted(per.items()):
            if set(dm) == set(docs):
                s = stats(np.array([dm[d] for d in docs], float), B)
                report.append(f"B={B:>3} linear-{model:<6} miss {100*s['miss_rate']:.2f}%  G {s['G']:+.3f} "
                              f"[{s['G_lo']:+.3f},{s['G_hi']:+.3f}]  better {s['better']}/{len(docs)}")

    (args.res / "checks_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
