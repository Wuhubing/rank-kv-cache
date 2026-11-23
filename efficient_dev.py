"""Dev-only study of replacement policies on longctx_v1 npz traces (run where the traces are).

Outputs (CSV) under --out:
  misses.csv      doc, layer, config, B, accesses, misses, compulsory  (every fixed config)
  lodo.csv        per held-out doc: chosen config per family, its misses (leave-one-doc-out)
  learned.csv     per held-out doc: linear next-use regression (BC, DAgger) misses
  alloc.csv       per held-out doc: per-layer budget allocation (base lru / best score policy)
  gap.csv         LRU eviction decisions at the primary budget: feature AUCs and error stats
  bypass.csv      share of evictions that evict a page admitted on the previous step (use-once)
  summary.csv     family x B: misses, miss rate, G and document-bootstrap CI, repeat-miss drop
Hyperparameters are chosen only on the 13 training documents of each fold.
"""
import argparse
import bisect
import csv
import itertools
import math
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import kvpolicy as kp

GRID = [64, 72, 80, 88, 96, 104, 112, 128, 160, 192, 256]
CURVE = [72, 80, 96, 128, 256]
PRIMARY = 96
LEARN_BUDGETS = (96, 128, 256)
FIXED = {"lru": {}, "fifo": {}, "lfu": {}, "lru2": {}, "arc": {}, "score": {}, "belady": {}}
FAMILIES = {
    "rank": {f"rank:{d}:{b}": dict(delta=d, beta=b) for d in (0, 4, 8) for b in (0, 2, 4)},
    "ema": {f"ema:{a}": dict(alpha=a) for a in (0.3, 0.5, 0.7, 0.9)},
    "hybrid": {f"hybrid:{w}:{a}": dict(w=w, alpha=a) for w in (1, 4, 16, 64) for a in (0, 0.5, 0.8)},
}
CONFIGS = {**{k: (k, v) for k, v in FIXED.items()},
           **{name: (fam, kw) for fam, d in FAMILIES.items() for name, kw in d.items()}}
H = 32  # "reused soon" horizon in decode steps for the gap analysis


def policy_of(config):
    return config.split(":")[0]


def run_doc(args):
    path, configs, budgets = args
    R, S, rid = kp.load(path)
    rows = []
    for config in configs:
        name = policy_of(config)
        kw = CONFIGS[config][1]
        for B in budgets:
            for l in range(R.shape[1]):
                c = kp.simulate(R[:, l], S[:, l], B, name, **kw)
                rows.append((rid, l, config, B, c["accesses"], c["misses"], c["compulsory"]))
    return rows


def next_use_table(rows):
    uses = defaultdict(list)
    for t, r in enumerate(rows):
        for p in r:
            if p >= 0:
                uses[int(p)].append(t)
    return uses


def collect(path, B, coef=None, layers=None):
    """Run the linear policy (coef None = LRU behaviour) and return per-decision features,
    log next-use targets, and LRU-victim flags, plus misses per layer."""
    R, S, rid = kp.load(path)
    T = R.shape[0]
    Xs, ys, victim, soon = [], [], [], []
    misses = {}
    for l in (range(R.shape[1]) if layers is None else layers):
        uses = next_use_table(R[:, l])
        log = []
        chosen = []

        def on_evict(t, victims, pol):
            chosen.append(set(victims))
        c = kp.simulate(R[:, l], S[:, l], B, "linear", coef=coef, collect=log, on_evict=on_evict)
        misses[l] = (c["accesses"], c["misses"], c["compulsory"])
        for (X, cands, t), vs in zip(log, chosen):
            d = []
            for p in cands:
                u = uses[p]
                i = bisect.bisect_right(u, t)
                d.append(u[i] - t if i < len(u) else T - t + 512)
            d = np.array(d, float)
            Xs.append(X)
            ys.append(np.log1p(np.minimum(d, 1024)))
            victim.append(np.array([p in vs for p in cands]))
            soon.append(d <= H)
    cat = (lambda a: np.concatenate(a) if a else np.zeros(0))
    return dict(rid=rid, X=np.vstack(Xs) if Xs else np.zeros((0, len(kp.FEATURES))),
                y=cat(ys), victim=cat(victim), soon=cat(soon), misses=misses)


def bypass_doc(path, B, name):
    R, S, rid = kp.load(path)
    n = once = 0
    for l in range(R.shape[1]):
        box = [0, 0]

        def cb(t, victims, pol):
            box[0] += len(victims)
            box[1] += sum(pol.last[v] == t - 1 and not pol.hit[v] for v in victims)
        kp.simulate(R[:, l], S[:, l], B, name, on_evict=cb)
        n += box[0]
        once += box[1]
    return rid, B, name, n, once


def ridge(X, y, lam=1.0, sub=200000, seed=0):
    if len(y) > sub:
        idx = np.random.default_rng(seed).choice(len(y), sub, replace=False)
        X, y = X[idx], y[idx]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    coef = w / sd
    return np.append(coef, y.mean() - mu @ coef)


def auc(score, label):
    label = np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(score, kind="stable"), kind="stable") + 1
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/efficient_dev"))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--skip-grid", action="store_true", help="reuse misses.csv")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.traces.glob("*.npz"))
    pool = Pool(args.procs)

    # 1. Every fixed config on the full budget grid.
    mpath = args.out / "misses.csv"
    if not args.skip_grid:
        jobs = [(str(p), [c], GRID) for p in paths for c in CONFIGS]
        rows = [r for part in pool.imap_unordered(run_doc, jobs) for r in part]
        write(mpath, ["doc", "layer", "config", "B", "accesses", "misses", "compulsory"], rows)
    M = defaultdict(lambda: defaultdict(dict))  # M[config][B][(doc, layer)] = (acc, miss, comp)
    with open(mpath) as f:
        for r in csv.DictReader(f):
            M[r["config"]][int(r["B"])][r["doc"], int(r["layer"])] = (
                int(r["accesses"]), int(r["misses"]), int(r["compulsory"]))
    docs = sorted({d for d, _ in M["lru"][PRIMARY]})
    L = 1 + max(l for _, l in M["lru"][PRIMARY])

    def doc_miss(config, B, doc, budgets=None):
        return sum(M[config][B][doc, l][1] for l in range(L))

    # 2. Leave-one-document-out choice inside each tuned family.
    lodo = []
    for fam, cfgs in FAMILIES.items():
        for B in GRID:
            for held in docs:
                train = [d for d in docs if d != held]
                best = min(cfgs, key=lambda c: (sum(doc_miss(c, B, d) for d in train), c))
                lodo.append((fam, B, held, best, doc_miss(best, B, held)))
    write(args.out / "lodo.csv", ["family", "B", "doc", "chosen", "misses"], lodo)

    # 3. Per-layer budget allocation (LODO): curves from training docs, total = L * B.
    best_score_family = "hybrid"
    alloc_rows = []
    for base in ("lru", "best_hybrid"):
        for B in CURVE:
            for held in docs:
                train = [d for d in docs if d != held]
                if base == "lru":
                    cfg = "lru"
                else:
                    cfg = min(FAMILIES[best_score_family],
                              key=lambda c: (sum(doc_miss(c, B, d) for d in train), c))
                curves = {l: {g: sum(M[cfg][g][d, l][1] for d in train) for g in GRID} for l in range(L)}
                a = kp.allocate(curves, L, L * B, GRID)
                miss = sum(M[cfg][a[l]][held, l][1] for l in range(L))
                alloc_rows.append((base, B, held, cfg, miss, sum(a.values()),
                                   " ".join(str(a[l]) for l in range(L))))
    write(args.out / "alloc.csv", ["base", "B", "doc", "config", "misses", "slots", "per_layer"], alloc_rows)

    # 4. Gap analysis + behaviour-cloned / DAgger next-use regression.
    learned = []
    gap_rows = []
    for B in LEARN_BUDGETS:
        data = dict(zip([p.name for p in paths], pool.starmap(collect, [(str(p), B) for p in paths])))
        if B == PRIMARY:
            X = np.vstack([d["X"] for d in data.values()])
            soon = np.concatenate([d["soon"] for d in data.values()])
            vic = np.concatenate([d["victim"] for d in data.values()])
            y = np.concatenate([d["y"] for d in data.values()])
            for j, f in enumerate(kp.FEATURES):
                gap_rows.append(("auc_soon", f, auc(X[:, j], soon), ""))
            gap_rows.append(("lru_victims", "n", int(vic.sum()), ""))
            gap_rows.append(("lru_victims", "reused_within_H", float(soon[vic].mean()), H))
            gap_rows.append(("kept_candidates", "reused_within_H", float(soon[~vic].mean()), H))
            for j, f in enumerate(kp.FEATURES):
                for grp, m in (("victim_soon", vic & soon), ("victim_late", vic & ~soon),
                               ("kept_soon", ~vic & soon), ("kept_late", ~vic & ~soon)):
                    gap_rows.append(("mean", f"{f}|{grp}", float(X[m, j].mean()) if m.any() else "", int(m.sum())))
            corr = [float(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])]
            for f, c in zip(kp.FEATURES, corr):
                gap_rows.append(("corr_log_next_use", f, c, ""))
        names = list(data)
        for held in names:
            train = [n for n in names if n != held]
            Xtr = np.vstack([data[n]["X"] for n in train])
            ytr = np.concatenate([data[n]["y"] for n in train])
            coef = ridge(Xtr, ytr)
            res = collect(str(args.traces / held), B, coef)
            learned.append(("bc", B, res["rid"], sum(m[1] for m in res["misses"].values()),
                            " ".join(f"{c:.4g}" for c in coef)))
            aggX, aggy = [Xtr], [ytr]
            for it in (1, 2):  # DAgger: add states visited by the student, Bélády-like labels
                visits = pool.starmap(collect, [(str(args.traces / n), B, coef) for n in train])
                aggX += [v["X"] for v in visits]
                aggy += [v["y"] for v in visits]
                coef = ridge(np.vstack(aggX), np.concatenate(aggy), seed=it)
            res = collect(str(args.traces / held), B, coef)
            learned.append(("dagger", B, res["rid"], sum(m[1] for m in res["misses"].values()),
                            " ".join(f"{c:.4g}" for c in coef)))
            print("learned", B, held, learned[-2][3], learned[-1][3], flush=True)
    write(args.out / "gap.csv", ["kind", "item", "value", "note"], gap_rows)
    write(args.out / "learned.csv", ["model", "B", "doc", "misses", "coef"], learned)

    # 5. Use-once evictions (the only "no-admission" action available under HiSparse semantics).
    by = pool.starmap(bypass_doc, [(str(p), B, nm) for p in paths for B in LEARN_BUDGETS[:2]
                                   for nm in ("lru", "belady")])
    write(args.out / "bypass.csv", ["doc", "B", "policy", "evictions", "use_once_evictions"], by)

    # 6. Summary with document-clustered bootstrap CIs.
    per = defaultdict(dict)  # per[(label, B)][doc] = misses
    for c in FIXED:
        for B in GRID:
            for d in docs:
                per[c, B][d] = doc_miss(c, B, d)
    for fam, B, d, _, m in lodo:
        per[f"{fam}(lodo)", B][d] = m
    for base, B, d, _, m, _, _ in alloc_rows:
        per[f"alloc+{base}", B][d] = m
    for model, B, d, m, _ in learned:
        per[f"linear-{model}", B][d] = m
    acc = {B: {d: sum(M["lru"][B][d, l][0] for l in range(L)) for d in docs} for B in GRID}
    comp = {B: {d: sum(M["lru"][B][d, l][2] for l in range(L)) for d in docs} for B in GRID}
    out = []
    for (label, B), dm in sorted(per.items(), key=lambda x: (x[0][1], x[0][0])):
        if set(dm) != set(docs):
            continue
        lru = [per["lru", B][d] for d in docs]
        opt = [per["belady", B][d] for d in docs]
        pol = [dm[d] for d in docs]
        A, C = sum(acc[B].values()), sum(comp[B].values())
        g = (sum(lru) - sum(pol)) / (sum(lru) - sum(opt)) if sum(lru) != sum(opt) else float("nan")
        lo, hi = kp.bootstrap(lru, pol, opt)
        out.append((label, B, B / 64, sum(pol), A, sum(pol) / A, (sum(pol) - C) / (A - C), g, lo, hi,
                    1 - (sum(pol) - C) / (sum(lru) - C) if sum(lru) != C else float("nan")))
    write(args.out / "summary.csv", ["policy", "B", "B_over_K", "misses", "accesses", "miss_rate",
                                     "repeat_miss_rate", "G", "G_lo", "G_hi", "repeat_miss_drop_vs_lru"], out)
    for r in out:
        if r[1] in CURVE:
            print(f"B={r[1]:>3} {r[0]:<20} miss {100*r[5]:.2f}%  G {r[7]:+.3f} [{r[8]:+.3f},{r[9]:+.3f}]  "
                  f"repeat-drop {100*r[10]:+.1f}%")


if __name__ == "__main__":
    main()
