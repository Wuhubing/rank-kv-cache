"""S4 (dev only): new baselines, budget sweep, alpha curves, factorials, statistics, noise injection.

  python -m analysis_v2.s4_dev replay   TRACES --out results_v2/dev/s4 [--procs 32]
  python -m analysis_v2.s4_dev summarize           --out results_v2/dev/s4
  python -m analysis_v2.s4_dev predict  TRACES --out results_v2/dev/prediction [--procs 32]
  python -m analysis_v2.s4_dev noise    TRACES --out results_v2/dev/noise [--procs 32]

replay writes misses.csv (doc, layer, config, B, accesses, misses, compulsory) for every config in
CONFIGS on the sweep budgets plus LRU on a fine grid (iso-miss capacity). summarize writes, per
experiment subfolder, CSV tables and PNG plots; families are tuned leave-one-document-out (minimum
training misses, ties by config name), exactly as efficient_dev.py does. Bootstraps resample
documents (2000 reps, seed 0) with the same resamples for every policy, so differences are paired.
"""
import argparse
import csv
import math
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import kvpolicy as kp

K = 64
SWEEP = [64, 80, 96, 128, 192, 256, 384, 512]          # B/K = 1, 1.25, 1.5, 2, 3, 4, 6, 8
FINE_LRU = list(range(64, 1025, 8))                       # iso-miss capacity
PRIMARY = 96
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
REPS, SEED = 2000, 0


def _configs():
    C = {}
    fam = {}

    def add(family, name, pol, **kw):
        C[name] = (pol, kw)
        fam.setdefault(family, []).append(name)
    # 6.1 existing (re-run on the sweep budgets; the efficient_dev/checks grids are run separately)
    for n in ("lru", "fifo", "lfu", "lru2", "arc", "score", "belady"):
        add(n, n, n)
    add("lru_hit_first", "lru_hit_first", "lru", tie="hit_first")
    for d in (0.1, 0.5, 0.8, 0.9, 0.95, 0.99):
        add("infllm_proxy_sel", f"infllm:{d}", "infllm", decay=d)          # existing InfLLM class
        add("dsum_softmax", f"dsum_softmax:{d}", "dsum", decay=d, v="softmax")
        add("dcount", f"dcount:{d}", "dsum", decay=d, v="indicator")
    # alpha curves: three signals, mode all
    for sig in ("raw", "z", "allrank"):
        for a in ALPHAS:
            add(f"ema_{sig}", f"ema_{sig}:{a}", "ema", alpha=a, signal=sig)
    # 6.2 new
    for s in range(5):
        add("random", f"random:{s}", "random", seed=s)
    add("arkvale_2_2", "arkvale:2:2", "arkvale", sink=2, win=2)
    add("arkvale_4_4", "arkvale:4:4", "arkvale", sink=4, win=4)
    for d in (0.0, 0.1, 0.3, 0.5, 0.8, 0.9, 0.95):
        add("infllm_eq1", f"infllm_eq1:{d}", "infllm_eq1", decay=d)
        add("infllm_lrus", f"infllm_lrus:{d}", "infllm_lrus", decay=d)
    add("sieve", "sieve", "sieve")
    add("s3fifo", "s3fifo", "s3fifo")
    for a in (0.3, 0.5, 0.7, 0.9):
        for rho in (0.0, 0.25, 0.5):
            for N in (0, 4, 16):
                add("co2", f"co2:{a}:{rho}:{N}", "co2", alpha=a, rho=rho, N=N)
        for r in (2, 4, 8):
            add(f"lazy_r{r}", f"lazy:{a}:{r}", "lazy", alpha=a, r=r)
    add("frozen", "frozen", "lazy", alpha=1.0, r=1)
    for a in (0.2, 0.4, 0.6, 0.8):
        for b in (0.1, 0.2, 0.3, 0.4, 0.5):
            for g in (0.0, 0.25, 0.5, 0.75):
                add("holt", f"holt:{a}:{b}:{g}", "holt", alpha=a, beta=b, gamma=g)
    for g in (0.5, 1.0, 2.0):
        add("powerlaw", f"powerlaw:{g}", "powerlaw", gamma=g)
    return C, fam


CONFIGS, FAMILIES = _configs()

_CACHE = {}


def load(path):
    if path not in _CACHE:
        _CACHE.clear()
        _CACHE[path] = kp.load(path)
    return _CACHE[path]


def run_job(job):
    path, config, budgets = job
    R, S, rid = load(path)
    pol, kw = CONFIGS[config] if config in CONFIGS else ("lru", {})
    out = []
    for B in budgets:
        for l in range(R.shape[1]):
            c = kp.simulate(R[:, l], S[:, l], B, pol, **kw)
            out.append((rid, l, config, B, c["accesses"], c["misses"], c["compulsory"]))
    return out


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def cmd_replay(args):
    paths = sorted(str(p) for p in args.traces.glob("*.npz"))
    jobs = [(p, c, SWEEP) for p in paths for c in CONFIGS]
    jobs += [(p, "lru_fine", [b for b in FINE_LRU if b not in SWEEP]) for p in paths]
    out = args.out / "misses.csv"
    n = 0
    with Pool(args.procs) as pool, open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc", "layer", "config", "B", "accesses", "misses", "compulsory"])
        for rows in pool.imap_unordered(run_job, jobs, chunksize=1):
            w.writerows(rows)
            n += 1
            if n % 200 == 0:
                print(f"{n}/{len(jobs)} jobs", flush=True)
                f.flush()


# ------------------------------------------------------------------------------ summarize

def read_misses(path):
    M = defaultdict(lambda: defaultdict(lambda: np.zeros(3, np.int64)))  # M[(cfg,B)][doc]
    L = defaultdict(lambda: np.zeros(3, np.int64))                        # per (cfg,B,layer)
    with open(path) as f:
        for r in csv.DictReader(f):
            cfg = "lru" if r["config"] == "lru_fine" else r["config"]
            v = np.array([int(r["accesses"]), int(r["misses"]), int(r["compulsory"])])
            M[cfg, int(r["B"])][r["doc"]] += v
            L[cfg, int(r["B"]), int(r["layer"])] += v
    return M, L


def boot_idx(n, reps=REPS, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, size=(reps, n))


def g_of(lru, pol, opt):
    d = lru - opt
    return (lru - pol) / d if d else float("nan")


def summarize_rows(per, docs, M, budgets):
    """per[(label, B)] = np.array of per-doc misses (doc order = docs)."""
    idx = boot_idx(len(docs))
    rows = []
    for (label, B), pol in sorted(per.items(), key=lambda x: (x[0][1], x[0][0])):
        lru = per["lru", B]
        opt = per["belady", B]
        acc = np.array([M["lru", B][d][0] for d in docs])
        comp = np.array([M["lru", B][d][2] for d in docs])
        g = g_of(lru.sum(), pol.sum(), opt.sum())
        gs = [g_of(lru[i].sum(), pol[i].sum(), opt[i].sum()) for i in idx]
        lo, hi = np.nanpercentile(gs, [2.5, 97.5])
        drop = 1 - (pol.sum() - comp.sum()) / (lru.sum() - comp.sum()) if lru.sum() != comp.sum() else float("nan")
        wins = int(((lru - pol) > 0).sum())
        losses = int(((lru - pol) < 0).sum())
        rows.append(dict(policy=label, B=B, B_over_K=B / K, misses=int(pol.sum()), accesses=int(acc.sum()),
                         miss_rate=pol.sum() / acc.sum(),
                         repeat_miss_rate=(pol.sum() - comp.sum()) / (acc.sum() - comp.sum()),
                         compulsory_share_of_misses=comp.sum() / pol.sum(),
                         G=g, G_lo=lo, G_hi=hi, repeat_miss_drop_vs_lru=drop,
                         docs_better_than_lru=wins, docs_worse_than_lru=losses))
    return rows


def lodo(M, docs, family, B, layers=None):
    """Per held-out doc: config chosen on the other docs (min misses, ties by name) and its misses."""
    cfgs = FAMILIES[family]
    out, chosen = [], []
    for d in docs:
        train = [x for x in docs if x != d]
        best = min(cfgs, key=lambda c: (sum(M[c, B][x][1] for x in train), c))
        chosen.append(best)
        out.append(M[best, B][d][1])
    return np.array(out), chosen


def iso_capacity(M, docs, misses_total):
    """LRU budget (linear interpolation on the fine LRU grid) with the same total misses."""
    grid = sorted(b for b in FINE_LRU)
    lru = [sum(M["lru", b][d][1] for d in docs) for b in grid]
    if misses_total > lru[0]:
        return float("nan")
    for (b0, m0), (b1, m1) in zip(zip(grid, lru), zip(grid[1:], lru[1:])):
        if m1 <= misses_total <= m0:
            return b0 + (b1 - b0) * (m0 - misses_total) / (m0 - m1) if m0 != m1 else b0
    return float("inf")


def cmd_summarize(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M, Lm = read_misses(args.out / "misses.csv")
    docs = sorted(M["lru", PRIMARY])
    nL = 1 + max(l for (_, _, l) in Lm)
    per, chosen_rows = {}, []
    for B in SWEEP:
        for c in ("lru", "belady"):
            per[c, B] = np.array([M[c, B][d][1] for d in docs])
        for fam, cfgs in FAMILIES.items():
            if fam in ("lru", "belady"):
                continue
            if len(cfgs) == 1:
                per[fam, B] = np.array([M[cfgs[0], B][d][1] for d in docs])
            else:
                per[f"{fam}(lodo)", B], ch = lodo(M, docs, fam, B)
                for d, c in zip(docs, ch):
                    chosen_rows.append((fam, B, d, c))
        # fixed headline configs (no tuning)
        for c in ("infllm_eq1:0.1", "infllm_lrus:0.1", "ema_z:0.5", "frozen"):
            per[c, B] = np.array([M[c, B][d][1] for d in docs])
        per["random(mean of 5 seeds)", B] = np.mean([[M[f"random:{s}", B][d][1] for d in docs]
                                                    for s in range(5)], axis=0)
    rows = summarize_rows(per, docs, M, SWEEP)
    for r in rows:
        r["iso_miss_lru_B"] = iso_capacity(M, docs, r["misses"])
        r["iso_capacity_ratio"] = r["iso_miss_lru_B"] / r["B"] if r["B"] else float("nan")
    hdr = list(rows[0])
    (args.out / "baselines").mkdir(parents=True, exist_ok=True)
    write(args.out / "baselines/summary.csv", hdr, [[r[h] for h in hdr] for r in rows])
    write(args.out / "baselines/lodo_choices.csv", ["family", "B", "doc", "chosen"], chosen_rows)
    # all fixed configs, untuned, every budget
    fixed = []
    for (cfg, B), dm in sorted(M.items(), key=lambda x: (x[0][1], x[0][0])):
        if B not in SWEEP:
            continue
        lru, opt = sum(M["lru", B][d][1] for d in docs), sum(M["belady", B][d][1] for d in docs)
        acc = sum(M["lru", B][d][0] for d in docs)
        m = sum(dm[d][1] for d in docs)
        fixed.append((cfg, B, m, acc, m / acc, g_of(lru, m, opt)))
    write(args.out / "baselines/all_configs.csv", ["config", "B", "misses", "accesses", "miss_rate", "G"], fixed)

    # budget sweep figure + iso capacity table
    sweep = args.out / "budget_sweep"
    sweep.mkdir(parents=True, exist_ok=True)
    show = ["lru", "belady", "score", "ema_z(lodo)", "ema_allrank(lodo)", "infllm_eq1:0.1", "infllm_lrus(lodo)",
            "arc", "s3fifo", "sieve", "random(mean of 5 seeds)", "fifo"]
    byk = {(r["policy"], r["B"]): r for r in rows}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name in show:
        xs = [B / K for B in SWEEP if (name, B) in byk]
        ax[0].plot(xs, [100 * byk[name, B]["miss_rate"] for B in SWEEP if (name, B) in byk], marker="o", ms=3, label=name)
        if name not in ("lru", "belady"):
            ax[1].plot(xs[1:], [byk[name, B]["G"] for B in SWEEP[1:] if (name, B) in byk], marker="o", ms=3, label=name)
    ax[0].set_xscale("log", base=2); ax[1].set_xscale("log", base=2)
    ax[0].set_xlabel("B/K"); ax[0].set_ylabel("miss rate (%)"); ax[1].set_xlabel("B/K"); ax[1].set_ylabel("G")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(sweep / "miss_budget.png", dpi=130); plt.close(fig)
    write(sweep / "iso_capacity.csv", ["policy", "B", "B_over_K", "misses", "iso_miss_lru_B", "iso_capacity_ratio"],
          [[r["policy"], r["B"], r["B_over_K"], r["misses"], r["iso_miss_lru_B"], r["iso_capacity_ratio"]]
           for r in rows if r["policy"] in show])

    # alpha curves
    ac = args.out / "alpha_curve"
    arows = []
    for sig in ("raw", "z", "allrank"):
        for a in ALPHAS:
            for B in SWEEP[1:]:
                m = per_doc = np.array([M[f"ema_{sig}:{a}", B][d][1] for d in docs])
                arows.append((sig, a, B, int(m.sum()), g_of(per["lru", B].sum(), m.sum(), per["belady", B].sum())))
    write(ac / "alpha_curve.csv", ["signal", "alpha", "B", "misses", "G"], arows)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for axx, B in zip(axs, (96, 128, 256)):
        for sig in ("raw", "z", "allrank"):
            pts = [(a, g) for s, a, b, _, g in arows if s == sig and b == B]
            axx.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, label=sig)
        for ref, ls in (("score", ":"), ("infllm_eq1:0.1", "--")):
            axx.axhline(byk[ref, B]["G"], ls=ls, color="gray", lw=1, label=ref)
        axx.set_title(f"B={B} (B/K={B/K:g})"); axx.set_xlabel("alpha (weight on history)")
        axx.axhline(0, color="k", lw=0.5)
    axs[0].set_ylabel("G"); axs[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ac / "alpha_curve.png", dpi=130); plt.close(fig)

    # factorials (CO2, lazy, holt): untuned grids at the primary budgets
    fac = args.out / "factorials"
    frows = []
    for cfg in CONFIGS:
        if cfg.startswith(("co2:", "lazy:", "holt:", "powerlaw:")) or cfg == "frozen":
            for B in (96, 128, 256):
                m = sum(M[cfg, B][d][1] for d in docs)
                frows.append((cfg.split(":")[0], cfg, B, m, g_of(per["lru", B].sum(), m, per["belady", B].sum())))
    write(fac / "factorial_grid.csv", ["family", "config", "B", "misses", "G"], frows)
    # CO2 main effects at B=96: mean G over the other factors
    eff = []
    for B in (96, 128, 256):
        g = {c: x for _, c, b, _, x in frows if b == B and c.startswith("co2:")}
        for fname, pos in (("alpha", 1), ("rho", 2), ("N", 3)):
            levels = sorted({c.split(":")[pos] for c in g}, key=float)
            for lv in levels:
                vals = [x for c, x in g.items() if c.split(":")[pos] == lv]
                eff.append(("co2", B, fname, lv, float(np.mean(vals)), float(np.max(vals))))
        h = {c: x for _, c, b, _, x in frows if b == B and c.startswith("holt:")}
        for fname, pos in (("alpha_new", 1), ("beta", 2), ("gamma", 3)):
            for lv in sorted({c.split(":")[pos] for c in h}, key=float):
                vals = [x for c, x in h.items() if c.split(":")[pos] == lv]
                eff.append(("holt", B, fname, lv, float(np.mean(vals)), float(np.max(vals))))
    write(fac / "main_effects.csv", ["family", "B", "factor", "level", "mean_G", "max_G"], eff)

    # per-layer miss and G at the primary budget
    pl = []
    best_ema = defaultdict(int)
    for fam, B, d, c in chosen_rows:
        if fam == "ema_z" and B == PRIMARY:
            best_ema[c] += 1
    ema_cfg = max(best_ema, key=best_ema.get)
    for l in range(nL):
        lru, opt, ema, sc = (Lm[c, PRIMARY, l][1] for c in ("lru", "belady", ema_cfg, "score"))
        acc = Lm["lru", PRIMARY, l][0]
        pl.append((l, acc, lru / acc, opt / acc, ema / acc, sc / acc, g_of(lru, ema, opt), g_of(lru, sc, opt)))
    write(args.out / "per_layer/per_layer_B96.csv",
          ["layer", "accesses", "lru_miss_rate", "belady_miss_rate", f"{ema_cfg}_miss_rate", "score_miss_rate",
           f"G_{ema_cfg}", "G_score"], pl)

    # statistics: per-doc G distribution + paired contrasts
    st = args.out / "stats"
    st.mkdir(parents=True, exist_ok=True)
    idx = boot_idx(len(docs))
    contrasts = []
    for B in (80, 96, 128, 256):
        lru, opt = per["lru", B], per["belady", B]
        ref = per["ema_z(lodo)", B]
        for other in ("score", "ema_allrank(lodo)", "ema_raw(lodo)", "infllm_eq1:0.1", "infllm_lrus(lodo)",
                      "infllm_eq1(lodo)", "dsum_softmax(lodo)", "holt(lodo)", "co2(lodo)", "arkvale_2_2",
                      "lazy_r4(lodo)", "powerlaw(lodo)", "arc", "s3fifo"):
            if (other, B) not in per:
                continue
            o = per[other, B]
            diff = g_of(lru.sum(), ref.sum(), opt.sum()) - g_of(lru.sum(), o.sum(), opt.sum())
            ds = [g_of(lru[i].sum(), ref[i].sum(), opt[i].sum()) - g_of(lru[i].sum(), o[i].sum(), opt[i].sum())
                  for i in idx]
            lo, hi = np.nanpercentile(ds, [2.5, 97.5])
            contrasts.append(("ema_z(lodo)", other, B, diff, lo, hi, int((o - ref > 0).sum()), len(docs)))
    write(st / "contrasts.csv", ["policy", "vs", "B", "G_diff", "lo", "hi", "docs_policy_fewer_misses", "n_docs"],
          contrasts)
    dl = []
    for B in (96, 128):
        lru, opt = per["lru", B], per["belady", B]
        for name in ("ema_z(lodo)", "score", "infllm_eq1:0.1"):
            for d, a, o, p in zip(docs, lru, opt, per[name, B]):
                dl.append((name, B, d, int(a), int(o), int(p), g_of(a, p, o)))
    write(st / "per_doc_G.csv", ["policy", "B", "doc", "lru", "belady", "misses", "G"], dl)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = ("ema_z(lodo)", "score", "infllm_eq1:0.1")
    data = [[x[6] for x in dl if x[0] == n and x[1] == 96] for n in names]
    ax.boxplot(data); ax.set_xticks(range(1, len(names) + 1), names, fontsize=7)
    for i, ys in enumerate(data, 1):
        ax.scatter(np.full(len(ys), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(ys)), ys, s=10)
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("per-document G (B=96)")
    fig.tight_layout(); fig.savefig(st / "per_doc_G_B96.png", dpi=130); plt.close(fig)
    print(f"summarized {len(docs)} docs; primary EMA config by LODO majority: {ema_cfg}")
    for r in rows:
        if r["B"] in (96, 128):
            print(f"B={r['B']:>3} {r['policy']:<28} miss {100*r['miss_rate']:.2f}%  G {r['G']:+.3f} "
                  f"[{r['G_lo']:+.3f},{r['G_hi']:+.3f}] wins {r['docs_better_than_lru']}/{len(docs)}")


# ------------------------------------------------------------------------------ prediction quality

def predict_doc(args):
    """Per layer: predicted-set overlap (EMA top-K over all eligible pages vs next step's top-K),
    adjacent-step overlap, churn (1 - Jaccard), for alpha in a small grid."""
    path, alphas = args
    R, S, rid = load(path)
    T, L, Kk = R.shape
    out = []
    for l in range(L):
        sets = [set(int(p) for p in R[t, l] if p >= 0) for t in range(T)]
        adj = [len(sets[t] & sets[t + 1]) / Kk for t in range(T - 1)]
        jac = [1 - len(sets[t] & sets[t + 1]) / len(sets[t] | sets[t + 1]) for t in range(T - 1)]
        row = [rid, l, T, float(np.mean(adj)), float(np.mean(jac))]
        for a in alphas:
            e = None
            ov = []
            for t in range(T - 1):
                s = S[t, l]
                n = int(np.isfinite(s).sum())
                ranked = R[t, l]
                kth, top = s[ranked[-1]], s[ranked[0]]
                z = (s[:n] - kth) / (top - kth + 1e-6)
                if e is None:
                    e = z.astype(np.float64)
                else:
                    m = len(e)
                    e = np.concatenate([a * e + (1 - a) * z[:m], z[m:]])
                pred = np.argsort(-e, kind="stable")[:Kk]
                ov.append(len(set(pred.tolist()) & sets[t + 1]) / Kk)
            row.append(float(np.mean(ov)))
        out.append(row)
    return out


def collect_masked(path):
    """LRU decisions at B=PRIMARY with the per-decision flag complete_horizon (t + H < T)."""
    import efficient_dev as ed
    return ed.collect(path, PRIMARY, track_horizon=True)


def mask_tail(d):
    """Drop decisions in the last H steps of a trace (followup-v3 A7): there 'not re-selected within H' is
    censored by the end of the trace, not observed."""
    m = d["complete_horizon"]
    return dict(d, X=d["X"][m], soon=d["soon"][m])


def cmd_predict(args):
    import efficient_dev as ed
    from gap_analysis import auc as tie_auc
    paths = sorted(str(p) for p in args.traces.glob("*.npz"))
    alphas = [0.0, 0.3, 0.5, 0.7, 0.9]
    args.out.mkdir(parents=True, exist_ok=True)
    with Pool(args.procs) as pool:
        rows = [r for part in pool.imap_unordered(predict_doc, [(p, alphas) for p in paths]) for r in part]
        write(args.out / "overlap_by_layer.csv",
              ["doc", "layer", "steps", "adjacent_overlap", "topk_churn"] + [f"ema_pred_overlap_a{a}" for a in alphas],
              rows)
        # incremental predictive power of the score signal beyond recency and frequency
        raw = pool.map(collect_masked, paths)
    res = []
    for label, data in (("masked_last_H", [mask_tail(d) for d in raw]), ("unmasked_v2", raw)):
        res += auc_tables(data, label)
    write(args.out / "incremental_auc.csv", ["mask", "kind", "item", "value", "sd"], res)
    print(*res, sep="\n")


def auc_tables(data, mask_label):
    import efficient_dev as ed
    from gap_analysis import auc as tie_auc
    feats = list(kp.FEATURES)
    ia, iz, ize, ic = feats.index("age"), feats.index("z"), feats.index("z_ema"), feats.index("log_cnt")
    res = []

    def logit_fit(X, y, iters=25, lam=1e-3):
        X = np.column_stack([X, np.ones(len(X))])
        mu, sd = X[:, :-1].mean(0), X[:, :-1].std(0) + 1e-9
        X[:, :-1] = (X[:, :-1] - mu) / sd
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-X @ w))
            Hm = X.T @ (X * (p * (1 - p))[:, None]) + lam * np.eye(X.shape[1])
            w -= np.linalg.solve(Hm, X.T @ (p - y) + lam * w)
        return w, mu, sd

    def logit_pred(model, X):
        w, mu, sd = model
        return np.column_stack([(X - mu) / sd, np.ones(len(X))]) @ w

    sets = {"recency": [ia], "frequency": [ic], "recency+freq": [ia, ic], "+z": [ia, ic, iz], "+z_ema": [ia, ic, ize],
            "+z+z_ema": [ia, ic, iz, ize]}
    names = [d["rid"] for d in data]
    for name, cols in sets.items():
        aucs = []
        for i, held in enumerate(names):
            tr = [j for j in range(len(data)) if j != i]
            Xtr = np.vstack([data[j]["X"][:, cols] for j in tr])
            ytr = np.concatenate([data[j]["soon"] for j in tr]).astype(float)
            sub = np.random.default_rng(i).choice(len(ytr), min(len(ytr), 300000), replace=False)
            model = logit_fit(Xtr[sub], ytr[sub])
            pred = logit_pred(model, data[i]["X"][:, cols])
            aucs.append(tie_auc(pred, data[i]["soon"].astype(bool)))
        res.append(("lodo_logistic_auc", name, float(np.nanmean(aucs)), float(np.nanstd(aucs))))
    # stratified: within (age decile x count bin) strata, AUC of z_ema for reuse within H
    X = np.vstack([d["X"] for d in data])
    y = np.concatenate([d["soon"] for d in data]).astype(bool)
    ab = np.searchsorted(np.quantile(X[:, ia], np.linspace(0, 1, 11)[1:-1]), X[:, ia])
    cb = np.searchsorted(np.quantile(X[:, ic], [0.25, 0.5, 0.75]), X[:, ic])
    for fi, fname in ((ize, "z_ema"), (iz, "z")):
        num = den = 0.0
        for a in range(10):
            for c in range(4):
                m = (ab == a) & (cb == c)
                if m.sum() > 50 and 0 < y[m].sum() < m.sum():
                    v = tie_auc(X[m, fi], y[m])
                    if v is not None:
                        num += v * m.sum()
                        den += m.sum()
        res.append(("stratified_auc_age_decile_x_count", fname, num / den if den else float("nan"), ""))
    res.append(("n_decisions", "lru_B96_candidates", int(len(y)), ""))
    res.append(("base_rate", f"reused_within_{ed.H}", float(y.mean()), ""))
    return [(mask_label, *r) for r in res]


# ------------------------------------------------------------------------------ noise injection

def noise_job(args):
    path, sigma, seed, cfgs = args
    R, S, rid = load(path)
    rng = np.random.default_rng(seed)
    out = []
    for l in range(R.shape[1]):
        s = S[:, l].astype(np.float64)
        sd = np.nanstd(s, axis=1, keepdims=True)
        Sn = s + sigma * sd * rng.standard_normal(s.shape)
        for name, (pol, kw) in cfgs.items():
            c = kp.simulate(R[:, l], Sn, PRIMARY, pol, **kw)
            out.append((rid, l, sigma, seed, name, c["misses"]))
    return out


def cmd_noise(args):
    """Gaussian noise (sigma x per-step score std) added to the scores the policies read; the
    selection (ranked) is unchanged, so LRU and Belady are unchanged."""
    paths = sorted(str(p) for p in args.traces.glob("*.npz"))
    cfgs = {"ema_z:0.5": ("ema", dict(alpha=0.5)), "ema_z:0.7": ("ema", dict(alpha=0.7)),
            "score": ("score", {}), "ema_allrank:0.5": ("ema", dict(alpha=0.5, signal="allrank"))}
    jobs = [(p, s, seed, cfgs) for p in paths for s in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0)
            for seed in ((0,) if s == 0 else (0, 1, 2))]
    args.out.mkdir(parents=True, exist_ok=True)
    with Pool(args.procs) as pool:
        rows = [r for part in pool.imap_unordered(noise_job, jobs) for r in part]
        base = [r for part in pool.imap_unordered(run_job, [(p, c, [PRIMARY]) for p in paths for c in ("lru", "belady")])
                for r in part]
    write(args.out / "noise_misses.csv", ["doc", "layer", "sigma", "seed", "config", "misses"], rows)
    lru = sum(r[5] for r in base if r[2] == "lru")
    opt = sum(r[5] for r in base if r[2] == "belady")
    agg = defaultdict(list)
    tot = defaultdict(int)
    for d, l, s, seed, c, m in rows:
        tot[c, s, seed] += m
    for (c, s, seed), m in tot.items():
        agg[c, s].append(g_of(lru, m, opt))
    summ = [(c, s, float(np.mean(v)), float(np.min(v)), float(np.max(v)), len(v)) for (c, s), v in sorted(agg.items())]
    write(args.out / "noise_summary.csv", ["config", "sigma", "G_mean", "G_min", "G_max", "seeds"], summ)
    print(*summ, sep="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["replay", "summarize", "predict", "noise"])
    ap.add_argument("traces", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=32)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dict(replay=cmd_replay, summarize=cmd_summarize, predict=cmd_predict, noise=cmd_noise)[args.cmd](args)


if __name__ == "__main__":
    main()
