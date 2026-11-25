"""followup-v3 part A (fixes to replication-v2), dev only: power-law kernel variants (A1), standard Holt
(A2), the 97 % baseline (A3) and S3-FIFO per Algorithm 1 (A5).

  python -m analysis_v2.fix_v3 replay TRACES --out results_v2/dev/fixes [--procs 16]
  python -m analysis_v2.fix_v3 summarize --out results_v2/dev/fixes --old results_v2/dev/s4/misses.csv.gz

replay writes misses.csv (same columns as s4_dev) for the configs below plus LRU and Belady; summarize
reads it together with the replication-v2 table (--old, for the EMA, Holt and current-score rows) and
writes CSV tables. Families are tuned leave-one-document-out exactly as in s4_dev (s4_dev.lodo); bootstraps
resample documents (2000 reps, seed 0) with the same resamples for every policy (paired).
"""
import argparse
import csv
import gzip
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import kvpolicy as kp
import kvpolicy_v2 as k2
from analysis_v2 import s4_dev as s4

A_BUDGETS = [80, 96, 128, 256]
GAMMAS = (0.5, 1.0, 2.0)


def _configs():
    C, fam = {}, {}

    def add(family, name, pol, budgets, **kw):
        C[name] = (pol, kw, budgets)
        fam.setdefault(family, []).append(name)
    for n in ("lru", "belady"):
        add(n, n, n, s4.SWEEP)
    add("s3fifo_alg1", "s3fifo_alg1", "s3fifo", s4.SWEEP)
    for norm in (False, True):
        for zero in ("kth", "median", "min"):
            for g in GAMMAS:
                f = f"pl_{'norm' if norm else 'sum'}_{zero}"
                add(f, f"{f}:{g}", "powerlaw", A_BUDGETS, gamma=g, normalize=norm, zero=zero)
    for g in GAMMAS:  # the reviewer's "z shifted by +1" (zero point one unit below the K-th score)
        add("pl_sum_kth_shift1", f"pl_sum_kth_shift1:{g}", "powerlaw", A_BUDGETS, gamma=g, shift=1.0)
    for a in (0.2, 0.4, 0.6, 0.8):  # same grid as the PRR-style Holt in s4_dev
        for b in (0.1, 0.2, 0.3, 0.4, 0.5):
            for g in (0.0, 0.25, 0.5, 0.75):
                add("holt_std", f"holt_std:{a}:{b}:{g}", "holt_std", A_BUDGETS, alpha=a, beta=b, gamma=g)
    return C, fam


CONFIGS, FAMILIES = _configs()


def run_job(job):
    path, config = job
    R, S, rid = s4.load(path)
    pol, kw, budgets = CONFIGS[config]
    out = []
    for B in budgets:
        for l in range(R.shape[1]):
            c = kp.simulate(R[:, l], S[:, l], B, pol, **kw)
            out.append((rid, l, config, B, c["accesses"], c["misses"], c["compulsory"]))
    return out


def cmd_replay(args):
    paths = sorted(str(p) for p in args.traces.glob("*.npz"))
    jobs = [(p, c) for p in paths for c in CONFIGS]
    with Pool(args.procs) as pool, open(args.out / "misses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc", "layer", "config", "B", "accesses", "misses", "compulsory"])
        for n, rows in enumerate(pool.imap_unordered(run_job, jobs, chunksize=1), 1):
            w.writerows(rows)
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} jobs", flush=True)
                f.flush()


def read(paths):
    M = defaultdict(lambda: defaultdict(lambda: np.zeros(3, np.int64)))
    for path in paths:
        op = gzip.open if str(path).endswith(".gz") else open
        with op(path, "rt") as f:
            for r in csv.DictReader(f):
                cfg = "lru" if r["config"] == "lru_fine" else r["config"]
                M[cfg, int(r["B"])][r["doc"]] += np.array([int(r["accesses"]), int(r["misses"]), int(r["compulsory"])])
    return M


def per_doc(M, docs, cfg, B):
    return np.array([M[cfg, B][d][1] for d in docs], float)


def contrast(per, idx, a, b, B):
    """Paired G(a) - G(b) with its document-bootstrap CI and the number of documents where a has fewer misses."""
    lru, opt, x, y = per["lru", B], per["belady", B], per[a, B], per[b, B]
    d = s4.g_of(lru.sum(), x.sum(), opt.sum()) - s4.g_of(lru.sum(), y.sum(), opt.sum())
    ds = [s4.g_of(lru[i].sum(), x[i].sum(), opt[i].sum()) - s4.g_of(lru[i].sum(), y[i].sum(), opt[i].sum()) for i in idx]
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    return dict(a=a, b=b, B=B, dG=d, lo=lo, hi=hi, docs_a_fewer=int((x < y).sum()), docs_b_fewer=int((y < x).sum()),
                n=len(lru))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_summarize(args):
    M = read([args.old, args.out / "misses.csv"])
    s4.FAMILIES.update(FAMILIES)
    docs = sorted(M["lru", 96])
    idx = s4.boot_idx(len(docs))
    per, choices = {}, []
    budgets = sorted({B for (c, B) in M if c == "s3fifo_alg1"})
    for B in budgets:
        for c in ("lru", "belady", "score", "s3fifo", "s3fifo_alg1"):
            if (c, B) in M:
                per[c, B] = per_doc(M, docs, c, B)
        fams = ["ema_z", "ema_allrank", "ema_raw", "holt", "powerlaw"] + [f for f in FAMILIES if f not in
                                                                         ("lru", "belady", "s3fifo_alg1")]
        for fam in fams:
            if all((c, B) in M for c in s4.FAMILIES[fam]):
                per[f"{fam}(lodo)", B], ch = s4.lodo(M, docs, fam, B)
                choices += [dict(family=fam, B=B, doc=d, chosen=c) for d, c in zip(docs, ch)]
        for c in list(FAMILIES.get("holt_std", [])) + [c for f in FAMILIES if f.startswith("pl_") for c in FAMILIES[f]]:
            if (c, B) in M:
                per[c, B] = per_doc(M, docs, c, B)
        for a in (0.0,):
            for sig in ("z", "allrank", "raw"):
                if (f"ema_{sig}:{a}", B) in M:
                    per[f"ema_{sig}:{a}", B] = per_doc(M, docs, f"ema_{sig}:{a}", B)
    rows = s4.summarize_rows({k: v for k, v in per.items() if k[1] in budgets}, docs, M, budgets)
    write(args.out / "summary.csv", rows)
    write(args.out / "lodo_choices.csv", choices)
    con = []
    for B in budgets:
        pairs = [("s3fifo_alg1", "s3fifo"), ("s3fifo_alg1", "lru")]
        if B in A_BUDGETS:
            pairs += [("holt(lodo)", "ema_z(lodo)"), ("holt_std(lodo)", "ema_z(lodo)"), ("holt_std(lodo)", "holt(lodo)"),
                      ("ema_z(lodo)", "score"), ("ema_allrank(lodo)", "score"), ("ema_raw(lodo)", "score"),
                      ("ema_z(lodo)", "ema_allrank(lodo)")]
            for f in FAMILIES:
                if f.startswith("pl_"):
                    pairs += [(f"{f}(lodo)", "ema_z(lodo)"), (f"{f}(lodo)", "powerlaw(lodo)")]
        con += [contrast(per, idx, a, b, B) for a, b in pairs if (a, B) in per and (b, B) in per]
    write(args.out / "contrasts.csv", con)
    # A3: share of the history gain kept by the ordering, against two baselines
    a3 = []
    for B in A_BUDGETS:
        g = {k: s4.g_of(per["lru", B].sum(), per[k, B].sum(), per["belady", B].sum())
             for k in ("score", "ema_z(lodo)", "ema_allrank(lodo)", "ema_z:0.0", "ema_allrank:0.0", "ema_raw:0.0")}
        a3.append(dict(B=B, G_score=g["score"], G_z_alpha0=g["ema_z:0.0"], G_allrank_alpha0=g["ema_allrank:0.0"],
                       G_raw_alpha0=g["ema_raw:0.0"], G_z=g["ema_z(lodo)"], G_allrank=g["ema_allrank(lodo)"],
                       share_vs_lru=g["ema_allrank(lodo)"] / g["ema_z(lodo)"],
                       gain_z_over_score=g["ema_z(lodo)"] - g["score"],
                       gain_allrank_over_score=g["ema_allrank(lodo)"] - g["score"],
                       share_vs_score=(g["ema_allrank(lodo)"] - g["score"]) / (g["ema_z(lodo)"] - g["score"])))
    write(args.out / "a3_history_gain.csv", a3)
    mix = [dict(gamma=g, max_rel_error=k2.powerlaw_mixture(g)[2],
                terms=int((k2.powerlaw_mixture(g)[1] > 0).sum())) for g in GAMMAS]
    write(args.out / "powerlaw_mixture_error.csv", mix)
    for r in rows:
        if r["B"] in (96, 128):
            print(f"B={r['B']:>3} {r['policy']:<32} G {r['G']:+.4f} [{r['G_lo']:+.4f},{r['G_hi']:+.4f}] "
                  f"wins {r['docs_better_than_lru']}")
    for c in con:
        if c["B"] in (96, 128):
            print(f"B={c['B']} {c['a']} - {c['b']}: {c['dG']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] "
                  f"{c['docs_a_fewer']}/{c['n']}")
    print(*a3, *mix, sep="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["replay", "summarize"])
    ap.add_argument("traces", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--old", type=Path, default=Path("results_v2/dev/s4/misses.csv.gz"))
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dict(replay=cmd_replay, summarize=cmd_summarize)[args.cmd](args)


if __name__ == "__main__":
    main()
