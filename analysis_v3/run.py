"""Generic replay + summary for the followup-v3 experiments (analysis_v3.experiments).

  python -m analysis_v3.run replay EXP TRACES --out DIR [--procs 16] [--budgets ...]
  python -m analysis_v3.run summarize EXP --out DIR [--misses more.csv ...]

replay writes DIR/misses.csv (doc, layer, config, B, accesses, misses, compulsory). summarize writes summary.csv
(G, document-bootstrap CI, wins, iso-miss capacity is not computed here), contrasts.csv (paired ΔG with CI and the
number of documents where each side has fewer misses), lodo_choices.csv and all_configs.csv (every untuned config).
LODO: per held-out document, the config of the family with the fewest misses on the other documents (ties by name),
as analysis_v2.s4_dev.lodo. Bootstrap: 2000 resamples of documents, seed 0, the same resamples for every policy.
"""
import argparse
import csv
import gzip
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import kvpolicy as kp
import kvpolicy_v3  # noqa: F401  (registers the v3 policies)
from analysis_v3 import experiments

REPS, SEED = 2000, 0
_CACHE = {}


def load(path, signal):
    key = (path, signal)
    if key not in _CACHE:
        _CACHE.clear()
        R, S, rid = kp.load(path)
        if signal == "attn_all":
            with np.load(path, allow_pickle=False) as z:
                A = z["attn_all"].astype(np.float32)
            if A.shape != S.shape or not np.array_equal(np.isfinite(A), np.isfinite(S)):
                raise ValueError(f"{path}: attn_all does not cover the same pages as scores")
            S = A
        _CACHE[key] = (R, S, rid)
    return _CACHE[key]


def run_job(job):
    path, exp_name, config, budgets = job
    e = experiments.get(exp_name)
    R, S, rid = load(path, e.signal)
    pol, kw = e.configs[config]
    out = []
    for B in budgets:
        for l in range(R.shape[1]):
            c = kp.simulate(R[:, l], S[:, l], B, pol, **kw)
            out.append((rid, l, config, B, c["accesses"], c["misses"], c["compulsory"]))
    return out


def cmd_replay(a):
    e = experiments.get(a.exp)
    budgets = a.budgets or e.budgets
    paths = sorted(str(p) for p in a.traces.glob("*.npz"))
    jobs = [(p, a.exp, c, budgets) for p in paths for c in e.configs]
    a.out.mkdir(parents=True, exist_ok=True)
    with Pool(a.procs) as pool, open(a.out / "misses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc", "layer", "config", "B", "accesses", "misses", "compulsory"])
        for n, rows in enumerate(pool.imap(run_job, jobs, chunksize=4), 1):
            w.writerows(rows)
            if n % 200 == 0:
                print(f"{n}/{len(jobs)} jobs", flush=True)
                f.flush()
    print(f"done {len(jobs)} jobs", flush=True)


def read_misses(paths):
    M = defaultdict(lambda: defaultdict(lambda: np.zeros(3, np.int64)))
    for path in paths:
        op = gzip.open if str(path).endswith(".gz") else open
        with op(path, "rt") as f:
            for r in csv.DictReader(f):
                M[r["config"], int(r["B"])][r["doc"]] += np.array(
                    [int(r["accesses"]), int(r["misses"]), int(r["compulsory"])])
    return M


def g_of(lru, pol, opt):
    d = lru - opt
    return (lru - pol) / d if d else float("nan")


def boot_idx(n):
    return np.random.default_rng(SEED).integers(0, n, size=(REPS, n))


def lodo(M, docs, cfgs, B):
    out, chosen = [], []
    for d in docs:
        tr = [x for x in docs if x != d]
        best = min(cfgs, key=lambda c: (sum(M[c, B][x][1] for x in tr), c))
        chosen.append(best)
        out.append(M[best, B][d][1])
    return np.array(out, float), chosen


def per_doc_table(M, e, budgets, docs):
    """per[(label, B)] = per-document misses; label = config name for single configs, '<family>(lodo)' else."""
    per, choices = {}, []
    for B in budgets:
        for fam, cfgs in e.families.items():
            if not all((c, B) in M for c in cfgs):
                continue
            if len(cfgs) == 1:
                per[cfgs[0], B] = np.array([M[cfgs[0], B][d][1] for d in docs], float)
            else:
                per[f"{fam}(lodo)", B], ch = lodo(M, docs, cfgs, B)
                choices += [dict(family=fam, B=B, doc=d, chosen=c) for d, c in zip(docs, ch)]
    return per, choices


def contrast(per, idx, a, b, B):
    lru, opt, x, y = per["lru", B], per["belady", B], per[a, B], per[b, B]
    d = g_of(lru.sum(), x.sum(), opt.sum()) - g_of(lru.sum(), y.sum(), opt.sum())
    ds = [g_of(lru[i].sum(), x[i].sum(), opt[i].sum()) - g_of(lru[i].sum(), y[i].sum(), opt[i].sum()) for i in idx]
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    return dict(a=a, b=b, B=B, dG=d, lo=lo, hi=hi, docs_a_fewer=int((x < y).sum()), docs_b_fewer=int((y < x).sum()),
                n=len(lru))


def summary_rows(per, M, docs, idx):
    rows = []
    for (label, B), pol in sorted(per.items(), key=lambda x: (x[0][1], x[0][0])):
        lru, opt = per["lru", B], per["belady", B]
        acc = sum(M["lru", B][d][0] for d in docs)
        comp = sum(M["lru", B][d][2] for d in docs)
        gs = [g_of(lru[i].sum(), pol[i].sum(), opt[i].sum()) for i in idx]
        lo, hi = np.nanpercentile(gs, [2.5, 97.5])
        rows.append(dict(policy=label, B=B, misses=int(pol.sum()), accesses=int(acc), miss_rate=pol.sum() / acc,
                         G=g_of(lru.sum(), pol.sum(), opt.sum()), G_lo=lo, G_hi=hi,
                         repeat_miss_drop_vs_lru=1 - (pol.sum() - comp) / (lru.sum() - comp),
                         docs_better_than_lru=int((pol < lru).sum()), docs_worse_than_lru=int((pol > lru).sum()),
                         n_docs=len(docs)))
    return rows


def write(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(e, M, out, budgets=None, extra_contrasts=()):
    docs = sorted(M["lru", e.budgets[0]])
    budgets = budgets or [B for B in e.budgets if ("lru", B) in M]
    idx = boot_idx(len(docs))
    per, choices = per_doc_table(M, e, budgets, docs)
    rows = summary_rows(per, M, docs, idx)
    con = []
    for B in budgets:
        for a_, b_ in list(e.contrasts) + list(extra_contrasts):
            if (a_, B) in per and (b_, B) in per:
                con.append(contrast(per, idx, a_, b_, B))
    allc = []
    for (cfg, B), dm in sorted(M.items(), key=lambda x: (x[0][1], x[0][0])):
        if B in budgets and cfg in e.configs:
            m = sum(dm[d][1] for d in docs)
            allc.append(dict(config=cfg, B=B, misses=int(m), G=g_of(per["lru", B].sum(), m, per["belady", B].sum())))
    write(out / "summary.csv", rows)
    write(out / "contrasts.csv", con)
    write(out / "lodo_choices.csv", choices)
    write(out / "all_configs.csv", allc)
    return per, rows, con, docs


def cmd_summarize(a):
    e = experiments.get(a.exp)
    M = read_misses([a.out / "misses.csv"] + list(a.misses or []))
    per, rows, con, docs = summarize(e, M, a.out)
    print(f"{e.name}: {len(docs)} docs")
    for r in rows:
        if r["B"] in (96, 128):
            print(f"B={r['B']:>3} {r['policy']:<32} G {r['G']:+.4f} [{r['G_lo']:+.4f},{r['G_hi']:+.4f}] "
                  f"wins {r['docs_better_than_lru']}/{r['n_docs']}")
    for c in con:
        if c["B"] in (96, 128):
            print(f"B={c['B']} {c['a']} - {c['b']}: {c['dG']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] "
                  f"{c['docs_a_fewer']}/{c['n']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["replay", "summarize"])
    ap.add_argument("exp", choices=sorted(experiments.REGISTRY))
    ap.add_argument("traces", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--budgets", type=int, nargs="*")
    ap.add_argument("--misses", type=Path, nargs="*")
    a = ap.parse_args()
    dict(replay=cmd_replay, summarize=cmd_summarize)[a.cmd](a)


if __name__ == "__main__":
    main()
