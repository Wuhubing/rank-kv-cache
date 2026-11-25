"""D5 (followup-v3): protection zone and pinned pages, overall and by the layers' adjacent-step overlap.

  python -m analysis_v3.d5_summary --misses results_v3/d5/d5_dev/misses.csv.gz --stats results_v3/b2/dev.csv \
      --out results_v3/d5/dev

For each protected / pinned configuration and its base (LRU or EMA alpha 0.5 without protection): pooled G and paired
dG = G(config) - G(base) with a document bootstrap; the same within three groups of layers split at the terciles of the
layers' mean adjacent-step top-K overlap (from B2), and per layer the number of layers where the config has fewer misses than
its base on at least 12 of 14 documents.
"""
import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np


def g_of(l, p, o):
    return (l - p) / (l - o) if l != o else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--misses", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    M = defaultdict(int)
    with gzip.open(a.misses, "rt") if str(a.misses).endswith(".gz") else open(a.misses) as f:
        for r in csv.DictReader(f):
            M[r["config"], int(r["B"]), r["doc"], int(r["layer"])] = int(r["misses"])
    ov = defaultdict(list)
    for r in csv.DictReader(open(a.stats)):
        ov[int(r["layer"])].append(float(r["overlap"]))
    layer_ov = {l: float(np.mean(v)) for l, v in ov.items()}
    L = len(layer_ov)
    cut = np.percentile(list(layer_ov.values()), [33.3, 66.7])
    group = {l: ("low" if o <= cut[0] else "mid" if o <= cut[1] else "high") for l, o in layer_ov.items()}
    docs = sorted({k[2] for k in M})
    cfgs = sorted({k[0] for k in M})
    budgets = sorted({k[1] for k in M})
    idx = np.random.default_rng(0).integers(0, len(docs), size=(2000, len(docs)))
    rows, layer_rows = [], []
    for B in budgets:
        for c in cfgs:
            if not (c.startswith("prot_") or c.startswith("pin_")):
                continue
            base = "lru" if "_lru_" in c else "ema_fixed:0.5"
            for grp in ("all", "low", "mid", "high"):
                ls = [l for l in range(L) if grp == "all" or group[l] == grp]

                def tot(name):
                    return np.array([sum(M[name, B, d, l] for l in ls) for d in docs], float)
                lu, bu, x, y = tot("lru"), tot("belady"), tot(c), tot(base)
                dG = g_of(lu.sum(), x.sum(), bu.sum()) - g_of(lu.sum(), y.sum(), bu.sum())
                ds = [g_of(lu[i].sum(), x[i].sum(), bu[i].sum()) - g_of(lu[i].sum(), y[i].sum(), bu[i].sum()) for i in idx]
                rows.append(dict(B=B, config=c, base=base, layers=grp, n_layers=len(ls),
                                 G=g_of(lu.sum(), x.sum(), bu.sum()), G_base=g_of(lu.sum(), y.sum(), bu.sum()), dG=dG,
                                 lo=np.nanpercentile(ds, 2.5), hi=np.nanpercentile(ds, 97.5),
                                 docs_fewer=int((x < y).sum())))
            better = 0
            for l in range(L):
                x = np.array([M[c, B, d, l] for d in docs])
                y = np.array([M[base, B, d, l] for d in docs])
                if (x < y).sum() >= 12:
                    better += 1
                    layer_rows.append(dict(B=B, config=c, layer=l, overlap=layer_ov[l], docs_fewer=int((x < y).sum())))
            rows[-4]["layers_better_on_12_of_14_docs"] = better
    a.out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(a.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if layer_rows:
        with open(a.out / "layers_better.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(layer_rows[0]))
            w.writeheader()
            w.writerows(layer_rows)
    print("overlap terciles", np.round(cut, 3))
    for r in rows:
        if r["layers"] == "all" and r["B"] in (96, 128):
            print(f"B={r['B']} {r['config']:<18} G {r['G']:+.3f} (base {r['G_base']:+.3f}) dG {r['dG']:+.3f} "
                  f"[{r['lo']:+.3f},{r['hi']:+.3f}] layers better {r.get('layers_better_on_12_of_14_docs')}")


if __name__ == "__main__":
    main()
