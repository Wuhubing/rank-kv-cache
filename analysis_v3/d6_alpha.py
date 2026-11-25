"""D6 (followup-v3): why the best alpha grows with B, and a tuning-free per-layer alpha rule.

  python -m analysis_v3.d6_alpha --synth results_v3/b2/synthetic --stats results_v3/b2/dev.csv \
      --misses results_v2/dev/s4/misses.csv.gz --out results_v3/d6/dev
  (8B: --stats results_v3/b2/8b.csv --misses <b1 8b misses with the ema_z grid>)

1. From the synthetic model (analysis_v3.b2_model curves): for every regime and B/K, the alpha with the most avoided misses;
   the rule alpha*(s, B/K) = clip(c0 + c1 * s + c2 * log2(B/K), 0, 0.95) with s = 1 - (lag-16 autocorrelation of the
   unselected pages' percentile) is fitted by least squares on the SYNTHETIC optima only (no dev data).
2. Applied to the real traces: each layer gets alpha from its own lag-16 autocorrelation measured on the document itself
   (no tuning; the statistic is computable online), rounded to the nearest grid value; misses = sum over layers of the
   EMA(alpha_layer) misses of that layer (layers are independent streams). Compared with: one alpha per budget chosen LODO,
   per-layer alpha chosen LODO, and fixed alpha = 0.5.
"""
import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def g_of(l, p, o):
    return (l - p) / (l - o) if l != o else float("nan")


def fit_rule(synth):
    rows = list(csv.DictReader(open(Path(synth) / "curves_summary.csv")))
    stats = json.loads((Path(synth) / "regimes.json").read_text())["stats"]
    pts = []
    for reg in {r["regime"] for r in rows}:
        st = stats[reg]
        ac16 = float(np.mean([v["ac_lag16"] for v in st.values()]))
        for ratio in sorted({float(r["B_over_K"]) for r in rows}):
            ema = [(float(r["param"]), float(r["G"])) for r in rows
                   if r["regime"] == reg and float(r["B_over_K"]) == ratio and r["policy"] == "ema"]
            if not ema:
                continue
            best = max(ema, key=lambda x: (x[1], -x[0]))
            pts.append((1 - ac16, math.log2(ratio), best[0], reg))
    X = np.array([[1, s, lb] for s, lb, _, _ in pts])
    y = np.array([a for _, _, a, _ in pts])
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return c, pts


def rule(c, ac16, ratio):
    a = c[0] + c[1] * (1 - ac16) + c[2] * math.log2(ratio)
    a = min(max(a, 0.0), 0.95)
    return min(GRID, key=lambda g: abs(g - a))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--misses", type=Path, nargs="+", required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=[80, 96, 128, 256])
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    c, pts = fit_rule(a.synth)
    st = {(r["doc"], int(r["layer"])): float(r["ac_lag16"]) for r in csv.DictReader(open(a.stats))}
    M = defaultdict(int)          # (doc, layer, config, B) -> misses
    for path in a.misses:
        op = gzip.open if str(path).endswith(".gz") else open
        with op(path, "rt") as f:
            for r in csv.DictReader(f):
                cfg = r["config"]
                if cfg in ("lru", "belady") or cfg.startswith("ema_z:"):
                    M[r["doc"], int(r["layer"]), cfg, int(r["B"])] = int(r["misses"])
    docs = sorted({d for d, _ in st})
    L = 1 + max(l for _, l in st)
    out, chosen = [], []
    idx = np.random.default_rng(0).integers(0, len(docs), size=(2000, len(docs)))
    for B in a.budgets:
        ratio = B / a.K
        per = defaultdict(list)
        for d in docs:
            train = [x for x in docs if x != d]
            lru = sum(M[d, l, "lru", B] for l in range(L))
            bel = sum(M[d, l, "belady", B] for l in range(L))
            glob = min(GRID, key=lambda g: (sum(M[x, l, f"ema_z:{g}", B] for x in train for l in range(L)), g))
            per_layer = [min(GRID, key=lambda g: (sum(M[x, l, f"ema_z:{g}", B] for x in train), g)) for l in range(L)]
            rl = [rule(c, st[d, l], ratio) for l in range(L)]
            per["lru"].append(lru)
            per["belady"].append(bel)
            per["alpha_lodo_global"].append(sum(M[d, l, f"ema_z:{glob}", B] for l in range(L)))
            per["alpha_lodo_per_layer"].append(sum(M[d, l, f"ema_z:{per_layer[l]}", B] for l in range(L)))
            per["alpha_rule"].append(sum(M[d, l, f"ema_z:{rl[l]}", B] for l in range(L)))
            per["alpha_0.5"].append(sum(M[d, l, "ema_z:0.5", B] for l in range(L)))
            chosen.append(dict(B=B, doc=d, lodo_global=glob, rule_mean=float(np.mean(rl)), rule_min=min(rl), rule_max=max(rl)))
        per = {k: np.array(v, float) for k, v in per.items()}
        lu, bu = per["lru"], per["belady"]
        for k in ("alpha_lodo_global", "alpha_lodo_per_layer", "alpha_rule", "alpha_0.5"):
            v = per[k]
            gs = [g_of(lu[i].sum(), v[i].sum(), bu[i].sum()) for i in idx]
            ref = per["alpha_lodo_global"]
            ds = [g_of(lu[i].sum(), v[i].sum(), bu[i].sum()) - g_of(lu[i].sum(), ref[i].sum(), bu[i].sum()) for i in idx]
            out.append(dict(B=B, method=k, G=g_of(lu.sum(), v.sum(), bu.sum()), lo=np.percentile(gs, 2.5),
                            hi=np.percentile(gs, 97.5), dG_vs_lodo_global=g_of(lu.sum(), v.sum(), bu.sum())
                            - g_of(lu.sum(), ref.sum(), bu.sum()), dG_lo=np.percentile(ds, 2.5), dG_hi=np.percentile(ds, 97.5),
                            docs_fewer_than_lodo=int((v < ref).sum())))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rule.json").write_text(json.dumps(dict(coef=c.tolist(), points=pts), indent=1))
    for name, data in (("summary.csv", out), ("chosen.csv", chosen)):
        with open(a.out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print("rule coef", np.round(c, 3))
    for r in out:
        print(f"B={r['B']:>3} {r['method']:<22} G {r['G']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] dG vs LODO {r['dG_vs_lodo_global']:+.4f} "
              f"[{r['dG_lo']:+.4f},{r['dG_hi']:+.4f}]")


if __name__ == "__main__":
    main()
