"""Summary of analysis_v3.learn: G (LODO, BC and DAgger) with document-bootstrap CI against the LRU/Belady rows of the b1
replay, and the mean standardised coefficients (linear) or permutation importances (trees).

  python -m analysis_v3.learn_summary --learn out/v3/learn_linear_dev --b1 out/v3/b1_dev/misses.csv
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", type=Path, required=True)
    ap.add_argument("--b1", type=Path, required=True)
    a = ap.parse_args()
    M = defaultdict(lambda: defaultdict(int))
    op = gzip.open if str(a.b1).endswith(".gz") else open
    with op(a.b1, "rt") as f:
        for r in csv.DictReader(f):
            if r["config"] in ("lru", "belady", "pow_norm_kth:1.0", "ema_z:0.5"):
                M[r["config"], r["B"]][r["doc"]] += int(r["misses"])
    rows = list(csv.DictReader(open(a.learn / "learned.csv")))
    out = {}
    for B in sorted({r["B"] for r in rows}):
        docs = sorted(M["lru", B])
        lru = np.array([M["lru", B][d] for d in docs], float)
        bel = np.array([M["belady", B][d] for d in docs], float)
        dd = {r["doc"]: r for r in rows if r["B"] == B}
        idx = np.random.default_rng(0).integers(0, len(docs), (2000, len(docs)))

        def G(v, i=slice(None)):
            return float((lru[i].sum() - v[i].sum()) / (lru[i].sum() - bel[i].sum()))
        res = {}
        for k in ("misses_bc", "misses_dagger"):
            v = np.array([int(dd[d][k]) for d in docs], float)
            res[k] = dict(G=G(v), ci=[float(x) for x in np.percentile([G(v, i) for i in idx], [2.5, 97.5])],
                          wins=int((v < lru).sum()))
        pw = np.array([M["pow_norm_kth:1.0", B][d] for d in docs], float)
        v = np.array([int(dd[d]["misses_dagger"]) for d in docs], float)
        res["dagger_minus_powerlaw"] = dict(dG=G(v) - G(pw), ci=[float(x) for x in np.percentile(
            [G(v, i) - G(pw, i) for i in idx], [2.5, 97.5])], docs_fewer=int((v < pw).sum()))
        out[B] = res
    info = json.load(open(a.learn / "models.json"))
    key = "standardised" if "standardised" in info[0] else "importance"
    for B in out:
        st = [i[key] for i in info if str(i["B"]) == B and isinstance(i.get(key), dict) and "error" not in i[key]]
        if st:
            out[B][key] = {k: float(np.mean([s[k] for s in st])) for k in st[0]}
    (a.learn / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
