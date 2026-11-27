"""followup-v3 A10: G with and without the documents whose generation stopped on the repetition (loop) rule.

  python -m analysis_v2.degenerate_v3 --out results_v2/ext/degenerate \
      --setting 64k_dev results_v2/ext/64k_dev/misses.csv.gz results_v2/provenance/manifests/64k_dev.jsonl \
      --setting ruler_dev results_v2/ext/ruler_dev/misses.csv.gz results_v2/provenance/manifests/ruler_dev.jsonl

Uses the replay rows of analysis_v2.explore as they are (no new replay). Families are re-tuned
leave-one-document-out inside each document subset (same rule as explore.py), bootstraps resample the
documents of the subset (2000, seed 0).
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis_v2.explore import FAMILIES, g_of

SHOW = ("score", "ema(lodo)", "ema_allrank(lodo)", "infllm_eq1:0.1", "infllm_lrus(lodo)")


def read(path):
    M = defaultdict(lambda: defaultdict(lambda: np.zeros(3, np.int64)))
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            M[r["config"], int(r["B"])][r["doc"]] += (int(r["accesses"]), int(r["misses"]), int(r["compulsory"]))
    return M


def summarize(M, docs, K=64):
    idx = np.random.default_rng(0).integers(0, len(docs), size=(2000, len(docs)))
    out = []
    for B in sorted({b for _, b in M}):
        per = {c: np.array([M[c, B][d][1] for d in docs]) for c, b in M if b == B}
        for fam, prefix in FAMILIES.items():
            cfgs = sorted(c for c in per if c.startswith(prefix))
            if not cfgs:
                continue
            v = []
            for d in docs:
                tr = [x for x in docs if x != d]
                best = min(cfgs, key=lambda c: (sum(M[c, B][x][1] for x in tr), c))
                v.append(M[best, B][d][1])
            per[fam] = np.array(v)
        lru, opt = per["lru"], per["belady"]
        for name in SHOW:
            if name not in per:
                continue
            p = per[name]
            gs = [g_of(lru[i].sum(), p[i].sum(), opt[i].sum()) for i in idx]
            lo, hi = np.nanpercentile(gs, [2.5, 97.5])
            out.append(dict(policy=name, B=B, B_over_K=B / K, G=g_of(lru.sum(), p.sum(), opt.sum()), G_lo=lo, G_hi=hi,
                            wins=int((p < lru).sum()), n_docs=len(docs),
                            lru_miss_rate=lru.sum() / sum(M["lru", B][d][0] for d in docs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", nargs=3, action="append", metavar=("NAME", "MISSES", "MANIFESTS"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for name, misses, mans in a.setting:
        stop = {}
        for line in open(mans):
            m = json.loads(line)
            stop[m["rid"]] = m["stop"]
        M = read(misses)
        docs = sorted(M["lru", 96])
        missing = [d for d in docs if d not in stop]
        if missing:
            raise SystemExit(f"{name}: no manifest for {missing}")
        subsets = {"all": docs, "without_loop_stops": [d for d in docs if stop[d] != "loop"],
                   "loop_stops_only": [d for d in docs if stop[d] == "loop"]}
        for sub, ds in subsets.items():
            for r in summarize(M, ds):
                rows.append(dict(setting=name, subset=sub, **r))
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["B"] in (96, 128) and r["policy"] in ("score", "ema(lodo)", "ema_allrank(lodo)"):
            print(f"{r['setting']:<10} {r['subset']:<20} B={r['B']:>3} {r['policy']:<18} G {r['G']:+.3f} "
                  f"[{r['G_lo']:+.3f},{r['G_hi']:+.3f}] {r['wins']}/{r['n_docs']}")


if __name__ == "__main__":
    main()
