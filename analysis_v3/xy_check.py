"""Check the key numbers of Xiaoyang Wu's PRs #9/#10 from his committed result files (read-only; files are extracted from
refs/remotes/pr/9 and pr/10 into a scratch folder, not copied into this branch).

  python -m analysis_v3.xy_check --dir /data/weibing/tmp/xy --out results_v3/external_xiaoyang/check.json

Recomputes: (a) MPS held-out G of the EMA (CI, wins) and allocation + EMA − EMA at B=96, plus the budget curve, from
mps_heldout/misses.csv; (b) the same for the new documents in v2_ood/misses.csv, split by task (NarrativeQA vs truncated
GovReport); (c) the transfer share and step speedups of batch_bench.csv from its raw time columns; (d) the decision-change rate
and the repeat-miss difference from solo_rank_dev/event_counts.csv. Bootstrap: documents, 2000 resamples, seed 0.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def g_of(l, p, o):
    return (l - p) / (l - o) if l != o else float("nan")


def held(path, docs_filter=None):
    M = defaultdict(lambda: defaultdict(int))
    for r in csv.DictReader(open(path)):
        if docs_filter and not docs_filter(r["doc"]):
            continue
        M[r["config"], int(r["B"])][r["doc"]] += int(r["misses"])
    out = {}
    for B in sorted({b for _, b in M}):
        docs = sorted(M["lru", B])
        if not docs:
            continue
        a = {c: np.array([M[c, B][d] for d in docs], float) for (c, b) in M if b == B}
        idx = np.random.default_rng(0).integers(0, len(docs), (2000, len(docs)))
        lu, bu = a["lru"], a["belady"]

        def G(v, i=slice(None)):
            return float(g_of(lu[i].sum(), v[i].sum(), bu[i].sum()))
        res = {"n_docs": len(docs)}
        for c, v in a.items():
            if c in ("lru", "belady"):
                continue
            res[c] = dict(G=G(v), ci=[float(x) for x in np.percentile([G(v, i) for i in idx], [2.5, 97.5])],
                          wins=int((v < lu).sum()))
        if "alloc_ema" in a and "ema" in a:
            d = [G(a["alloc_ema"], i) - G(a["ema"], i) for i in idx]
            res["alloc_minus_ema"] = dict(dG=G(a["alloc_ema"]) - G(a["ema"]), ci=[float(x) for x in np.percentile(d, [2.5, 97.5])])
        out[B] = res
    return out


def batch(path):
    rows = list(csv.DictReader(open(path)))
    out = []
    lru = {(r["B"], r["batch"]): r for r in rows if r["policy"] == "lru"}
    for r in rows:
        dec, sc, tr = float(r["decode_ms"]), float(r["scoring_ms"]), float(r["transfer_ms_mean"])
        step = dec + sc + tr
        l = lru[r["B"], r["batch"]]
        lstep = float(l["decode_ms"]) + float(l["scoring_ms"]) + float(l["transfer_ms_mean"])
        out.append(dict(B=r["B"], batch=r["batch"], policy=r["policy"], step_recomputed=step, step_reported=float(r["step_ms"]),
                        share_recomputed=tr / step, share_reported=float(r["transfer_share"]),
                        speedup_recomputed=lstep / step - 1, speedup_reported=float(r["speedup_vs_lru"]),
                        us_per_page_recomputed=1000 * tr / (float(r["pages_per_step"]) or float("nan")),
                        us_per_page_reported=float(r["us_per_page"])))
    return out


def rank(path):
    agg = defaultdict(lambda: np.zeros(4))
    for r in csv.DictReader(open(path)):
        agg[r["beta"]] += [int(r["evictions"]), int(r["changed"]), int(r["base_repeats"]), int(r["rank_repeats"])]
    return {b: dict(evictions=int(v[0]), changed=int(v[1]), change_rate=v[1] / v[0], base_repeats=int(v[2]),
                    rank_repeats=int(v[3]), saved=int(v[2] - v[3]), saved_share=(v[2] - v[3]) / v[2]) for b, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = a.dir
    res = dict(
        a_mps_heldout=held(d / "mps_heldout_misses.csv"),
        b_narrativeqa=held(d / "v2_ood_misses.csv", lambda x: x.startswith("narrativeqa")),
        b_govreport_truncated=held(d / "v2_ood_misses.csv", lambda x: not x.startswith("narrativeqa")),
        c_batch=batch(d / "batch_bench.csv"),
        d_rank=rank(d / "event_counts.csv"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for k in ("a_mps_heldout", "b_narrativeqa", "b_govreport_truncated"):
        r = res[k][96]
        print(k, r["n_docs"], {c: (round(v["G"], 3), [round(x, 3) for x in v["ci"]], v["wins"]) for c, v in r.items()
                                if isinstance(v, dict) and "G" in v}, "alloc-ema", round(r["alloc_minus_ema"]["dG"], 3),
              [round(x, 3) for x in r["alloc_minus_ema"]["ci"]])
    mx = max(abs(r["share_recomputed"] - r["share_reported"]) for r in res["c_batch"])
    ms = max(abs(r["speedup_recomputed"] - r["speedup_reported"]) for r in res["c_batch"])
    print("batch: max |share diff|", mx, "max |speedup diff|", ms)
    for r in res["c_batch"]:
        if r["B"] == "96" and r["batch"] in ("1", "8", "32", "64"):
            print(" ", r["batch"], r["policy"], round(r["share_recomputed"], 4), round(r["speedup_recomputed"], 4),
                  round(r["us_per_page_recomputed"], 2))
    print("rank", res["d_rank"])


if __name__ == "__main__":
    main()
