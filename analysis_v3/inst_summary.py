"""Summaries of analysis_v3.instrument outputs (B3 errors, B5 phases, B6 spatial, B7 staleness, D4 frequency, D5 pins).

  python -m analysis_v3.inst_summary results_v3/inst/<model> [--overlap results_v3/b2/<model>.csv]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis_v3.instrument import PCT_BINS, RES_BINS, RET_BINS


def rd(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def g_of(l, p, o):
    return (l - p) / (l - o) if l != o else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    a = ap.parse_args()
    F = a.folder
    out = {}
    # B5 phases
    ph = rd(F / "phase.csv")
    docs = sorted({r["doc"] for r in ph})
    idx = np.random.default_rng(0).integers(0, len(docs), size=(2000, len(docs)))
    tab = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
    for r in ph:
        tab[r["phase"]][r["doc"]] += [int(r["steps"]), int(r["lru"]), int(r["belady"]), int(r["score"]), int(r["ema"])]
    phases = {}
    for p, dm in tab.items():
        A = np.array([dm[d] for d in docs])
        present = A[:, 0] > 0
        row = dict(docs_with_phase=int(present.sum()), steps=int(A[:, 0].sum()))
        if present.sum() >= 2:
            Ap = A[present]
            sub = np.random.default_rng(0).integers(0, len(Ap), size=(2000, len(Ap)))
            for j, n in ((3, "score"), (4, "ema")):
                g = g_of(Ap[:, 1].sum(), Ap[:, j].sum(), Ap[:, 2].sum())
                gs = [g_of(Ap[i, 1].sum(), Ap[i, j].sum(), Ap[i, 2].sum()) for i in sub]
                row[f"G_{n}"] = g
                row[f"G_{n}_ci"] = [float(np.nanpercentile(gs, 2.5)), float(np.nanpercentile(gs, 97.5))]
            row["lru_misses_per_step_layer"] = float(Ap[:, 1].sum() / Ap[:, 0].sum())
            row["belady_misses_per_step_layer"] = float(Ap[:, 2].sum() / Ap[:, 0].sum())
        phases[p] = row
    out["B5_phases"] = phases
    # B3 errors
    er = rd(F / "errors.csv")
    agg = defaultdict(lambda: np.zeros(2))
    for r in er:
        agg[r["group"], int(r["bin"])] += [int(r["victims"]), int(r["errors"])]
    names = {"res": RES_BINS, "pct": PCT_BINS, "ret": RET_BINS}
    errs = {}
    for (g, b), (n, e) in sorted(agg.items()):
        if g in names:
            bins = names[g]
            lab = f"[{bins[b]}, {bins[b + 1]})" if b + 1 < len(bins) else f">= {bins[b]}" if g != "ret" else "never"
        else:
            lab = {"phase": ["thinking", "answer"], "all": ["all"]}[g][b]
        errs.setdefault(g, []).append(dict(bin=lab, victims=int(n), errors=int(e), error_rate=e / n if n else None))
    tot = agg["all", 0]
    out["B3_errors"] = dict(total_victims=int(tot[0]), total_errors=int(tot[1]), error_rate=tot[1] / tot[0], groups=errs)
    # share of errors per group bin
    for g in ("res", "pct", "ret"):
        for row in errs.get(g, []):
            row["share_of_errors"] = row["errors"] / tot[1] if tot[1] else None
    # B6 spatial
    sp = rd(F / "spatial.csv")
    s = np.array([[float(r[k]) for k in ("repeat_misses", "within1_prev_sel", "within2_prev_sel", "random_pages",
                                         "random_within1", "random_within2")] for r in sp])
    aucs = np.array([float(r["neighbour_score_auc"]) for r in sp])
    out["B6_spatial"] = dict(repeat_miss_within1=s[:, 1].sum() / s[:, 0].sum(), repeat_miss_within2=s[:, 2].sum() / s[:, 0].sum(),
                             random_within1=s[:, 4].sum() / s[:, 3].sum(), random_within2=s[:, 5].sum() / s[:, 3].sum(),
                             neighbour_score_auc_mean=float(np.nanmean(aucs)))
    # B7 staleness
    st = rd(F / "stale.csv")
    stale = {}
    for v in ("sel_only", "full"):
        rs = [r for r in st if r["variant"] == v]
        stale[v] = {k: float(np.nanmean([float(r[k]) for r in rs])) for k in
                    ("mean_staleness", "median_staleness", "p90_staleness", "spearman_value_vs_current_z")}
    out["B7_staleness"] = stale
    # D4 frequency
    fq = rd(F / "freq.csv")
    n = sum(int(r["decisions"]) for r in fq)
    out["D4_frequency"] = dict(decisions=n, share_all_equal=sum(int(r["decisions_all_equal"]) for r in fq) / n,
                               mean_var_log_count=float(np.average([float(r["mean_var_log_count"]) for r in fq],
                                                                   weights=[int(r["decisions"]) for r in fq])),
                               mean_range_log_count=float(np.average([float(r["mean_range_log_count"]) for r in fq],
                                                                     weights=[int(r["decisions"]) for r in fq])))
    # D5 pins
    pn = rd(F / "pins.csv")
    out["D5_pins"] = {k: float(np.mean([float(r[k]) for r in pn])) for k in
                      ("page0", "page1", "page2", "page3", "recent1", "recent2", "recent3", "recent4")}
    (F / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
