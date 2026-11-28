"""B4 read-out: how well the Quest score and the true attention mass of UNSELECTED pages predict re-selection by Quest.

  python -m analysis_v3.b4_signal data/dev_attn_all --out results_v3/b4/signal.csv

For every 4th step t and layer, over the pages that are eligible and not selected at t: Spearman(Quest score, true
attention mass); AUC of each signal for "selected again within 32 steps" (only t + 32 < T); and the same over all pages.
"""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from gap_analysis import auc as tie_auc


def rank(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def job(path):
    with np.load(path, allow_pickle=False) as z:
        R, S, A, rid = z["ranked"], z["scores"].astype(np.float32), z["attn_all"].astype(np.float32), str(z["rid"])
    T, L, K = R.shape
    out = []
    for l in range(L):
        sel = np.zeros((T, S.shape[2]), bool)
        for t in range(T):
            sel[t, R[t, l][R[t, l] >= 0]] = True
        # selected again within 32 steps (reverse cumulative OR over a window)
        soon = np.zeros_like(sel)
        for t in range(T - 1):
            soon[t] = sel[t + 1:t + 33].any(0)
        rho_u, rho_a, qs, at, lab = [], [], [], [], []
        for t in range(0, T - 33, 4):
            n = int(np.isfinite(S[t, l]).sum())
            u = ~sel[t, :n]
            s, a = S[t, l, :n], A[t, l, :n]
            rho_a.append(np.corrcoef(rank(s), rank(a))[0, 1])
            if u.sum() > 3:
                rho_u.append(np.corrcoef(rank(s[u]), rank(a[u]))[0, 1])
                qs.append(rank(s[u]) / u.sum())
                at.append(rank(a[u]) / u.sum())
                lab.append(soon[t, :n][u])
        q, a_, y = np.concatenate(qs), np.concatenate(at), np.concatenate(lab)
        out.append([rid, l, float(np.mean(rho_a)), float(np.mean(rho_u)), tie_auc(q, y), tie_auc(a_, y), float(y.mean())])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    paths = sorted(str(p) for p in a.traces.glob("*.npz"))
    with Pool(a.procs) as pool:
        rows = [r for part in pool.map(job, paths) for r in part]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc", "layer", "spearman_all_pages", "spearman_unselected", "auc_quest_unselected",
                    "auc_attn_unselected", "base_rate"])
        w.writerows(rows)
    r = np.array([x[2:] for x in rows], float)
    print("mean over doc x layer: rho all %.3f, rho unselected %.3f, AUC quest %.3f, AUC attention %.3f, base %.3f"
          % tuple(np.nanmean(r, 0)))


if __name__ == "__main__":
    main()
