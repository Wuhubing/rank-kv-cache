"""B2 (followup-v3): trace statistics per (document, layer) and per-layer G, for explaining why G is stable.

  python -m analysis_v3.b2_stats SETTING TRACES --out results_v3/b2/<SETTING>.csv [--ratios 1.25 1.5 2 4] [--procs 16]

Per layer of every trace:
  overlap          mean |S_t & S_t+1| / K (adjacent-step top-K overlap)
  ac_lag{L}        autocorrelation of the all-page score percentile of pages that are NOT selected at t, between
                   t and t+L (Pearson over all such (t, page) pairs with the page still eligible at t+L), L = 1..64
  sel_ac_lag{L}    same for the selection indicator of unselected pages at t (P[selected at t+L | unselected at t])
  reuse_le{g}      share of re-selections (page selected again) whose gap since its previous selection is <= g
  ret_le{g}        LRU repeat misses at B = 1.5K: share whose gap since the page's previous selection is <= g
  lru/ema/bel_B{B} misses of LRU, score EMA (z, alpha 0.5, fixed) and Belady at B = ratio * K
"""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import kvpolicy as kp

LAGS = (1, 2, 4, 8, 16, 32, 64)
GAPS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def percentiles(S):
    """[T, P] score rows (NaN = not yet eligible) -> mid-rank percentile among the eligible pages of each row."""
    T, P = S.shape
    out = np.full((T, P), np.nan)
    for t in range(T):
        n = int(np.isfinite(S[t]).sum())
        if n < 2:
            continue
        s = S[t, :n].astype(np.float64)
        order = np.argsort(s, kind="stable")
        ranks = np.empty(n)
        ranks[order] = np.arange(n)
        # mid-ranks for ties
        srt = s[order]
        _, start, cnt = np.unique(srt, return_index=True, return_counts=True)
        mid = start + (cnt - 1) / 2
        grp = np.repeat(np.arange(len(start)), cnt)
        ranks[order] = mid[grp]
        out[t, :n] = ranks / (n - 1)
    return out


def layer_stats(R, S, K):
    T = R.shape[0]
    P = S.shape[1]
    sel = np.zeros((T, P), bool)
    for t in range(T):
        sel[t, R[t][R[t] >= 0]] = True
    out = {}
    out["overlap"] = float(np.mean((sel[:-1] & sel[1:]).sum(1) / K))
    pct = percentiles(S)
    for L in LAGS:
        if T <= L:
            out[f"ac_lag{L}"] = out[f"sel_ac_lag{L}"] = np.nan
            continue
        a, b = pct[:-L], pct[L:]
        m = (~sel[:-L]) & np.isfinite(a) & np.isfinite(b)
        x, y = a[m], b[m]
        out[f"ac_lag{L}"] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else np.nan
        out[f"sel_ac_lag{L}"] = float(sel[L:][m].mean()) if m.any() else np.nan
    gaps = []
    lastsel = {}
    for t in range(T):
        for p in R[t][R[t] >= 0]:
            p = int(p)
            if p in lastsel:
                gaps.append(t - lastsel[p])
            lastsel[p] = t
    gaps = np.array(gaps)
    for g in GAPS:
        out[f"reuse_le{g}"] = float((gaps <= g).mean()) if len(gaps) else np.nan
    out["reuse_mean_log_gap"] = float(np.log(gaps).mean()) if len(gaps) else np.nan
    return out, sel


class _RecLRU(kp.LRU):
    """LRU that records (t, page, gap since previous selection) of every repeat miss."""
    def __init__(self, K, **kw):
        super().__init__(K, **kw)
        self.prev, self.rep = {}, []

    def admit(self, hits, misses):
        for p in misses:
            if p in self.prev:
                self.rep.append(self.t - self.prev[p])
        super().admit(hits, misses)
        for p in self.ranked:
            self.prev[p] = self.t


kp.POLICIES["_rec_lru_b2"] = _RecLRU


def job(args):
    setting, path, ratios = args
    R, S, rid = kp.load(path)
    T, L, K = R.shape
    budgets = sorted({max(K, int(round(r * K))) for r in ratios})
    Bret = max(K, int(round(1.5 * K)))
    rows = []
    for l in range(L):
        st, _ = layer_stats(R[:, l], S[:, l], K)
        row = dict(setting=setting, doc=rid, layer=l, steps=T, K=K, **st)
        # repeat-miss return gaps under LRU at B = 1.5K (the policy object is captured at its first eviction)
        box = {}
        kp.simulate(R[:, l], None, Bret, "_rec_lru_b2", on_evict=lambda t, v, pol: box.setdefault("p", pol))
        rep = np.array(box["p"].rep if box else [])
        for g in GAPS:
            row[f"ret_le{g}"] = float((rep <= g).mean()) if len(rep) else np.nan
        row["repeat_misses_B1.5K"] = len(rep)
        for B in budgets:
            for name, pol, kw in (("lru", "lru", {}), ("ema", "ema", dict(alpha=0.5)), ("bel", "belady", {})):
                row[f"{name}_B{B}"] = kp.simulate(R[:, l], S[:, l], B, pol, **kw)["misses"]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("setting")
    ap.add_argument("traces", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ratios", type=float, nargs="+", default=[1.25, 1.5, 2, 4])
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--glob", default="*.npz")
    a = ap.parse_args()
    paths = sorted(str(p) for p in a.traces.glob(a.glob))
    with Pool(a.procs) as pool:
        rows = [r for part in pool.imap_unordered(job, [(a.setting, p, a.ratios) for p in paths]) for r in part]
    rows.sort(key=lambda r: (r["doc"], r["layer"]))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("setting", "doc", "layer"), k))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{a.setting}: {len(paths)} traces, {len(rows)} layer rows")


if __name__ == "__main__":
    main()
