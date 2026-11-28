"""B2 / D5 / D6 (followup-v3): does the temporal structure of the score ordering explain G?

  python -m analysis_v3.b2_model regress --stats results_v3/b2 --out results_v3/b2/regression
  python -m analysis_v3.b2_model fit     --stats results_v3/b2/dev.csv --out results_v3/b2/synthetic [--procs 16]
  python -m analysis_v3.b2_model curves  --out results_v3/b2/synthetic [--procs 16]

regress  units = (setting, document, layer). Target: G of the score EMA (z, alpha 0.5) against LRU and Belady at
         B = 1.5K (and 2K). Features: lag-1/4/16/64 autocorrelation of unselected pages' score percentile, adjacent-step
         top-K overlap, P[selected again within 1/4/16 steps | unselected now] (sel_ac), reuse-gap shares, LRU
         repeat-miss return-gap shares. Linear least squares, leave-one-DOCUMENT-out within each setting (the model is
         fitted on the other documents of all settings); reported: Spearman between predicted and measured G over all
         held-out units, over layers (mean over documents) and over settings (mean over layers and documents).
synthetic model  each page has a latent score x_p(t) = m_p + s_p(t) + f_p(t) + r * exp(-age_p(t) / tau): a static part,
         a slow AR(1) (phi_s), a fast AR(1) (phi_f) and a recency boost for newly completed pages; one new page every 16
         steps; the selection is the top-K of x; the policies read x. `fit` grid-searches (phi_f, phi_s, variance split,
         r) to match the measured mean overlap and the lag-1..64 autocorrelation curve of 0.6B dev; `curves` replays
         LRU, Belady, current score and EMA(alpha) at B/K = 1.25..4 on synthetic traces of the fitted parameters, plus the
         protection zone (D5) on regimes with lower overlap / stronger recency, and records per-layer-like stats for the
         alpha rule (D6).
"""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import kvpolicy as kp
import kvpolicy_v3  # noqa: F401
from analysis_v3.b2_stats import LAGS, layer_stats

FEATS = ["ac_lag1", "ac_lag4", "ac_lag16", "ac_lag64", "overlap", "sel_ac_lag1", "sel_ac_lag4", "sel_ac_lag16",
         "reuse_le1", "reuse_le4", "reuse_le16", "ret_le4", "ret_le16", "ret_le64"]


def read_stats(folder):
    rows = []
    for f in sorted(Path(folder).glob("*.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def g_unit(r, ratio):
    K = int(r["K"])
    B = max(K, int(round(ratio * K)))
    lru, ema, bel = (float(r[f"{n}_B{B}"]) for n in ("lru", "ema", "bel"))
    return (lru - ema) / (lru - bel) if lru > bel else float("nan"), lru - bel


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1]) if ok.sum() > 2 else float("nan")


def cmd_regress(a):
    rows = read_stats(a.stats)
    out = []
    for ratio in (1.5, 2.0):
        units = []
        for r in rows:
            g, gap = g_unit(r, ratio)
            if not np.isfinite(g) or gap < 20:          # too few avoidable misses for a per-layer G
                continue
            x = [float(r[f]) for f in FEATS]
            if not all(np.isfinite(x)):
                continue
            units.append((r["setting"], r["doc"], int(r["layer"]), g, x))
        X = np.array([u[4] for u in units])
        y = np.array([u[3] for u in units])
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])
        pred = np.full(len(y), np.nan)
        keys = sorted({(u[0], u[1]) for u in units})
        for key in keys:
            te = np.array([(u[0], u[1]) == key for u in units])
            w = np.linalg.lstsq(Z[~te], y[~te], rcond=None)[0]
            pred[te] = Z[te] @ w
        res = dict(ratio=ratio, units=len(y), spearman_units=spearman(pred, y))
        by_layer, by_set = defaultdict(list), defaultdict(list)
        for u, p in zip(units, pred):
            by_layer[u[0], u[2]].append((u[3], p))
            by_set[u[0]].append((u[3], p))
        lay = [(np.mean([v[0] for v in vs]), np.mean([v[1] for v in vs])) for vs in by_layer.values()]
        st = [(np.mean([v[0] for v in vs]), np.mean([v[1] for v in vs])) for vs in by_set.values()]
        res["spearman_setting_layer"] = spearman([x[1] for x in lay], [x[0] for x in lay])
        res["n_setting_layer"] = len(lay)
        res["spearman_settings"] = spearman([x[1] for x in st], [x[0] for x in st])
        res["n_settings"] = len(st)
        for s_ in sorted({u[0] for u in units}):
            sel = [(u[3], p) for u, p in zip(units, pred) if u[0] == s_]
            lay_s = defaultdict(list)
            for u, p in zip(units, pred):
                if u[0] == s_:
                    lay_s[u[2]].append((u[3], p))
            ls = [(np.mean([v[0] for v in vs]), np.mean([v[1] for v in vs])) for vs in lay_s.values()]
            res[f"spearman_layers_{s_}"] = spearman([x[1] for x in ls], [x[0] for x in ls])
        # single-feature Spearman with G (descriptive)
        for j, f in enumerate(FEATS):
            res[f"rho_{f}"] = spearman(X[:, j], y)
        w = np.linalg.lstsq(Z, y, rcond=None)[0]
        res["coef_full_fit"] = dict(zip(FEATS + ["bias"], np.round(w, 4).tolist()))
        out.append(res)
        a.out.mkdir(parents=True, exist_ok=True)
        with open(a.out / f"units_ratio{ratio}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["setting", "doc", "layer", "G", "pred"] + FEATS)
            for u, p in zip(units, pred):
                wr.writerow([u[0], u[1], u[2], u[3], p] + u[4])
    (a.out / "regression.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print({k: v for k, v in r.items() if not k.startswith("rho_") and k != "coef_full_fit"})


# ------------------------------------------------------------------------------------------- synthetic model

def synth(seed, T=600, P0=900, K=64, ps=16, phi_f=0.9, phi_s=0.995, w_static=0.4, w_slow=0.3, w_fast=0.3, r=1.0,
          tau=64.0):
    """ranked [T, K], scores [T, P] of the latent-score model (variance shares w_*; recency boost r, decay tau)."""
    rng = np.random.default_rng(seed)
    P = P0 + T // ps + 1
    m = rng.standard_normal(P) * np.sqrt(w_static)
    s = rng.standard_normal(P) * np.sqrt(w_slow)
    f = rng.standard_normal(P) * np.sqrt(w_fast)
    born = np.concatenate([np.full(P0, -10 ** 6), np.arange(P - P0) * ps])
    S = np.full((T, P), np.nan, np.float32)
    R = np.zeros((T, K), np.int64)
    for t in range(T):
        s = phi_s * s + np.sqrt(1 - phi_s ** 2) * np.sqrt(w_slow) * rng.standard_normal(P)
        f = phi_f * f + np.sqrt(1 - phi_f ** 2) * np.sqrt(w_fast) * rng.standard_normal(P)
        n = P0 + t // ps
        age = t - born[:n]
        x = m[:n] + s[:n] + f[:n] + r * np.exp(-np.maximum(age, 0) / tau)
        S[t, :n] = x
        R[t] = np.argsort(-x, kind="stable")[:K]
    return R, S


def stats_of(params, seeds=(0, 1)):
    acc = defaultdict(list)
    for sd in seeds:
        R, S = synth(sd, **params)
        st, _ = layer_stats(R, S, R.shape[1])
        for k, v in st.items():
            acc[k].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


GRID = dict(phi_f=(0.5, 0.8, 0.9, 0.95), phi_s=(0.99, 0.998), w=((0.5, 0.3, 0.2), (0.3, 0.4, 0.3), (0.2, 0.3, 0.5),
                                                                  (0.1, 0.2, 0.7)), r=(0.0, 0.5, 1.5))


def fit_job(args):
    pf, ps_, w, r = args
    p = dict(phi_f=pf, phi_s=ps_, w_static=w[0], w_slow=w[1], w_fast=w[2], r=r)
    return p, stats_of(p)


def cmd_fit(a):
    rows = [r for r in read_stats_file(a.stats)]
    target = {k: float(np.nanmean([float(r[k]) for r in rows])) for k in
              ["overlap"] + [f"ac_lag{L}" for L in LAGS] + [f"sel_ac_lag{L}" for L in LAGS]}
    combos = list(itertools.product(GRID["phi_f"], GRID["phi_s"], GRID["w"], GRID["r"]))
    with Pool(a.procs) as pool:
        res = pool.map(fit_job, combos)
    keys = ["overlap"] + [f"ac_lag{L}" for L in LAGS] + [f"sel_ac_lag{L}" for L in (1, 4, 16)]

    def loss(st):
        return sum((st[k] - target[k]) ** 2 * (4 if k == "overlap" else 1) for k in keys)
    res.sort(key=lambda x: loss(x[1]))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "fit.json").write_text(json.dumps(dict(target=target, best=res[0][0], best_stats=res[0][1],
                                                    top5=[dict(params=p, loss=loss(s)) for p, s in res[:5]]), indent=1))
    print("target", {k: round(v, 3) for k, v in target.items() if k in keys})
    print("best", res[0][0], {k: round(res[0][1][k], 3) for k in keys}, "loss", round(loss(res[0][1]), 4))


def read_stats_file(path):
    with open(path) as f:
        return list(csv.DictReader(f))


ALPHAS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
RATIOS = (1.25, 1.5, 2, 4)


def curve_job(args):
    tag, params, seed = args
    R, S = synth(seed, **params)
    K = R.shape[1]
    out = []
    st, _ = layer_stats(R, S, K)
    for ratio in RATIOS:
        B = int(round(ratio * K))
        base = {n: kp.simulate(R, S, B, n)["misses"] for n in ("lru", "belady", "score")}
        for n, m in base.items():
            out.append((tag, seed, ratio, n, "", m))
        for al in ALPHAS:
            out.append((tag, seed, ratio, "ema", al, kp.simulate(R, S, B, "ema", alpha=al)["misses"]))
        for base_pol in ("lru", "ema"):
            for rho in (0.1, 0.3, 0.5):
                m = kp.simulate(R, S, B, "protect", base=base_pol, alpha=0.5, rho=rho)["misses"]
                out.append((tag, seed, ratio, f"prot_{base_pol}", rho, m))
    return out, (tag, seed, st)


def cmd_curves(a):
    fit = json.loads((a.out / "fit.json").read_text())
    best = fit["best"]
    regimes = {"fitted": best}
    for pf in (0.5, 0.8, 0.95, 0.99):          # alpha rule (D6): vary the fast autocorrelation
        regimes[f"phi_f{pf}"] = dict(best, phi_f=pf)
    for r in (0.0, 3.0):                       # D5: recency boost (new pages revisited soon) and low persistence
        for pf in (0.3, best["phi_f"]):
            regimes[f"r{r}_phi_f{pf}"] = dict(best, r=r, phi_f=pf)
    jobs = [(tag, p, sd) for tag, p in regimes.items() for sd in range(4)]
    with Pool(a.procs) as pool:
        res = pool.map(curve_job, jobs)
    with open(a.out / "curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "seed", "B_over_K", "policy", "param", "misses"])
        for rows, _ in res:
            w.writerows(rows)
    stats = defaultdict(dict)
    for _, (tag, sd, st) in res:
        stats[tag][sd] = st
    (a.out / "regimes.json").write_text(json.dumps(dict(regimes=regimes, stats=stats), indent=1))
    tot = defaultdict(int)
    for rows, _ in res:
        for tag, sd, ratio, pol, prm, m in rows:
            tot[tag, ratio, pol, prm] += m
    summ = []
    for (tag, ratio, pol, prm), m in sorted(tot.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], str(x[0][3]))):
        lru, bel = tot[tag, ratio, "lru", ""], tot[tag, ratio, "belady", ""]
        summ.append(dict(regime=tag, B_over_K=ratio, policy=pol, param=prm, misses=m,
                         G=(lru - m) / (lru - bel) if lru > bel else float("nan")))
    with open(a.out / "curves_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    for s_ in summ:
        if s_["policy"] in ("ema", "score", "prot_lru", "prot_ema") and s_["B_over_K"] in (1.5, 4):
            print(f"{s_['regime']:<16} B/K={s_['B_over_K']} {s_['policy']:<9} {s_['param']!s:<5} G {s_['G']:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["regress", "fit", "curves"])
    ap.add_argument("--stats", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    dict(regress=cmd_regress, fit=cmd_fit, curves=cmd_curves)[a.cmd](a)


if __name__ == "__main__":
    main()
