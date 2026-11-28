"""C2 (linear) and B3 (boosted trees, upper-bound reference only): learned next-use regression with richer features.

  python -m analysis_v3.learn TRACES --out results_v3/learn/<model> --model linear|gbdt [--B 96 128] [--procs 14]

Both regress log(1 + min(next-use distance, 1024)) of every eviction candidate and evict the largest prediction.
Training: behaviour cloning on LRU's decisions, then DAgger (linear: 2 iterations; trees: 1) adding the states the
student visits; leave-one-document-out (the held-out document never enters training). Features (FEATS):
  age          log(1 + steps since last selected)          z            current z
  z_ema        EMA(z), alpha 0.5                           z_slow       EMA(z), alpha 0.9
  pw_norm      normalised power-law kernel of z (gamma 1)  log_cnt      log(1 + selections while resident)
  last_rank    rank at the last selection / K              log_resid    log(1 + steps since admission)
  deficit      sum over the residency of min(z, 0)         log_unsel    log(1 + steps resident and not selected)
  pct          all-page percentile of the current score    nb_z         max z of the two neighbouring pages
  trend        z - z_ema
Trees also get `layer / L` and `t / T` (they can model interactions; the linear model cannot use them).
"""
import argparse
import bisect
import csv
import json
import math
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import kvpolicy as kp
from kvpolicy_v2 import powerlaw_mixture

FEATS = ["age", "z", "z_ema", "z_slow", "pw_norm", "log_cnt", "last_rank", "log_resid", "deficit", "log_unsel", "pct",
         "nb_z", "trend"]
_R, _W, _ = powerlaw_mixture(1.0)
RATES = [float(r) for r, w in zip(_R, _W) if w > 0]
WTS = [float(w) for w in _W if w > 0]


class FeatPolicy(kp.Policy):
    needs_scores = True

    def __init__(self, K, model=None, collect=None, layer_frac=0.0, T=1, **kw):
        super().__init__(K, **kw)
        self.model, self.collect, self.layer_frac, self.T = model, collect, layer_frac, T
        self.ema, self.slow, self.acc, self.nrm, self.cnt, self.lrank, self.adm, self.D, self.uns = \
            {}, {}, {}, {}, {}, {}, {}, {}, {}

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        self.table = kp._allrank_table(srow)
        self.n_el = len(self.table)
        sel = set(ranked)
        for p in self.ema:
            x = kp._z(srow, ranked, p)
            self.ema[p] = 0.5 * self.ema[p] + 0.5 * x
            self.slow[p] = 0.9 * self.slow[p] + 0.1 * x
            a, nm = self.acc[p], self.nrm[p]
            for j, r in enumerate(RATES):
                a[j] = r * a[j] + x
                nm[j] = r * nm[j] + 1.0
            self.D[p] += min(x, 0.0)
            if p not in sel:
                self.uns[p] += 1

    def feat(self, p):
        s, rk = self.srow, self.ranked
        z = kp._z(s, rk, p)
        nb = [kp._z(s, rk, q) for q in (p - 1, p + 1) if 0 <= q < self.n_el]
        pw = sum(w * x for w, x in zip(WTS, self.acc[p])) / sum(w * x for w, x in zip(WTS, self.nrm[p]))
        return [math.log1p(self.t - self.last[p]), z, self.ema[p], self.slow[p], pw, math.log1p(self.cnt[p]),
                self.lrank[p] / self.K, math.log1p(self.t - self.adm[p]), self.D[p], math.log1p(self.uns[p]),
                kp._allrank(self.table, s[p]), max(nb) if nb else z, z - self.ema[p]]

    def evict(self, cands, n):
        cands = list(cands)
        X = np.array([self.feat(p) for p in cands])
        if self.collect is not None:
            self.collect.append((X, cands, self.t))
        if self.model is None:
            order = sorted(range(len(cands)), key=lambda i: (self.last[cands[i]], self.hit[cands[i]], -cands[i]))
        else:
            pred = self.model(X, self)
            order = np.argsort(-pred, kind="stable")
        return [cands[i] for i in order[:n]]

    def dropped(self, p):
        super().dropped(p)
        for d in (self.ema, self.slow, self.acc, self.nrm, self.cnt, self.lrank, self.adm, self.D, self.uns):
            d.pop(p, None)

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for r, p in enumerate(self.ranked, 1):
            self.lrank[p] = r
            self.cnt[p] = self.cnt.get(p, 0) + 1
        for p in misses:
            x = kp._z(self.srow, self.ranked, p)
            self.ema[p] = self.slow[p] = x
            self.acc[p] = [x] * len(RATES)
            self.nrm[p] = [1.0] * len(RATES)
            self.adm[p] = self.t
            self.D[p] = min(x, 0.0)
            self.uns[p] = 0


kp.POLICIES["featpol"] = FeatPolicy


def make_model(kind, params):
    if params is None:
        return None
    if kind == "linear":
        coef = np.asarray(params, float)
        return lambda X, pol: X @ coef[:-1] + coef[-1]
    tree = params

    def f(X, pol):
        extra = np.column_stack([np.full(len(X), pol.layer_frac), np.full(len(X), pol.t / pol.T)])
        return tree.predict(np.hstack([X, extra]))
    return f


def collect(path, B, kind, params, layers=None):
    R, S, rid = kp.load(path)
    T, L, _ = R.shape
    X, y, misses, ext = [], [], 0, []
    for l in (range(L) if layers is None else layers):
        uses = defaultdict(list)
        for t, row in enumerate(R[:, l]):
            for p in row:
                if p >= 0:
                    uses[int(p)].append(t)
        log = []
        c = kp.simulate(R[:, l], S[:, l], B, "featpol", model=make_model(kind, params), collect=log,
                        layer_frac=l / L, T=T)
        misses += c["misses"]
        for Xs, cands, t in log:
            d = []
            for p in cands:
                u = uses[p]
                i = bisect.bisect_right(u, t)
                d.append(u[i] - t if i < len(u) else T - t + 512)
            X.append(Xs)
            y.append(np.log1p(np.minimum(np.array(d, float), 1024)))
            ext.append(np.column_stack([np.full(len(cands), l / L), np.full(len(cands), t / T)]))
    return dict(rid=rid, X=np.vstack(X), y=np.concatenate(y), ext=np.vstack(ext), misses=misses)


def fit(kind, X, y, ext, seed):
    rng = np.random.default_rng(seed)
    n = min(len(y), 400000 if kind == "linear" else 300000)
    idx = rng.choice(len(y), n, replace=False)
    if kind == "linear":
        Xs, ys = X[idx], y[idx]
        mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
        Z = (Xs - mu) / sd
        w = np.linalg.solve(Z.T @ Z + np.eye(Z.shape[1]), Z.T @ (ys - ys.mean()))
        coef = w / sd
        return np.append(coef, ys.mean() - mu @ coef), dict(standardised=dict(zip(FEATS, np.round(w, 4).tolist())))
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=seed)
    m.fit(np.hstack([X[idx], ext[idx]]), y[idx])
    imp = {}
    try:
        from sklearn.inspection import permutation_importance
        sub = rng.choice(n, min(n, 20000), replace=False)
        pi = permutation_importance(m, np.hstack([X[idx], ext[idx]])[sub], y[idx][sub], n_repeats=3, random_state=seed)
        imp = dict(zip(FEATS + ["layer", "t_frac"], np.round(pi.importances_mean, 4).tolist()))
    except Exception as e:  # noqa: BLE001
        imp = {"error": str(e)}
    return m, dict(importance=imp)


def _collect(args):
    return collect(*args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", choices=["linear", "gbdt"], required=True)
    ap.add_argument("--B", type=int, nargs="+", default=[96, 128])
    ap.add_argument("--procs", type=int, default=14)
    a = ap.parse_args()
    paths = sorted(str(p) for p in a.traces.glob("*.npz"))
    a.out.mkdir(parents=True, exist_ok=True)
    iters = 2 if a.model == "linear" else 1
    rows, info = [], []
    with Pool(a.procs) as pool:
        for B in a.B:
            base = pool.map(_collect, [(p, B, a.model, None) for p in paths])     # LRU behaviour
            for i, held in enumerate(paths):
                tr = [j for j in range(len(paths)) if j != i]
                aggX = [base[j]["X"] for j in tr]
                aggy = [base[j]["y"] for j in tr]
                agge = [base[j]["ext"] for j in tr]
                params, meta = fit(a.model, np.vstack(aggX), np.concatenate(aggy), np.vstack(agge), seed=0)
                bc = collect(held, B, a.model, params)["misses"]
                for it in range(1, iters + 1):
                    visits = pool.map(_collect, [(paths[j], B, a.model, params) for j in tr])
                    aggX += [v["X"] for v in visits]
                    aggy += [v["y"] for v in visits]
                    agge += [v["ext"] for v in visits]
                    params, meta = fit(a.model, np.vstack(aggX), np.concatenate(aggy), np.vstack(agge), seed=it)
                res = collect(held, B, a.model, params)
                rows.append([a.model, B, res["rid"], bc, res["misses"]])
                info.append(dict(B=B, doc=res["rid"], **meta, **({"coef": params.tolist()} if a.model == "linear" else {})))
                print(a.model, B, res["rid"], bc, res["misses"], flush=True)
    with open(a.out / "learned.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "B", "doc", "misses_bc", "misses_dagger"])
        w.writerows(rows)
    (a.out / "models.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
