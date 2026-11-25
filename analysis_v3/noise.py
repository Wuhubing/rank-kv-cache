"""D3 / C3 (followup-v3): replacement policies under more realistic score noise (0.6B dev, B = 96 and 128).

  python -m analysis_v3.noise pool --attn TRACES_WITH_ATTN_ALL --out results_v3/d3      (empirical error pool)
  python -m analysis_v3.noise replay TRACES --out results_v3/d3 [--procs 16]
  python -m analysis_v3.noise summarize --out results_v3/d3

Noise models (sd_t = standard deviation of the eligible pages' scores at step t; one draw per (document, layer, seed)):
  iid:s          s * sd_t * N(0,1), independent per step and page (replication-v2's model, for reference)
  ar{rho}:s      AR(1) over steps per page: e_t = rho e_{t-1} + sqrt(1 - rho^2) xi_t, noise s * sd_t * e_t
  mono{k}        order-preserving distortion of each step's scores (x -> exp(k * standardised x), rescaled to the step's
                 spread) plus iid noise 0.05 * sd_t: the policies see a different scale and, for k=0.5, almost the same
                 order; k=2 compresses the low scores so much that the small perturbation reorders them
  emp            measured error: per layer, the error of the standardised Quest score against the standardised log true
                 attention (data/dev_attn_all; 48 pages per document and layer that are eligible from the first step,
                 whole sequences, emp_pool.npz), resampled by page: each page gets the error sequence of a random page
                 of ANOTHER document at the same layer, so its time correlation is the measured one; added in
                 standardised units
  cf_<model>     counterfactual: the selection also comes from the noisy scores (ranked = top-K of the noisy row); LRU and
                 Belady are recomputed on that stream. The generation is NOT re-run (fixed trace), so the stream is not
                 what a real noisy selector would produce. Labelled as such.
Without "cf_", the selection (and LRU, Belady) is the trace's.
Also reported per model: Spearman rho between the noisy and the clean scores (mean over steps), to place each noise
level against Quest's measured rho with the true attention (0.73).
"""
import argparse
import csv
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import kvpolicy as kp
import kvpolicy_v3  # noqa: F401

BUDGETS = (96, 128)
SEEDS = (0, 1, 2)
CONFIGS = {
    "score": ("score", {}),
    "ema_z:0.5": ("ema", dict(alpha=0.5)),
    "ema_z:0.7": ("ema", dict(alpha=0.7)),
    "ema_allrank:0.5": ("ema", dict(alpha=0.5, signal="allrank")),
    "pow_norm_kth:1.0": ("kernel", dict(kernel="power", gamma=1.0, normalize=True)),
    "pow_sum_kth:1.0": ("kernel", dict(kernel="power", gamma=1.0)),
    "shrink_mean:0.5": ("shrink_ema", dict(alpha=0.5, c=0.5, prior="mean")),
    "shrink_zero:0.5": ("shrink_ema", dict(alpha=0.5, c=0.5, prior="zero")),
    "shrink_mean_allrank:0.5": ("shrink_ema", dict(alpha=0.5, c=0.5, prior="mean", signal="allrank")),
}
MODELS = (["none"] + [f"iid:{s}" for s in (0.1, 0.3, 1.0)] + [f"ar{r}:{s}" for r in (0.5, 0.9) for s in (0.1, 0.3, 1.0)]
          + ["mono0.5", "mono2", "emp"] + ["cf_iid:0.3", "cf_ar0.9:0.3", "cf_emp"])


def step_sd(S):
    with np.errstate(invalid="ignore"):
        return np.nanstd(S, axis=1, keepdims=True)


def standardise(X):
    with np.errstate(invalid="ignore"):
        return (X - np.nanmean(X, axis=1, keepdims=True)) / (np.nanstd(X, axis=1, keepdims=True) + 1e-12)


def noisy(S, model, rng, emp_pool=None):
    S = S.astype(np.float64)
    sd = step_sd(S)
    T, P = S.shape
    if model == "none":
        return S
    if model.startswith("iid:"):
        return S + float(model[4:]) * sd * rng.standard_normal(S.shape)
    if model.startswith("ar"):
        rho, s = model[2:].split(":")
        rho, s = float(rho), float(s)
        e = np.empty((T, P))
        e[0] = rng.standard_normal(P)
        for t in range(1, T):
            e[t] = rho * e[t - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(P)
        return S + s * sd * e
    if model.startswith("mono"):
        k = float(model[4:])
        Z = standardise(S)
        F = np.exp(k * np.clip(Z, -6, 6))
        F = standardise(F) * sd + np.nanmean(S, axis=1, keepdims=True)
        return F + 0.05 * sd * rng.standard_normal(S.shape)
    if model == "emp":
        E = np.zeros((T, P))
        for p in range(P):
            src = emp_pool[rng.integers(len(emp_pool))]
            seq = src[rng.integers(src.shape[0])]
            E[:, p] = np.resize(seq, T)          # wraps around if the source document is shorter
        return S + sd * E
    raise ValueError(model)


def topk(Sn, S, K):
    """ranked [T, K] from the noisy scores over the pages that are eligible in the clean trace."""
    R = np.full((S.shape[0], K), -1, np.int64)
    for t in range(S.shape[0]):
        n = int(np.isfinite(S[t]).sum())
        R[t, :min(K, n)] = np.argsort(-Sn[t, :n], kind="stable")[:K]
    return R


NPOOL = 48


def cmd_pool(a):
    """emp_pool.npz: per document, errors [L, NPOOL, T] (float16) of standardised Quest vs standardised log attention
    for NPOOL random pages eligible from the first step."""
    out = {}
    for p in sorted(a.attn.glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            rid = str(z["rid"])
            S, A = z["scores"].astype(np.float64), z["attn_all"].astype(np.float64)
        T, L, P = S.shape
        n0 = int(np.isfinite(S[0, 0]).sum())
        rng = np.random.default_rng(zlib.crc32(rid.encode()))
        pages = rng.choice(n0, NPOOL, replace=False)
        E = np.zeros((L, NPOOL, T), np.float16)
        for l in range(L):
            e = standardise(S[:, l]) - standardise(np.log(np.maximum(A[:, l], 1e-12)))
            E[l] = e[:, pages].T
        out[rid] = E
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "emp_pool.npz", **{k.replace("/", "__"): v for k, v in out.items()})
    print(f"pool: {len(out)} documents")


def emp_errors(pool_path, layer, exclude_rid):
    """Error sequences [NPOOL, T] of every other document at this layer."""
    with np.load(pool_path) as z:
        return [z[k][layer].astype(np.float64) for k in z.files if k != exclude_rid.replace("/", "__")]


def job(args):
    path, pool_path, model, seed = args
    R, S, rid = kp.load(path)
    T, L, K = R.shape
    rows, rhos = [], []
    for l in range(L):
        rng = np.random.default_rng([seed, l, zlib.crc32(rid.encode()), MODELS.index(model)])
        base = model[3:] if model.startswith("cf_") else model
        pool = emp_errors(pool_path, l, rid) if base == "emp" else None
        Sn = noisy(S[:, l], base, rng, pool)
        Sn = np.where(np.isfinite(S[:, l]), Sn, np.nan)
        Rl = topk(Sn, S[:, l], K) if model.startswith("cf_") else R[:, l]
        # Spearman between noisy and clean scores at every 8th step
        rs = []
        for t in range(0, T, 8):
            n = int(np.isfinite(S[t, l]).sum())
            if n > 2:
                a = np.argsort(np.argsort(S[t, l, :n]))
                b = np.argsort(np.argsort(Sn[t, :n]))
                rs.append(np.corrcoef(a, b)[0, 1])
        rhos.append((rid, l, model, seed, float(np.mean(rs))))
        for B in BUDGETS:
            for name, (pol, kw) in list(CONFIGS.items()) + [("lru", ("lru", {})), ("belady", ("belady", {}))]:
                if name in ("lru", "belady") and not model.startswith("cf_") and seed != SEEDS[0]:
                    continue
                c = kp.simulate(Rl, Sn, B, pol, **kw)
                rows.append((rid, l, model, seed, name, B, c["accesses"], c["misses"]))
    return rows, rhos


def cmd_replay(a):
    paths = sorted(str(p) for p in a.traces.glob("*.npz"))
    pool_path = str(a.out / "emp_pool.npz")
    jobs = [(p, pool_path, m, s) for p in paths for m in MODELS for s in (SEEDS if m != "none" else SEEDS[:1])]
    a.out.mkdir(parents=True, exist_ok=True)
    with Pool(a.procs) as pool, open(a.out / "misses.csv", "w", newline="") as f, \
            open(a.out / "rho.csv", "w", newline="") as g:
        w, wr = csv.writer(f), csv.writer(g)
        w.writerow(["doc", "layer", "model", "seed", "config", "B", "accesses", "misses"])
        wr.writerow(["doc", "layer", "model", "seed", "spearman_noisy_clean"])
        for n, (rows, rhos) in enumerate(pool.imap_unordered(job, jobs), 1):
            w.writerows(rows)
            wr.writerows(rhos)
            if n % 50 == 0:
                print(f"{n}/{len(jobs)}", flush=True)


def g_of(l, p, o):
    return (l - p) / (l - o) if l != o else float("nan")


def cmd_summarize(a):
    tot = defaultdict(lambda: defaultdict(int))     # (model, seed, config, B) -> doc -> misses
    with open(a.out / "misses.csv") as f:
        for r in csv.DictReader(f):
            tot[r["model"], int(r["seed"]), r["config"], int(r["B"])][r["doc"]] += int(r["misses"])
    rho = defaultdict(list)
    with open(a.out / "rho.csv") as f:
        for r in csv.DictReader(f):
            rho[r["model"]].append(float(r["spearman_noisy_clean"]))
    docs = sorted(tot["none", 0, "lru", 96])
    out = []
    for (model, seed, cfg, B), dm in sorted(tot.items()):
        if cfg in ("lru", "belady"):
            continue
        ref = (model, seed) if model.startswith("cf_") else ("none", 0)
        lru = sum(tot[ref[0], ref[1], "lru", B][d] for d in docs)
        opt = sum(tot[ref[0], ref[1], "belady", B][d] for d in docs)
        m = sum(dm[d] for d in docs)
        out.append(dict(model=model, seed=seed, config=cfg, B=B, misses=m, G=g_of(lru, m, opt)))
    agg = defaultdict(list)
    for r in out:
        agg[r["model"], r["config"], r["B"]].append(r["G"])
    rows = [dict(model=m, config=c, B=B, G_mean=float(np.mean(v)), G_min=float(np.min(v)), G_max=float(np.max(v)),
                 seeds=len(v), spearman_noisy_clean=float(np.mean(rho[m])) if rho[m] else float("nan"))
            for (m, c, B), v in sorted(agg.items())]
    with open(a.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["B"] == 96:
            print(f"{r['model']:<14} rho {r['spearman_noisy_clean']:.3f} {r['config']:<24} G {r['G_mean']:+.3f} "
                  f"[{r['G_min']:+.3f},{r['G_max']:+.3f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pool", "replay", "summarize"])
    ap.add_argument("traces", type=Path, nargs="?")
    ap.add_argument("--attn", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    dict(pool=cmd_pool, replay=cmd_replay, summarize=cmd_summarize)[a.cmd](a)


if __name__ == "__main__":
    main()
