"""Preregistered held-out evaluation (docs/efficient_policy_dev.md, approved 2026-11-23).

  python heldout.py freeze DEV_TRACES --out frozen.json   # dev only: allocation + linear coefficients
  python heldout.py eval HELDOUT_TRACES --frozen frozen.json --out DIR

Nothing here is tuned on held-out data: `eval` only reads the frozen file.
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import efficient_dev as ed
import kvpolicy as kp

GRID = ed.GRID
CURVE = [72, 80, 96, 128, 256]
PRIMARY = 96
ALPHA = {72: 0.4, 80: 0.4, 96: 0.5, 128: 0.7, 256: 0.95}
LINEAR_B = (96, 128, 256)


def configs(B, frozen=None):
    out = {"lru": ("lru", {}), "belady": ("belady", {}), "ema": ("ema", dict(alpha=ALPHA[B])),
           "score": ("score", {}), "dsum": ("dsum", dict(decay=0.5, v="softmax")),
           "dcount": ("dsum", dict(decay=0.8, v="indicator"))}
    if frozen and str(B) in frozen["linear"]:
        out["linear"] = ("linear", dict(coef=np.array(frozen["linear"][str(B)])))
    return out


def per_layer(args):
    """Misses per layer for one doc, one policy, a list of budgets."""
    path, name, kw, budgets = args
    R, S, rid = kp.load(path)
    rows = []
    for B in budgets:
        for l in range(R.shape[1]):
            c = kp.simulate(R[:, l], S[:, l], B, name, **kw)
            rows.append((rid, l, B, c["accesses"], c["misses"], c["compulsory"]))
    return rows


def freeze(args):
    paths = [str(p) for p in sorted(args.traces.glob("*.npz"))]
    pool = Pool(args.procs)
    out = {"dev_docs": [Path(p).name for p in paths], "alpha": ALPHA, "alloc": {}, "linear": {},
           "dev_check": {}}
    for B in CURVE:
        rows = [r for part in pool.map(per_layer, [(p, "ema", dict(alpha=ALPHA[B]), GRID) for p in paths])
                for r in part]
        L = 1 + max(r[1] for r in rows)
        curves = {l: {g: 0 for g in GRID} for l in range(L)}
        for _, l, g, _, m, _ in rows:
            curves[l][g] += m
        a = kp.allocate(curves, L, L * B, GRID)
        out["alloc"][str(B)] = [a[l] for l in range(L)]
        print("alloc", B, out["alloc"][str(B)], flush=True)
    for B in LINEAR_B:  # DAgger, two iterations, as in efficient_dev.py, on all dev docs
        data = pool.starmap(ed.collect, [(p, B) for p in paths])
        X, y = [d["X"] for d in data], [d["y"] for d in data]
        coef = ed.ridge(np.vstack(X), np.concatenate(y))
        for it in (1, 2):
            visits = pool.starmap(ed.collect, [(p, B, coef) for p in paths])
            X += [v["X"] for v in visits]
            y += [v["y"] for v in visits]
            coef = ed.ridge(np.vstack(X), np.concatenate(y), seed=it)
        out["linear"][str(B)] = [float(c) for c in coef]
        print("linear", B, out["linear"][str(B)], flush=True)
    for name in ("lru", "ema", "belady"):  # reproducibility check against the dev report
        kw = dict(alpha=0.5) if name == "ema" else {}
        rows = [r for part in pool.map(per_layer, [(p, name, kw, [PRIMARY]) for p in paths]) for r in part]
        out["dev_check"][name] = [sum(r[3] for r in rows), sum(r[4] for r in rows)]
    print("dev_check", out["dev_check"])
    json.dump(out, open(args.out, "w"), indent=1)


def evaluate(args):
    frozen = json.load(open(args.frozen))
    paths = [str(p) for p in sorted(args.traces.glob("*.npz"))]
    assert not set(Path(p).name for p in paths) & set(frozen["dev_docs"]), "dev doc in held-out set"
    args.out.mkdir(parents=True, exist_ok=True)
    pool = Pool(args.procs)
    jobs, keys = [], []
    for B in CURVE:
        for cfg, (name, kw) in configs(B, frozen).items():
            for p in paths:
                jobs.append((p, name, kw, [B])); keys.append(cfg)
        # allocation + EMA: each layer at its frozen budget
        for g in sorted(set(frozen["alloc"][str(B)])):
            for p in paths:
                jobs.append((p, "ema", dict(alpha=ALPHA[B]), [g])); keys.append(f"alloc_ema@{B}")
    rows = []
    for cfg, part in zip(keys, pool.map(per_layer, jobs)):
        rows += [(cfg,) + r for r in part]
    # keep only the layer/budget pairs of the frozen plan for alloc rows; relabel to the total budget
    keep = []
    for cfg, rid, l, B, acc, mis, comp in rows:
        if cfg.startswith("alloc_ema@"):
            tot = int(cfg.split("@")[1])
            if frozen["alloc"][str(tot)][l] != B:
                continue
            keep.append(("alloc_ema", rid, l, tot, acc, mis, comp))
        else:
            keep.append((cfg, rid, l, B, acc, mis, comp))
    with open(args.out / "misses.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "doc", "layer", "B", "accesses", "misses", "compulsory"]); w.writerows(keep)
    print("rows", len(keep))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["freeze", "eval"])
    ap.add_argument("traces", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frozen", type=Path)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    freeze(args) if args.cmd == "freeze" else evaluate(args)


if __name__ == "__main__":
    main()
