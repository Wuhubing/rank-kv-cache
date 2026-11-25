"""Preregistered held-out evaluation (docs/efficient_policy_dev.md, approved 2026-11-23).

  python heldout.py freeze DEV_TRACES --out frozen.json   # dev only: allocation + linear coefficients
  python heldout.py eval HELDOUT_TRACES --frozen frozen.json --out DIR

Nothing here is tuned on held-out data: `eval` only reads the frozen file. Both commands check
the trace folder against the split file (--split): every trace's internal rid must belong to the
expected subset, the folder must hold exactly that subset's documents once each, file names must
match the rids, and every trace must have the split's top-K width and the expected layer count.
"""
import argparse
import csv
import hashlib
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import efficient_dev as ed
import kvpolicy as kp

GRID = ed.GRID
CURVE = [72, 80, 96, 128, 256]
PRIMARY = 96
ALPHA = {72: 0.4, 80: 0.4, 96: 0.5, 128: 0.7, 256: 0.95}  # written by `freeze` only; `eval` reads the frozen file
LINEAR_B = (96, 128, 256)
SPLIT = Path(__file__).resolve().parent / "splits/longctx_v1.json"


def trace_name(rid):
    return rid.replace("/", "__") + ".npz"


def check_traces(paths, split, subset, layers=None):
    """Check a trace folder against split membership; returns the sorted rids. `subset` is "dev"
    or "heldout"; `layers` is the required layer count (None: equal across traces)."""
    prompts = split["prompts"]
    expected = {p["rid"]: p for p in prompts if p["split"] == subset}
    elsewhere = {p["rid"]: p["split"] for p in prompts if p["split"] != subset}
    top_k = split["selector"]["top_k"]
    if not expected:
        raise ValueError(f"split {split.get('name')} has no {subset} documents")
    seen = {}
    for path in map(Path, paths):
        with np.load(path, allow_pickle=False) as z:
            if "rid" not in z.files or "ranked" not in z.files:
                raise ValueError(f"{path.name}: trace lacks rid or ranked")
            rid, shape = str(z["rid"]), z["ranked"].shape
            ctx = str(z["ctx"]) if "ctx" in z.files else None
        if rid in elsewhere:
            raise ValueError(f"{path.name}: document {rid} is a {elsewhere[rid]} document, "
                             f"not {subset} (split {split.get('name')})")
        if rid not in expected:
            raise ValueError(f"{path.name}: document {rid} is not in split {split.get('name')}")
        if rid in seen:
            raise ValueError(f"{path.name}: document {rid} duplicates {seen[rid].name}")
        if path.name != trace_name(rid):
            raise ValueError(f"{path.name}: file name does not match its document id {rid}")
        if ctx is not None and ctx != expected[rid]["ctx"]:
            raise ValueError(f"{path.name}: context {ctx} differs from the split's {expected[rid]['ctx']}")
        if len(shape) != 3 or shape[2] != top_k:
            raise ValueError(f"{path.name}: ranked shape {shape}, expected [T, L, {top_k}]")
        if layers is None:
            layers = shape[1]
        if shape[1] != layers:
            raise ValueError(f"{path.name}: {shape[1]} layers, expected {layers}")
        seen[rid] = path
    missing = sorted(set(expected) - set(seen))
    if missing:
        raise ValueError(f"{len(missing)} {subset} documents missing from the trace folder: {missing}")
    ctxs = [expected[r]["ctx"] for r in seen]
    other_ctx = {p["ctx"] for p in prompts if p["split"] != subset}
    if len(set(ctxs)) != len(ctxs) or set(ctxs) & other_ctx:
        raise ValueError(f"{subset} documents do not have unique contexts disjoint from other splits")
    return sorted(seen)


def load_split(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def frozen_alpha(frozen, B):
    """The frozen EMA decay for total budget B (a real number in [0, 1])."""
    a = frozen.get("alpha", {}).get(str(B)) if isinstance(frozen, dict) else None
    if isinstance(a, bool) or not isinstance(a, (int, float)) or not 0 <= a <= 1:
        raise ValueError(f"frozen file has no valid alpha for B={B}: {a!r}")
    return float(a)


def check_frozen(frozen, curve=None):
    """Validate every policy parameter of a frozen file for the budgets in `curve` and return the
    layer count of its allocation plans. Raises ValueError; nothing here uses `assert`."""
    curve = CURVE if curve is None else curve
    if not isinstance(frozen, dict):
        raise ValueError("frozen file must be a JSON object")
    missing = {"dev_docs", "alpha", "alloc", "linear"} - set(frozen)
    if missing:
        raise ValueError(f"frozen file lacks {sorted(missing)}")
    docs = frozen["dev_docs"]
    if not isinstance(docs, list) or not docs or not all(isinstance(d, str) for d in docs) \
            or len(set(docs)) != len(docs):
        raise ValueError("frozen dev_docs must be a non-empty list of unique names")
    L = None
    for B in curve:
        frozen_alpha(frozen, B)
        plan = frozen["alloc"].get(str(B))
        if not isinstance(plan, list) or not plan or not all(
                isinstance(g, int) and not isinstance(g, bool) and g in GRID for g in plan):
            raise ValueError(f"frozen alloc for B={B} must be a list of grid budgets {GRID}")
        if L is None:
            L = len(plan)
        if len(plan) != L:
            raise ValueError(f"frozen alloc for B={B} has {len(plan)} layers, expected {L}")
        if sum(plan) > L * B:
            raise ValueError(f"frozen alloc for B={B} uses {sum(plan)} slots > {L} x {B}")
    for key, coef in frozen["linear"].items():
        if not isinstance(coef, list) or len(coef) != len(kp.FEATURES) + 1 or not all(
                isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in coef):
            raise ValueError(f"frozen linear coefficients for B={key} must be "
                             f"{len(kp.FEATURES) + 1} finite numbers")
    return L


def load_frozen(path, curve=None):
    """Read and validate a frozen file. Returns (frozen, sha256 of its bytes, layer count)."""
    raw = Path(path).read_bytes()
    try:
        frozen = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON ({e})") from None
    return frozen, hashlib.sha256(raw).hexdigest(), check_frozen(frozen, curve)


def configs(B, frozen):
    """Policies evaluated at uniform budget B. The EMA decay and the linear coefficients come from
    the frozen file; the remaining baselines have fixed, preregistered parameters."""
    out = {"lru": ("lru", {}), "belady": ("belady", {}), "ema": ("ema", dict(alpha=frozen_alpha(frozen, B))),
           "score": ("score", {}), "dsum": ("dsum", dict(decay=0.5, v="softmax")),
           "dcount": ("dsum", dict(decay=0.8, v="indicator"))}
    if str(B) in frozen.get("linear", {}):
        out["linear"] = ("linear", dict(coef=np.array(frozen["linear"][str(B)])))
    return out


def jsonable(kw):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in kw.items()}


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
    split, split_sha = load_split(getattr(args, "split", SPLIT))
    check_traces(paths, split, "dev")
    pool = Pool(args.procs)
    out = {"dev_docs": [Path(p).name for p in paths], "alpha": ALPHA, "alloc": {}, "linear": {},
           "dev_check": {}, "split_sha256": split_sha}
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
    frozen, frozen_sha, L = load_frozen(args.frozen)
    split, split_sha = load_split(getattr(args, "split", SPLIT))
    dev = {trace_name(p["rid"]) for p in split["prompts"] if p["split"] == "dev"}
    if set(frozen["dev_docs"]) != dev:
        raise ValueError("frozen dev_docs do not match the split's dev documents")
    paths = [str(p) for p in sorted(args.traces.glob("*.npz"))]
    if set(Path(p).name for p in paths) & set(frozen["dev_docs"]):
        raise ValueError("a frozen dev document file is in the held-out folder")
    docs = check_traces(paths, split, "heldout", layers=L)
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
                jobs.append((p, "ema", dict(alpha=frozen_alpha(frozen, B)), [g])); keys.append(f"alloc_ema@{B}")
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
    prov = dict(frozen=str(args.frozen), frozen_sha256=frozen_sha, split=str(getattr(args, "split", SPLIT)),
                split_sha256=split_sha, docs=docs, curve=CURVE,
                configs={str(B): {c: [n, jsonable(kw)] for c, (n, kw) in configs(B, frozen).items()}
                         for B in CURVE},
                alloc_ema={str(B): dict(alpha=frozen_alpha(frozen, B), alloc=frozen["alloc"][str(B)])
                           for B in CURVE})
    (args.out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print("rows", len(keep), "frozen sha256", frozen_sha)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["freeze", "eval"])
    ap.add_argument("traces", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frozen", type=Path)
    ap.add_argument("--split", type=Path, default=SPLIT)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "eval" and args.frozen is None:
        ap.error("eval needs --frozen")
    freeze(args) if args.cmd == "freeze" else evaluate(args)


if __name__ == "__main__":
    main()
