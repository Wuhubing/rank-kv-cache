"""Preregistered held-out evaluation (docs/efficient_policy_dev.md, approved 2026-11-23).

  python heldout.py freeze DEV_TRACES --out frozen.json   # dev only: allocation + linear coefficients
  python heldout.py eval HELDOUT_TRACES --frozen frozen.json --out DIR

Nothing here is tuned on held-out data: `eval` only reads the frozen file.
"""
import argparse
import csv
import json
import hashlib
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
    out = {"lru": ("lru", {}), "belady": ("belady", {}), "ema": ("ema", dict(alpha=frozen["alpha"][str(B)] if frozen is not None else ALPHA[B])),
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


def replay_document(job):
    path, doc, frozen = job
    with np.load(path, allow_pickle=False) as trace:
        rid = str(trace["rid"])
        if rid != doc["rid"] or int(trace["prompt_tokens"]) != doc["prompt_tokens"]:
            raise ValueError(f"Trace identity or prompt length mismatch in {path}")
        ranked, scores = trace["ranked"], trace["scores"]
    if (ranked.ndim != 3 or ranked.shape[1:] != (28, 64) or not len(ranked)
            or scores.ndim != 3 or scores.shape[:2] != ranked.shape[:2]
            or scores.dtype.kind != "f" or ranked.dtype.kind not in "iu"
            or ranked.min() < 0 or ranked.max() >= scores.shape[2]
            or np.any(np.diff(np.sort(ranked, axis=2), axis=2) == 0)):
        raise ValueError(f"Invalid ranks or dimensions in {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    rows = []
    for layer in range(28):
        ranks = ranked[:, layer]
        values = scores[:, layer].astype(np.float32)
        for t, limit in enumerate(np.maximum.accumulate(ranks.max(axis=1))):
            if not np.isfinite(values[t, :int(limit)+1]).all():
                raise ValueError(f"Missing score for {rid}, layer {layer}, step {t}")
        for B in CURVE:
            for cfg, (policy, kw) in configs(B, frozen).items():
                counts = kp.simulate(ranks, values, B, policy, **kw)
                rows.append((cfg, rid, layer, B, *[counts[k] for k in METRICS]))
                if cfg == "ema":
                    capacity = frozen["alloc"][str(B)][layer]
                    allocated = (counts if capacity == B else
                                 kp.simulate(ranks, values, capacity, policy, **kw))
                    rows.append(("alloc_ema", rid, layer, B, *[allocated[k] for k in METRICS]))
    return (rid, doc["prompt_tokens"], len(ranked), digest), rows


METRICS = ("accesses", "misses", "compulsory")


def evaluate(args):
    frozen = json.loads(args.frozen.read_text())
    split = json.loads(Path(__file__).with_name("splits").joinpath("longctx_v1.json").read_text())
    docs = [d for d in split["prompts"] if d["split"] == "heldout"]
    expected = {d["rid"].replace("/", "__") + ".npz": d for d in docs}
    dev = {d["rid"].replace("/", "__") + ".npz" for d in split["prompts"] if d["split"] == "dev"}
    actual = {p.name for p in args.traces.glob("*.npz")}
    if (len(expected) != 21 or len(docs) != 21 or expected.keys() & dev
            or set(frozen["dev_docs"]) != dev or actual != expected.keys()):
        raise ValueError(f"Held-out split mismatch: missing {sorted(expected.keys()-actual)}, "
                         f"extra {sorted(actual-expected.keys())}")
    for B in CURVE:
        allocation = frozen["alloc"][str(B)]
        if (len(allocation) != 28 or sum(allocation) != 28 * B
                or any(type(x) is not int or x < 64 for x in allocation)
                or not 0 < frozen["alpha"][str(B)] <= 1):
            raise ValueError(f"Invalid frozen allocation or alpha at B={B}")
    jobs = [(args.traces / name, d, frozen) for name, d in sorted(expected.items())]
    rows, inputs = [], []
    with Pool(args.procs) as pool:
        for meta, part in pool.imap_unordered(replay_document, jobs):
            inputs.append(meta)
            rows.extend(part)
            print(f"{len(inputs)}/{len(jobs)} replayed {meta[0]}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, header, data in [
        ("inputs.csv", ["doc", "prompt_tokens", "steps", "sha256"], inputs),
        ("misses.csv", ["config", "doc", "layer", "B", *METRICS], rows),
    ]:
        with (args.out / filename).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(sorted(data))


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
