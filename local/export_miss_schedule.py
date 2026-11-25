"""Export per-step, per-layer miss counts of the frozen held-out policies for the Colab
transfer benchmark (notebooks/batch_transfer_bench.ipynb).

  python local/export_miss_schedule.py traces/mps_heldout --frozen splits/heldout_frozen_mps.json \
      --out docs/results/mps_heldout/miss_schedule.npz

npz keys: "rids" (document IDs), "budgets", and "<i>/<policy>@<B>" -> int16 [T, L] misses of
document i at decode step t, layer l. Policies: lru, ema, alloc_ema (frozen per-layer budgets
for total B), belady. Totals are checked against the held-out misses.csv when --check is given.
"""
import argparse
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import heldout as ho  # noqa: E402
import kvpolicy as kp  # noqa: E402


def doc_schedule(job):
    path, frozen, budgets = job
    R, S, rid = kp.load(path)
    L = R.shape[1]
    out = {}
    for B in budgets:
        alpha = ho.alpha_of(B, frozen)
        alloc = frozen["alloc"][str(B)]
        for pol, name, kw, per_layer_B in (("lru", "lru", {}, None), ("ema", "ema", dict(alpha=alpha), None),
                                           ("alloc_ema", "ema", dict(alpha=alpha), alloc),
                                           ("belady", "belady", {}, None)):
            M = np.zeros((R.shape[0], L), np.int16)
            for l in range(L):
                steps = []
                kp.simulate(R[:, l], S[:, l] if name == "ema" else None,
                            per_layer_B[l] if per_layer_B else B, name, per_step=steps, **kw)
                M[:, l] = steps
            out[f"{pol}@{B}"] = M
    print("done", rid, flush=True)
    return rid, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", type=Path)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=[96, 128])
    ap.add_argument("--check", type=Path, help="held-out misses.csv to compare totals with")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    import json
    frozen = json.loads(args.frozen.read_text())
    paths = sorted(args.traces.glob("*.npz"))
    with Pool(args.procs) as pool:
        res = sorted(pool.map(doc_schedule, [(str(p), frozen, args.budgets) for p in paths]))
    arrays = {"rids": np.array([r for r, _ in res]), "budgets": np.array(args.budgets)}
    for i, (_, out) in enumerate(res):
        for k, v in out.items():
            arrays[f"{i}/{k}"] = v
    if args.check:
        tot = defaultdict(int)
        for r in csv.DictReader(open(args.check)):
            tot[r["config"], r["doc"], int(r["B"])] += int(r["misses"])
        for i, (rid, out) in enumerate(res):
            for k, v in out.items():
                pol, B = k.split("@")
                assert int(v.sum()) == tot[pol, rid, int(B)], (rid, k, int(v.sum()), tot[pol, rid, int(B)])
        print("totals match", args.check)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
