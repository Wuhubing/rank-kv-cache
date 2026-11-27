"""Freeze the Qwen3-8B held-out parameters on the 14 8B dev documents (dev only).

  python -m analysis_v2.freeze_8b DEV8B_TRACES --split splits/longctx_8b_v1.json \
      --out splits/heldout_frozen_8b.json [--procs 32]

Same procedure as the original freeze, with the EMA decay chosen on the 8B dev documents instead of
copied from the 0.6B dev: for every budget B in heldout.CURVE, alpha_B = the value in ALPHA_GRID with
the fewest pooled dev misses (ties: smaller alpha). This is the choice leave-one-document-out
selection converges to when no document is held out; the per-fold LODO choices are written next to
the frozen file for reference. Then heldout.freeze runs unchanged (per-layer allocation from dev EMA
curves on heldout.GRID, two-iteration DAgger linear coefficients, dev check counts).
"""
import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import heldout
import kvpolicy as kp

ALPHA_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def doc_misses(job):
    path, budgets = job
    R, S, rid = kp.load(path)
    out = {}
    for a in ALPHA_GRID:
        for B in budgets:
            out[a, B] = sum(kp.simulate(R[:, l], S[:, l], B, "ema", alpha=a)["misses"] for l in range(R.shape[1]))
    return rid, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", type=Path)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    paths = sorted(str(p) for p in a.traces.glob("*.npz"))
    split, _ = heldout.load_split(a.split)
    heldout.check_traces(paths, split, "dev")
    with Pool(min(a.procs, len(paths))) as pool:
        per = dict(pool.map(doc_misses, [(p, heldout.CURVE) for p in paths]))
    docs = sorted(per)
    alpha, lodo = {}, defaultdict(dict)
    for B in heldout.CURVE:
        alpha[B] = min(ALPHA_GRID, key=lambda x: (sum(per[d][x, B] for d in docs), x))
        for d in docs:
            lodo[B][d] = min(ALPHA_GRID, key=lambda x: (sum(per[e][x, B] for e in docs if e != d), x))
    print("alpha", alpha)
    heldout.ALPHA = alpha
    heldout.freeze(SimpleNamespace(traces=a.traces, out=a.out, split=a.split, procs=a.procs))
    frozen = json.loads(a.out.read_text())
    frozen["alpha_selection"] = dict(grid=list(ALPHA_GRID), rule="fewest pooled dev misses, ties smaller alpha",
                                     dev_misses={str(B): {str(x): sum(per[d][x, B] for d in docs) for x in ALPHA_GRID}
                                                 for B in heldout.CURVE},
                                     lodo_choices={str(B): lodo[B] for B in heldout.CURVE})
    a.out.write_text(json.dumps(frozen, indent=1) + "\n")


if __name__ == "__main__":
    main()
