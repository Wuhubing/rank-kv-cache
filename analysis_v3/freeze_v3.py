"""F2 (followup-v3): write splits/heldout_frozen_v3.json from dev results only.

  python -m analysis_v3.freeze_v3 --out splits/heldout_frozen_v3.json \
      --p2-family pow_norm_kth --p2-misses-06b results_v3/b1/b1_dev/misses.csv.gz --p2-misses-8b results_v3/b1/b1_8b/misses.csv.gz \
      [--p4-rows-06b results_v3/c4/dev/rows.csv.gz --p4-rows-8b results_v3/c4/8b/rows.csv.gz --p4-family online]

Per model and budget B in {96, 128} (primary 96):
  ema_alpha       the alpha already frozen on that model's dev set (splits/heldout_frozen_v1.json for 0.6B,
                  splits/heldout_frozen_8b.json for 8B), as in the two earlier confirmations;
  offline_alloc   the per-layer allocation frozen in the same file (dev EMA curves, greedy, total L*B);
  p2              the config of the P2 family with the fewest pooled misses over all dev documents (the choice the LODO
                  folds converge to; the per-fold LODO choices are stored);
  p4              the online-reallocation / pool config with the fewest pooled dev misses (same rule), if given.
"""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from analysis_v3 import experiments

BUDGETS = (96, 128)
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    M = defaultdict(lambda: defaultdict(int))
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for r in csv.DictReader(f):
            M[r["config"], int(r["B"])][r["doc"]] += int(r["misses"])
    return M


def choose(M, cfgs, B):
    docs = sorted(M[cfgs[0], B])
    pooled = min(cfgs, key=lambda c: (sum(M[c, B].values()), c))
    lodo = []
    for d in docs:
        lodo.append(min(cfgs, key=lambda c: (sum(v for x, v in M[c, B].items() if x != d), c)))
    return pooled, lodo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--p2-family")
    ap.add_argument("--p2-misses-06b", type=Path)
    ap.add_argument("--p2-misses-8b", type=Path)
    ap.add_argument("--p4-family", choices=["online", "pool_ema:z", "pool_ema:allrank"])
    ap.add_argument("--p4-rows-06b", type=Path)
    ap.add_argument("--p4-rows-8b", type=Path)
    a = ap.parse_args()
    e = experiments.get("b1")
    out = {"primary_B": 96, "secondary_B": 128, "split": "splits/longctx_v3.json",
           "split_sha256": hashlib.sha256((ROOT / "splits/longctx_v3.json").read_bytes()).hexdigest(), "models": {}}
    for model, ffile, p2m, p4r in (("06b", "splits/heldout_frozen_v1.json", a.p2_misses_06b, a.p4_rows_06b),
                                   ("8b", "splits/heldout_frozen_8b.json", a.p2_misses_8b, a.p4_rows_8b)):
        fz = json.loads((ROOT / ffile).read_text())
        alloc = fz["alloc"]
        if isinstance(alloc, str):
            import ast
            alloc = ast.literal_eval(alloc)
        m = {"frozen_source": ffile, "budgets": {}}
        M2 = read(p2m) if (a.p2_family and p2m) else None
        M4 = read(p4r) if (a.p4_family and p4r) else None
        for B in BUDGETS:
            cfg = {"ema_alpha": fz["alpha"][str(B)], "offline_alloc": alloc[str(B)]}
            if M2 is not None:
                cfgs = e.families[a.p2_family]
                pooled, lodo = choose(M2, cfgs, B)
                pol, kw = e.configs[pooled]
                cfg["p2"] = {"family": a.p2_family, "config": pooled, "policy": pol, "kw": kw, "lodo_choices": lodo}
            if M4 is not None:
                cfgs = sorted(c for (c, b) in M4 if b == B and c.startswith(a.p4_family + ":"))
                pooled, lodo = choose(M4, cfgs, B)
                parts = pooled.split(":")
                if a.p4_family == "online":
                    kw = dict(N=int(parts[2]), s=int(parts[3]), decay=float(parts[4]))
                    cfg["p4"] = {"kind": "online", "config": pooled, "kw": kw, "lodo_choices": lodo,
                                 "note": "selected with alpha 0.5 on dev; replayed with the frozen ema_alpha"}
                else:
                    kw = dict(signal=parts[1], beta=float(parts[2]), alpha=0.5)
                    cfg["p4"] = {"kind": "pool", "config": pooled, "kw": kw, "lodo_choices": lodo}
            m["budgets"][str(B)] = cfg
        out["models"][model] = m
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: {B: {x: (y if x not in ("offline_alloc", "lodo_choices") else "...") for x, y in c.items()}
                          for B, c in v["budgets"].items()} for k, v in out["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
