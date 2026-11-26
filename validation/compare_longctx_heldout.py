"""Compare a completed held-out replay to the archive, then summarize paired documents."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEY = ("doc", "layer", "B", "config")
COUNTS = ("accesses", "misses", "compulsory")


def index(path):
    records = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = row["doc"], int(row["layer"]), int(row["B"]), row["config"]
            if key in records:
                raise ValueError(f"Duplicate key {key} in {path}")
            records[key] = tuple(int(row[k]) for k in COUNTS)
    return records


def write(path, keys, rows):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)


def compare(out):
    archived = index(ROOT / "docs/results/longctx_heldout/misses.csv")
    replay = index(out / "misses.csv")
    split = json.loads((ROOT / "splits/longctx_v1.json").read_text())
    docs = sorted(d["rid"] for d in split["prompts"] if d["split"] == "heldout")
    budgets = split["budgets"]["curve_B"]
    policies = {b: ["lru", "belady", "score", "ema", "alloc_ema", "dsum", "dcount"] +
                (["linear"] if b in (96, 128, 256) else []) for b in budgets}
    expected = {(d, l, b, p) for d in docs for l in range(28) for b in budgets for p in policies[b]}
    checks = []
    errors = 0
    totals = defaultdict(int)
    for key in sorted(expected | archived.keys() | replay.keys()):
        a, r = archived.get(key), replay.get(key)
        match = key in expected and a is not None and a == r
        errors += not match
        checks.append((*key, *(r or ("",)*3), *(a or ("",)*3), match))
        if r is not None:
            doc, _, b, policy = key
            totals[doc, b, policy] += r[1] - r[2]
    write(out/"layer_check.csv", [*KEY, *COUNTS, *["archive_"+c for c in COUNTS], "match"], checks)
    if errors:
        raise ValueError(f"{errors} missing, unexpected or mismatched keys; see layer_check.csv")
    print(f"{len(checks)} layer rows matched exactly.")
    write(out/"repeat_by_document.csv", ["doc","B","policy","repeat_misses"],
          [(*k,v) for k,v in sorted(totals.items())])
    idx = np.random.default_rng(0).integers(len(docs), size=(2000,len(docs)))
    rows = []
    for b in budgets:
        values = {p: np.array([totals[d,b,p] for d in docs]) for p in policies[b]}
        gap = values["lru"] - values["belady"]
        for base, policy in ([("lru",p) for p in policies[b]] + [("ema","alloc_ema"),("score","ema")]):
            ref = values[base]
            delta = ref - values[policy]
            def ratio(den):
                ci = np.percentile(delta[idx].sum(axis=1)/den[idx].sum(axis=1), [2.5,97.5])
                return (delta.sum()/den.sum(), *ci)
            rows.append((b,base,policy,values[policy].sum(),delta.sum(),*ratio(ref),*ratio(gap),
                         sum(delta>0),sum(delta==0),sum(delta<0)))
    write(out/"paired.csv", ["B","baseline","policy","repeat_misses","saved","relative_drop",
                             "drop_ci_low","drop_ci_high","delta_G","G_ci_low","G_ci_high",
                             "wins","ties","losses"], rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="Directory with replayed misses.csv")
    compare(parser.parse_args().out)
