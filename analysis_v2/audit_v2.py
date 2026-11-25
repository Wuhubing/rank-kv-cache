"""Check LRU and Belady counts of the kvpolicy replays against the independent simulator in replay.py
(replay_stream) on the new traces, for every document, layer and budget present in MISSES.

  python -m analysis_v2.audit_v2 TRACES --misses results_v2/dev/efficient_dev/misses.csv \
      [--misses more.csv ...] --out results_v2/dev/audit.txt [--budgets 72 96 128 256] [--procs 32]

Exit status 1 on any mismatch. Also checks Belady <= LRU per layer and a nonzero miss rate.
"""
import argparse
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import kvpolicy as kp
from replay import replay_stream


def doc_counts(args):
    path, budgets = args
    R, _, rid = kp.load(path)
    out = {}
    for l in range(R.shape[1]):
        steps = [[int(p) for p in row if p >= 0] for row in R[:, l]]
        for B in budgets:
            for pol in ("lru", "belady"):
                c = replay_stream(steps, B, pol)
                out[rid, l, pol, B] = (c["accesses"], c["misses"], c["compulsory_misses"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", type=Path)
    ap.add_argument("--misses", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budgets", type=int, nargs="+", default=[72, 96, 128, 256])
    ap.add_argument("--procs", type=int, default=32)
    args = ap.parse_args()
    ref = {}
    for m in args.misses:
        with open(m) as f:
            for r in csv.DictReader(f):
                cfg = r.get("config")
                cfg = "lru" if cfg == "lru_fine" else cfg
                if cfg in ("lru", "belady") and int(r["B"]) in args.budgets:
                    ref[r["doc"], int(r["layer"]), cfg, int(r["B"])] = (
                        int(r["accesses"]), int(r["misses"]), int(r["compulsory"]))
    paths = sorted(str(p) for p in args.traces.glob("*.npz"))
    with Pool(args.procs) as pool:
        ind = {}
        for part in pool.imap_unordered(doc_counts, [(p, args.budgets) for p in paths]):
            ind.update(part)
    bad, checked = [], 0
    for key, v in ind.items():
        if key in ref:
            checked += 1
            if ref[key] != v:
                bad.append((key, ref[key], v))
    order = []
    for (rid, l, pol, B), v in ind.items():
        if pol == "lru":
            o = ind[rid, l, "belady", B]
            if o[1] > v[1]:
                order.append((rid, l, B, v[1], o[1]))
    zero = sum(1 for (rid, l, pol, B), v in ind.items() if pol == "lru" and v[1] == 0)
    lines = [f"traces: {len(paths)}; independent replays: {len(ind)}; matched against kvpolicy rows: {checked}",
             f"mismatches: {len(bad)}", f"layers with Belady > LRU: {len(order)}",
             f"layer streams with zero LRU misses: {zero}"]
    lines += [f"MISMATCH {k}: kvpolicy {a} replay.py {b}" for k, a, b in bad[:50]]
    tot = defaultdict(lambda: [0, 0])
    for (rid, l, pol, B), v in ind.items():
        tot[pol, B][0] += v[0]
        tot[pol, B][1] += v[1]
    for (pol, B), (a, m) in sorted(tot.items(), key=lambda x: (x[0][1], x[0][0])):
        lines.append(f"{pol:>6} B={B:>3}: accesses {a}, misses {m}, miss rate {100*m/a:.3f}%")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(1 if (bad or order or checked == 0) else 0)


if __name__ == "__main__":
    main()
