"""Replay npz traces with a few policies and print miss rates and G per document and pooled.

  python local/quick_replay.py traces/pilot --budgets 96 128
  python local/quick_replay.py traces/mps_dev --compare docs/results/longctx_dev/misses.csv

Configs use efficient_dev.CONFIGS names (lru, belady, score, ema:0.5, rank:8:2, ...).
G = (LRU - policy) / (LRU - Belady) misses. --compare prints the same quantities from an earlier
misses.csv (e.g. the L4 dev run) for the documents both sets share. Rows are also written to
--out in the misses.csv layout (doc, layer, config, B, accesses, misses, compulsory).
"""
import argparse
import csv
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import efficient_dev as ed  # noqa: E402


def g(lru, pol, opt):
    return (lru - pol) / (lru - opt) if lru != opt else float("nan")


def table(counts, configs, budgets, label):
    """counts[(doc, config, B)] = [accesses, misses]"""
    docs = sorted({d for d, _, _ in counts})
    others = [c for c in configs if c not in ("lru", "belady")]
    for B in budgets:
        print(f"\n[{label}] B={B}   miss rate: LRU / Belady    G: " + "  ".join(others))
        tot = defaultdict(lambda: [0, 0])
        for d in docs + ["ALL"]:
            if d == "ALL":
                m = {c: tot[c] for c in configs}
            else:
                m = {c: counts.get((d, c, B)) for c in configs}
                if any(v is None for v in m.values()):
                    continue
                for c in configs:
                    tot[c][0] += m[c][0]; tot[c][1] += m[c][1]
            acc = m["lru"][0]
            lru, opt = m["lru"][1], m["belady"][1]
            gs = "  ".join(f"{g(lru, m[c][1], opt):+.3f}" for c in others)
            print(f"  {d[:28]:28s} {100*lru/acc:6.2f}% / {100*opt/acc:6.2f}%   {gs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", type=Path)
    ap.add_argument("--budgets", type=int, nargs="+", default=[96])
    ap.add_argument("--configs", nargs="+", default=["lru", "belady", "score", "ema:0.5", "rank:8:2"])
    ap.add_argument("--compare", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    for c in ("lru", "belady"):
        if c not in args.configs:
            args.configs.insert(0, c)
    paths = sorted(args.traces.glob("*.npz"))
    if not paths:
        sys.exit(f"no npz in {args.traces}")
    with Pool(args.procs) as pool:
        parts = pool.map(ed.run_doc, [(str(p), args.configs, args.budgets) for p in paths])
    rows = [r for part in parts for r in part]
    counts = defaultdict(lambda: [0, 0])
    for doc, l, c, B, acc, mis, comp in rows:
        counts[doc, c, B][0] += acc; counts[doc, c, B][1] += mis
    table(dict(counts), args.configs, args.budgets, args.traces.name)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["doc", "layer", "config", "B", "accesses", "misses", "compulsory"]); w.writerows(rows)
    if args.compare:
        mine = {d for d, _, _ in counts}
        old = defaultdict(lambda: [0, 0])
        with open(args.compare, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["doc"], r["config"], int(r["B"]))
                if r["doc"] in mine and key[1] in args.configs and key[2] in args.budgets:
                    old[key][0] += int(r["accesses"]); old[key][1] += int(r["misses"])
        if old:
            table(dict(old), args.configs, args.budgets, f"compare: {args.compare}")
        else:
            print("\nno shared documents/configs in", args.compare)


if __name__ == "__main__":
    main()
