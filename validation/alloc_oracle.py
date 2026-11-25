"""Allocation-aware anchors for per-layer budget allocation (needs the raw traces).

The committed G of allocated policies (alloc_ema on held-out, alloc+* on dev) uses LRU and Bélády at
the *uniform* budget B in every layer. Once slots move between layers, uniform Bélády is not the
optimum for that allocation, so that G is not a "fraction of the gap". This script replays LRU and
Bélády at exactly the per-layer budgets the allocated policy used and reports, per (label, B):

  G_uniform_anchors   (LRU_uniform - pol) / (LRU_uniform - Belady_uniform)      as committed
  G_vs_alloc_belady   (LRU_uniform - pol) / (LRU_uniform - Belady_same_alloc)   gap to the oracle at the same allocation
  G_within_alloc      (LRU_same_alloc - pol) / (LRU_same_alloc - Belady_same_alloc)   replacement-only gain at that allocation
  repeat_miss_drop_vs_lru   1 - (pol - compulsory) / (LRU_uniform - compulsory)   equal total slots

Held-out (per-layer budgets from frozen.json, the same for every document):
  python validation/alloc_oracle.py heldout HELDOUT_TRACES --out DIR \\
      [--misses docs/results/longctx_heldout/misses.csv] [--frozen docs/results/longctx_heldout/frozen.json]
Dev (per-document budgets from the LODO allocation in alloc.csv):
  python validation/alloc_oracle.py dev DEV_TRACES --out DIR \\
      [--misses docs/results/longctx_dev/misses.csv] [--alloc docs/results/longctx_dev/alloc.csv]

Writes alloc_anchors_by_doc.csv and alloc_anchors.csv (new files; committed results are untouched).
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import kvpolicy as kp  # noqa: E402

REPS, SEED = 2000, 0


def read_misses(path):
    """M[(config, B)][doc] = [accesses, misses, compulsory] summed over layers (dev or held-out layout)."""
    M = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            cell = M[r["config"], int(r["B"])][r["doc"]]
            for i, k in enumerate(("accesses", "misses", "compulsory")):
                cell[i] += int(r[k])
    return M


def heldout_jobs(M, frozen):
    jobs = []
    for B, per_layer in sorted(frozen["alloc"].items(), key=lambda kv: int(kv[0])):
        B = int(B)
        for doc, cell in M["alloc_ema", B].items():
            jobs.append(("alloc_ema", B, doc, tuple(per_layer), cell[1]))
    return jobs


def dev_jobs(alloc_csv):
    jobs = []
    with open(alloc_csv, newline="") as f:
        for r in csv.DictReader(f):
            jobs.append((f"alloc+{r['base']}", int(r["B"]), r["doc"],
                         tuple(int(x) for x in r["per_layer"].split()), int(r["misses"])))
    return jobs


def trace_index(folder):
    out = {}
    for path in sorted(Path(folder).glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            out[str(z["rid"])] = str(path)
    return out


def replay(job):
    path, budgets = job
    R, _, rid = kp.load(path)
    if R.shape[1] != len(budgets):
        raise ValueError(f"{rid}: {R.shape[1]} layers but {len(budgets)} per-layer budgets")
    per = {l: b for l, b in enumerate(budgets)}
    lru = kp.doc_counts(R, None, per, "lru")
    opt = kp.doc_counts(R, None, per, "belady")
    return rid, budgets, sum(c["misses"] for c in lru.values()), sum(c["misses"] for c in opt.values())


def run(jobs, M, traces, out, workers):
    index = trace_index(traces)
    missing = sorted({doc for _, _, doc, _, _ in jobs} - index.keys())
    if missing:
        raise ValueError(f"No trace for {len(missing)} documents, e.g. {missing[:3]}")
    keys = sorted({(index[doc], per) for _, _, doc, per, _ in jobs})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        same = {(rid, per): (lru, opt) for rid, per, lru, opt in pool.map(replay, keys)}
    by_doc = []
    for label, B, doc, per, pol in jobs:
        lru_a, opt_a = same[doc, per]
        lru_u, opt_u = M["lru", B][doc][1], M["belady", B][doc][1]
        comp = M["lru", B][doc][2]
        if sum(per) > len(per) * B:
            raise ValueError(f"{label} {doc} B={B}: allocation uses more than L*B slots")
        if all(b == B for b in per) and (lru_a, opt_a) != (lru_u, opt_u):
            raise ValueError(f"{label} {doc} B={B}: uniform plan but same-allocation anchors "
                             f"{(lru_a, opt_a)} differ from the archived uniform {(lru_u, opt_u)}")
        if label == "alloc+lru" and pol != lru_a:
            raise ValueError(f"{doc} B={B}: archived alloc+lru misses {pol} differ from replay {lru_a}")
        by_doc.append((label, B, doc, pol, lru_u, opt_u, lru_a, opt_a, comp, " ".join(map(str, per))))
    by_doc.sort()
    head = ["label", "B", "doc", "policy_misses", "lru_uniform", "belady_uniform", "lru_same_alloc",
            "belady_same_alloc", "compulsory", "per_layer"]
    write(out / "alloc_anchors_by_doc.csv", head, by_doc)
    summary = summarise(by_doc)
    write(out / "alloc_anchors.csv", list(summary[0]), [list(r.values()) for r in summary])
    for r in summary:
        print(f"{r['label']:<16} B={r['B']:>3}  G(uniform anchors) {r['G_uniform_anchors']:+.3f}  "
              f"G(vs same-alloc Belady) {r['G_vs_alloc_belady']:+.3f}  G(within alloc) {r['G_within_alloc']:+.3f}  "
              f"repeat-drop {100 * r['repeat_miss_drop_vs_lru']:+.1f}%")
    return summary


def summarise(by_doc):
    groups = defaultdict(list)
    for row in by_doc:
        groups[row[0], row[1]].append(row)
    out = []
    for (label, B), rows in sorted(groups.items()):
        a = np.array([r[3:9] for r in rows], float)  # pol, lru_u, opt_u, lru_a, opt_a, comp
        idx = np.random.default_rng(SEED).integers(0, len(rows), (REPS, len(rows)))

        def stats(i):
            pol, lu, ou, la, oa, c = a[i].sum(0)
            return ((lu - pol) / (lu - ou), (lu - pol) / (lu - oa), (la - pol) / (la - oa),
                    1 - (pol - c) / (lu - c))
        point = stats(np.arange(len(rows)))
        boot = np.array([stats(i) for i in idx])
        lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
        names = ("G_uniform_anchors", "G_vs_alloc_belady", "G_within_alloc", "repeat_miss_drop_vs_lru")
        row = dict(label=label, B=B, n_docs=len(rows), policy_misses=int(a[:, 0].sum()),
                   lru_uniform=int(a[:, 1].sum()), belady_uniform=int(a[:, 2].sum()),
                   lru_same_alloc=int(a[:, 3].sum()), belady_same_alloc=int(a[:, 4].sum()))
        for n, p, l, h in zip(names, point, lo, hi):
            row.update({n: float(p), f"{n}_lo": float(l), f"{n}_hi": float(h)})
        out.append(row)
    return out


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", choices=("heldout", "dev"))
    ap.add_argument("traces", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--misses", type=Path)
    ap.add_argument("--frozen", type=Path, default=ROOT / "docs/results/longctx_heldout/frozen.json")
    ap.add_argument("--alloc", type=Path, default=ROOT / "docs/results/longctx_dev/alloc.csv")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    default = "longctx_heldout" if args.split == "heldout" else "longctx_dev"
    M = read_misses(args.misses or ROOT / "docs/results" / default / "misses.csv")
    jobs = heldout_jobs(M, json.loads(args.frozen.read_text())) if args.split == "heldout" else dev_jobs(args.alloc)
    args.out.mkdir(parents=True, exist_ok=True)
    return run(jobs, M, args.traces, args.out, args.workers)


if __name__ == "__main__":
    main()
