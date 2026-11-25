"""Extra numbers for the replication report (not decision criteria).

  python -m analysis_v2.s5_extra doclevel results_v2/heldout/eval/misses.csv            # per-doc / per-task G
  python -m analysis_v2.s5_extra ties TRACES [TRACES ...] --out results_v2/heldout/ties.csv   # float16 ties
"""
import argparse
import csv
import glob
from collections import defaultdict
from multiprocessing import Pool

import numpy as np


def doclevel(path, B=96):
    M = defaultdict(lambda: defaultdict(int))
    for r in csv.DictReader(open(path)):
        if int(r["B"]) == B:
            M[r["config"]][r["doc"]] += int(r["misses"])
    docs = sorted(M["lru"])
    g = lambda c, ds: (sum(M["lru"][d] for d in ds) - sum(M[c][d] for d in ds)) / \
        (sum(M["lru"][d] for d in ds) - sum(M["belady"][d] for d in ds))
    for c in ("ema", "alloc_ema", "score", "dsum", "linear"):
        per = [g(c, [d]) for d in docs]
        tasks = sorted({d.split("/")[0] for d in docs})
        bytask = {t: round(g(c, [d for d in docs if d.startswith(t)]), 3) for t in tasks}
        print(f"{c:<10} B={B} per-doc G min {min(per):.3f} median {np.median(per):.3f} max {max(per):.3f}; "
              f"by task {bytask}; docs better than LRU {sum(M[c][d] < M['lru'][d] for d in docs)}/{len(docs)}")


def tie_doc(path):
    z = np.load(path)
    S, R = z["scores"], z["ranked"]
    T, L, P = S.shape
    tied_pages = total_pages = steps_with_tie = sel_boundary_tie = 0
    for t in range(0, T, 8):            # every 8th step (sampled)
        for l in range(L):
            s = S[t, l]
            s = s[np.isfinite(s)]
            u, c = np.unique(s, return_counts=True)
            tied_pages += int(c[c > 1].sum())
            total_pages += len(s)
            steps_with_tie += int((c > 1).any())
            kth = S[t, l, R[t, l, -1]]
            sel_boundary_tie += int((s == kth).sum() > 1)
    n = len(range(0, T, 8)) * L
    return str(z["rid"]), tied_pages / total_pages, steps_with_tie / n, sel_boundary_tie / n


def ties(dirs, out):
    paths = sorted(p for d in dirs for p in glob.glob(f"{d}/*.npz"))
    with Pool(14) as pool:
        rows = pool.map(tie_doc, paths)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc", "share_of_pages_with_tied_score", "share_of_layer_steps_with_any_tie",
                    "share_of_layer_steps_with_tie_at_kth_selected"])
        w.writerows(rows)
    a = np.array([r[1:] for r in rows])
    print(f"{len(rows)} docs; mean share of pages with a tied float16 score {a[:, 0].mean():.4f}; "
          f"layer-steps with any tie {a[:, 1].mean():.3f}; with a tie at the K-th selected score {a[:, 2].mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["doclevel", "ties"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "doclevel":
        for B in (96, 128):
            doclevel(a.paths[0], B)
    else:
        ties(a.paths, a.out)


if __name__ == "__main__":
    main()
