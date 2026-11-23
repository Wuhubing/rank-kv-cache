"""Recompute the held-out success criteria and budget curve from the committed CSV alone (no traces).

  python recompute_heldout.py [docs/results/longctx_heldout/misses.csv]

Numbers should match docs/heldout_results.md ("Preregistered success criteria" and "Budget curve").
"""
import csv
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "docs/results/longctx_heldout/misses.csv"
M = defaultdict(lambda: defaultdict(int))  # M[(config, B)][doc] = misses summed over layers
A = defaultdict(lambda: defaultdict(int))  # accesses
for r in csv.DictReader(open(path)):
    k = (r["config"], int(r["B"]))
    M[k][r["doc"]] += int(r["misses"])
    A[k][r["doc"]] += int(r["accesses"])
docs = sorted(M["lru", 96])
vec = lambda c, B: np.array([M[c, B][d] for d in docs], float)


def G(lru, pol, opt):
    return (lru.sum() - pol.sum()) / (lru.sum() - opt.sum())


def boot(fn, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(docs)
    return np.percentile([fn(rng.integers(0, n, n)) for _ in range(reps)], [2.5, 97.5])


lru, opt, ema, al = vec("lru", 96), vec("belady", 96), vec("ema", 96), vec("alloc_ema", 96)
lo, hi = boot(lambda i: G(lru[i], ema[i], opt[i]))
print(f"(i)   EMA G = {G(lru, ema, opt):.3f}  CI [{lo:.3f}, {hi:.3f}]   pass: {lo > 0.15}")
wins = int((ema < lru).sum())
print(f"(ii)  EMA < LRU on {wins}/{len(docs)} docs   pass: {wins >= 18}")
d = lambda i: G(lru[i], al[i], opt[i]) - G(lru[i], ema[i], opt[i])
lo, hi = boot(d)
print(f"(iii) Alloc+EMA - EMA: dG = {d(np.arange(len(docs))):.3f}  CI [{lo:.3f}, {hi:.3f}]   pass: {lo > 0}")

print("\nB    LRU%   Belady%  " + "  ".join(f"{c:>9}" for c in ("ema", "alloc_ema", "dsum", "score", "dcount", "linear")))
for B in (72, 80, 96, 128, 256):
    acc = sum(A["lru", B].values())
    l, o = vec("lru", B), vec("belady", B)
    cells = [f"{G(l, vec(c, B), o):9.3f}" if (c, B) in M else f"{'-':>9}"
             for c in ("ema", "alloc_ema", "dsum", "score", "dcount", "linear")]
    print(f"{B:<4} {100*l.sum()/acc:5.2f}  {100*o.sum()/acc:6.2f}   " + "  ".join(cells))
