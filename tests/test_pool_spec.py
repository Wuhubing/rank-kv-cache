"""R11-4 (review of PR #11): the global slot pool written only from the text of the appendix of
docs/preregistration_v3.md ("Exact semantics of the global slot pool", 2026-11-27) must match
analysis_v3.c4_alloc.pool_run miss for miss. The implementation below deliberately does not reuse pool_run or
kvpolicy helpers; it follows the numbered rules of the appendix."""
import math

import numpy as np
import pytest

from analysis_v3 import c4_alloc as c4


def spec_pool(R, S, total, kind, alpha=0.5, signal="allrank", beta=0.0):
    T, L, _ = R.shape
    resident = {}                       # (layer, page) -> {"last": u, "v": value}
    m = [0.0] * L                       # rule 2: running miss rate per layer, starts at 0
    uses = {}                           # rule 5 (Belady): global times of every selection of (layer, page)
    for t in range(T):
        for l in range(L):
            for p in R[t, l]:
                if p >= 0:
                    uses.setdefault((l, int(p)), []).append(t * L + l)
    nxt = {q: 0 for q in uses}          # index of the next not-yet-served use
    total_misses = 0
    for t in range(T):
        for l in range(L):              # rule 1: layers one at a time, u = t*L + l
            u = t * L + l
            sel = [int(p) for p in R[t, l] if p >= 0]    # rule 0: top-K in descending score order
            srow = S[t, l]

            def sig(p):
                if signal == "allrank":                  # rule 3: mid-rank percentile among finite scores
                    s = float(srow[p])
                    fin = srow[np.isfinite(srow)].astype(np.float64)
                    if not math.isfinite(s):
                        return 0.0
                    N = len(fin)
                    if N == 1:
                        return 1.0
                    less, eq = int((fin < s).sum()), int((fin == s).sum())
                    return (less + (eq - 1) / 2) / (N - 1)
                s1, sK = srow[sel[0]], srow[sel[-1]]     # rule 3, signal z (not frozen)
                return (srow[p] - sK) / (s1 - sK + 1e-6)

            if kind == "ema":           # rule 3: every resident page of layer l, before eviction
                for (ll, p), e in resident.items():
                    if ll == l:
                        e["v"] = alpha * e["v"] + (1 - alpha) * sig(p)
            miss = [p for p in sel if (l, p) not in resident]
            n = len(resident) + len(miss) - total
            if n > 0:                   # rule 1: only layer l's current selection is protected
                cands = [q for q in resident if not (q[0] == l and q[1] in sel)]

                def key(q):
                    e = resident[q]
                    if kind == "lru":   # rule 5
                        return (e["last"], q[0], q[1])
                    if kind == "belady":
                        k = nxt[q]
                        nu = uses[q][k] if k < len(uses[q]) else math.inf
                        return (-nu, q[0], q[1])
                    return (e["v"] + beta * m[q[0]], e["last"], q[0], q[1])   # rule 4

                for q in sorted(cands, key=key)[:n]:
                    del resident[q]
            for p in sel:
                if (l, p) in resident:
                    resident[l, p]["last"] = u
                else:                   # rule 3: a newly admitted page starts at its current signal
                    resident[l, p] = {"last": u, "v": sig(p) if kind == "ema" else 0.0}
                nxt[l, p] += 1
            m[l] = 0.99 * m[l] + 0.01 * (len(miss) / len(sel))   # rule 2: after layer l is served
            total_misses += len(miss)
    return total_misses


def trace(seed, L=3, T=60, P0=20, K=8, ties=False):
    """Per layer: scores follow a random walk over a growing set of pages (NaN = page not yet complete);
    R = top-K by score, descending, -1 padded while a layer has fewer than K pages. ties=True rounds the scores."""
    rng = np.random.default_rng(seed)
    P = P0 + T
    R = np.full((T, L, K), -1, np.int64)
    S = np.full((T, L, P), np.nan, np.float32)
    for l in range(L):
        base = rng.normal(0, 1, P)
        for t in range(T):
            n = P0 + t
            base[:n] += rng.normal(0, 0.15, n)
            row = base[:n] + rng.normal(0, 0.3, n)
            if ties:
                row = np.round(row, 1)
            S[t, l, :n] = row
            top = np.argsort(-S[t, l, :n], kind="stable")[:K]
            R[t, l, :len(top)] = top
    return R, S


TRACES = [trace(0), trace(1, ties=True), trace(2, P0=3, ties=True)]
CONFIGS = ([("lru", {}), ("belady", {})] +
           [("ema", dict(signal=s, beta=b, alpha=a)) for s in ("allrank", "z") for b in (0.0, 0.5, 2.0)
            for a in (0.5, 0.3)])


@pytest.mark.parametrize("ti", range(len(TRACES)))
@pytest.mark.parametrize("kind,kw", CONFIGS)
@pytest.mark.parametrize("B", [9, 12, 16])
def test_appendix_text_reproduces_pool_run(ti, kind, kw, B):
    R, S = TRACES[ti]
    total = B * R.shape[1]
    ref, _ = c4.pool_run(R, S, total, kind, **kw)
    assert spec_pool(R, S, total, kind, **kw) == ref


def test_short_rows_exercise_the_selection_size_rule():
    """trace(2) starts with 3 pages per layer < K = 8, so the miss rate divides by the selection size, not K."""
    R, _ = TRACES[2]
    assert (R[0] < 0).any() and (R[-1] >= 0).all()
