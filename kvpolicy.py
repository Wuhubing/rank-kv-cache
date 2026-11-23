"""Trace-driven page-cache simulator and replacement policy family for long npz traces.

A trace is one document: ranked int16 [T, L, K] (top-K page IDs per step, best first) and
scores float16 [T, L, P] (selector score of every completed historical page, NaN beyond).
HiSparse semantics: each step's top-K pages must be resident, so a step is a batch of
compulsory admissions; a policy only chooses which unselected residents to evict.

Every policy keeps state only for resident pages unless noted (ARC ghosts, LRU-2 history).
"""
import heapq
import math
from collections import OrderedDict, defaultdict, deque

import numpy as np


class Policy:
    """Base: LRU bookkeeping. evict() returns n victims from candidates (unselected residents)."""
    needs_scores = False

    def __init__(self, K, **kw):
        self.K = K
        self.last = {}
        self.hit = {}
        self.params = kw

    def begin(self, t, ranked, srow):
        self.t, self.ranked, self.srow = t, ranked, srow

    def evict(self, cands, n):
        return heapq.nsmallest(n, cands, key=self.key)

    def key(self, p):
        return self.last[p], self.hit[p], -p

    def dropped(self, p):
        self.last.pop(p, None)
        self.hit.pop(p, None)

    def admit(self, hits, misses):
        for p in self.ranked:
            self.last[p] = self.t
            self.hit[p] = p in hits


class LRU(Policy):
    pass


class FIFO(Policy):
    def __init__(self, K, **kw):
        super().__init__(K, **kw)
        self.adm = {}

    def key(self, p):
        return self.adm[p], -p

    def dropped(self, p):
        super().dropped(p)
        del self.adm[p]

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in misses:
            self.adm[p] = self.t


class LFU(Policy):
    """In-cache selection count (reset on eviction), ties by recency."""
    def __init__(self, K, **kw):
        super().__init__(K, **kw)
        self.cnt = defaultdict(int)

    def key(self, p):
        return self.cnt[p], self.last[p], self.hit[p], -p

    def dropped(self, p):
        super().dropped(p)
        self.cnt.pop(p, None)

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in self.ranked:
            self.cnt[p] += 1


class LRU2(Policy):
    """LRU-K with K=2: evict the oldest second-to-last selection (none = oldest). Keeps history
    for every page ever selected (classic LRU-K retained-information period = whole trace)."""
    def __init__(self, K, **kw):
        super().__init__(K, **kw)
        self.prev = {}
        self.hist = {}

    def key(self, p):
        return self.prev.get(p, -1), self.last[p], self.hit[p], -p

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in self.ranked:
            if p in self.hist:
                self.prev[p] = self.hist[p]
            self.hist[p] = self.t


class ARC(Policy):
    """ARC adapted to batched compulsory admissions: T1/T2 recency/frequency lists with ghosts
    B1/B2 (|B1|+|B2| <= c) and adaptive target p; victims never come from the current selection."""
    def __init__(self, K, B=None, **kw):
        super().__init__(K, **kw)
        self.c = B
        self.p = 0.0
        self.T1, self.T2, self.B1, self.B2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

    def evict(self, cands, n):
        cands = set(cands)
        out = []
        for _ in range(n):
            t1 = [x for x in self.T1 if x in cands]
            t2 = [x for x in self.T2 if x in cands]
            use1 = t1 and (len(self.T1) > self.p or not t2)
            v = t1[0] if use1 else t2[0]
            cands.discard(v)
            out.append(v)
            src, ghost = (self.T1, self.B1) if v in self.T1 else (self.T2, self.B2)
            del src[v]
            ghost[v] = None
            while len(self.B1) + len(self.B2) > self.c:
                (self.B1 if len(self.B1) > len(self.B2) else self.B2).popitem(last=False)
        return out

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for x in self.ranked:
            if x in hits:
                (self.T1 if x in self.T1 else self.T2).pop(x)
                self.T2[x] = None
            elif x in self.B1:
                self.p = min(self.c, self.p + max(len(self.B2) / len(self.B1), 1))
                del self.B1[x]
                self.T2[x] = None
            elif x in self.B2:
                self.p = max(0.0, self.p - max(len(self.B1) / len(self.B2), 1))
                del self.B2[x]
                self.T2[x] = None
            else:
                self.T1[x] = None


class RankFull(Policy):
    """Stage-1/2 rank policy: last + delta*[last rank in top quartile] + beta*min(freq over 8 steps, 4)."""
    def __init__(self, K, delta=4, beta=2, **kw):
        super().__init__(K, **kw)
        self.delta, self.beta = delta, beta
        self.lrank = {}
        self.window = deque(maxlen=8)
        self.q = -(-K // 4)

    def key(self, p):
        f = sum(p in s for s in self.window)
        return (self.last[p] + self.delta * (self.lrank[p] <= self.q) + self.beta * min(f, 4),
                self.last[p], -p)

    def dropped(self, p):
        super().dropped(p)
        del self.lrank[p]

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for r, p in enumerate(self.ranked, 1):
            self.lrank[p] = r
        self.window.append(frozenset(self.ranked))


def _z(srow, ranked, p):
    """Score of p relative to the K-th selected score, in units of (top1 - K-th)."""
    kth, top = srow[ranked[-1]], srow[ranked[0]]
    return (srow[p] - kth) / (top - kth + 1e-6)


class ScoreCurrent(Policy):
    """Evict the lowest current selector score (candidates nearest to re-selection stay)."""
    needs_scores = True

    def key(self, p):
        return self.srow[p], self.last[p], self.hit[p], -p


class ScoreEMA(Policy):
    """Evict the lowest exponential moving average of the normalised score z (one float per resident)."""
    needs_scores = True

    def __init__(self, K, alpha=0.5, **kw):
        super().__init__(K, **kw)
        self.alpha = alpha
        self.ema = {}

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        a = self.alpha
        for p in self.ema:
            self.ema[p] = a * self.ema[p] + (1 - a) * _z(srow, ranked, p)

    def key(self, p):
        return self.ema[p], self.last[p], self.hit[p], -p

    def dropped(self, p):
        super().dropped(p)
        del self.ema[p]

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in misses:
            self.ema[p] = _z(self.srow, self.ranked, p)


class Hybrid(Policy):
    """LRU with a score bonus: key = last + w * z (z = current normalised score, or its EMA)."""
    needs_scores = True

    def __init__(self, K, w=4.0, alpha=0.0, **kw):
        super().__init__(K, **kw)
        self.w, self.alpha = w, alpha
        self.ema = {}

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        a = self.alpha
        for p in self.ema:
            self.ema[p] = a * self.ema[p] + (1 - a) * _z(srow, ranked, p)

    def key(self, p):
        return self.last[p] + self.w * self.ema[p], self.hit[p], -p

    def dropped(self, p):
        super().dropped(p)
        del self.ema[p]

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in misses:
            self.ema[p] = _z(self.srow, self.ranked, p)


FEATURES = ("age", "z", "z_ema", "log_cnt", "last_rank", "log_resid")


class Linear(Policy):
    """Learned next-use regression: predict log(1+next-use distance) from FEATURES with a linear
    model; evict the largest prediction. coef has len(FEATURES)+1 entries (bias last)."""
    needs_scores = True

    def __init__(self, K, coef=None, alpha=0.5, collect=None, **kw):
        super().__init__(K, **kw)
        self.coef = None if coef is None else np.asarray(coef, float)
        self.alpha = alpha
        self.ema, self.cnt, self.lrank, self.adm = {}, defaultdict(int), {}, {}
        self.collect = collect  # list to append (features, page, t) at each eviction

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        a = self.alpha
        for p in self.ema:
            self.ema[p] = a * self.ema[p] + (1 - a) * _z(srow, ranked, p)

    def features(self, p):
        return (math.log1p(self.t - self.last[p]), _z(self.srow, self.ranked, p), self.ema[p],
                math.log1p(self.cnt[p]), self.lrank[p] / self.K, math.log1p(self.t - self.adm[p]))

    def evict(self, cands, n):
        cands = list(cands)
        X = np.array([self.features(p) for p in cands])
        if self.collect is not None:
            self.collect.append((X, cands, self.t))
        if self.coef is None:  # behaviour policy: LRU
            order = sorted(range(len(cands)), key=lambda i: (self.last[cands[i]], self.hit[cands[i]], -cands[i]))
        else:
            pred = X @ self.coef[:-1] + self.coef[-1]
            order = np.argsort(-pred, kind="stable")
        return [cands[i] for i in order[:n]]

    def dropped(self, p):
        super().dropped(p)
        for d in (self.ema, self.cnt, self.lrank, self.adm):
            d.pop(p, None)

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for r, p in enumerate(self.ranked, 1):
            self.lrank[p] = r
            self.cnt[p] += 1
        for p in misses:
            self.adm[p] = self.t
            self.ema[p] = _z(self.srow, self.ranked, p)


class Belady(Policy):
    def __init__(self, K, future=None, **kw):
        super().__init__(K, **kw)
        self.future = future

    def key(self, p):
        f = self.future[p]
        return -(f[0] if f else math.inf), p

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in self.ranked:
            self.future[p].popleft()


POLICIES = dict(lru=LRU, fifo=FIFO, lfu=LFU, lru2=LRU2, arc=ARC, rank=RankFull,
                score=ScoreCurrent, ema=ScoreEMA, hybrid=Hybrid, linear=Linear, belady=Belady)


def future_uses(ranked):
    fut = defaultdict(deque)
    for t, row in enumerate(ranked):
        for p in row:
            fut[p].append(t)
    return fut


def simulate(ranked, scores, B, name, *, on_evict=None, **params):
    """Replay one layer stream. ranked: [T, K] ints; scores: [T, P] or None.
    Returns dict(accesses, misses, compulsory). on_evict(t, victims, policy) is optional."""
    rows = [tuple(int(p) for p in r if p >= 0) for r in np.asarray(ranked)]
    K = max(len(r) for r in rows)
    if B < K:
        raise ValueError("budget below K")
    cls = POLICIES[name]
    if name == "belady":
        params["future"] = future_uses(rows)
    if name == "arc":
        params["B"] = B
    pol = cls(K, **params)
    if cls.needs_scores and scores is None:
        raise ValueError(f"{name} needs scores")
    res, seen = set(), set()
    acc = miss = 0
    for t, row in enumerate(rows):
        srow = scores[t].tolist() if cls.needs_scores else None
        pol.begin(t, row, srow)
        sel = set(row)
        hits = sel & res
        misses = sel - hits
        n = len(res) + len(misses) - B
        if n > 0:
            victims = pol.evict(res - sel, n)
            if on_evict is not None:
                on_evict(t, victims, pol)
            for v in victims:
                res.discard(v)
                pol.dropped(v)
        res |= sel
        pol.admit(hits, misses)
        assert len(res) <= B
        acc += len(sel)
        miss += len(misses)
        seen |= sel
    return dict(accesses=acc, misses=miss, compulsory=len(seen))


def load(path):
    z = np.load(path, allow_pickle=False)
    return z["ranked"], z["scores"].astype(np.float32), str(z["rid"])


def doc_counts(ranked, scores, budgets, name, layers=None, **params):
    """{(layer, B): counts} for one document; budgets may be an int or {layer: B}."""
    out = {}
    L = ranked.shape[1]
    for l in (range(L) if layers is None else layers):
        B = budgets[l] if isinstance(budgets, dict) else budgets
        out[l] = simulate(ranked[:, l], None if scores is None else scores[:, l], B, name, **params)
    return out


def bootstrap(per_doc_lru, per_doc_pol, per_doc_opt, reps=2000, seed=0):
    """Document-clustered bootstrap CI for G = (LRU - pol)/(LRU - Belady) and miss-rate delta."""
    rng = np.random.default_rng(seed)
    a = np.array([per_doc_lru, per_doc_pol, per_doc_opt], float)  # misses per doc
    n = a.shape[1]
    gs = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        s = a[:, idx].sum(1)
        gs.append((s[0] - s[1]) / (s[0] - s[2]) if s[0] != s[2] else np.nan)
    return np.nanpercentile(gs, [2.5, 97.5])


def allocate(curves, L, total, grid):
    """Greedy per-layer budgets. curves[l][B] = misses on training docs; each layer starts at
    grid[0]; repeatedly give the next grid step to the layer with the largest misses saved per
    slot, while the total stays within L*B budget."""
    alloc = {l: 0 for l in range(L)}
    used = L * grid[0]
    while True:
        best, gain = None, 0.0
        for l in range(L):
            i = alloc[l]
            if i + 1 >= len(grid) or used + grid[i + 1] - grid[i] > total:
                continue
            g = (curves[l][grid[i]] - curves[l][grid[i + 1]]) / (grid[i + 1] - grid[i])
            if g > gain:
                best, gain = l, g
        if best is None:
            break
        used += grid[alloc[best] + 1] - grid[alloc[best]]
        alloc[best] += 1
    return {l: grid[i] for l, i in alloc.items()}
