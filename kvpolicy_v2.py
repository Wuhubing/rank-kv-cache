"""replication-v2 replacement baselines (docs/workshop/rerun_plan.md section 6.2).

Registered into kvpolicy.POLICIES; the original classes in kvpolicy.py are unchanged. Same
semantics as kvpolicy: every step's top-K must be resident, a policy only picks which unselected
residents to evict. Policies that use the trace's attention mass read self.arow (attn_mass[t],
aligned with ranked; set by kvpolicy.simulate(..., attn=...)), else they use the softmax proxy of
the existing InfLLM class.
"""
import heapq
import math
import random
from collections import OrderedDict

import numpy as np

from kvpolicy import Policy, ScoreCurrent, ScoreEMA, _z


class RandomEvict(Policy):
    """Evict uniformly at random among the unselected residents (seeded, reproducible)."""
    def __init__(self, K, seed=0, **kw):
        super().__init__(K, **kw)
        self.rng = random.Random(seed)

    def evict(self, cands, n):
        return self.rng.sample(sorted(cands), n)


def eligible_pages(srow):
    """Number of completed pages at this step (scores are finite on a prefix, NaN beyond)."""
    lo, hi = 0, len(srow)
    while lo < hi:
        mid = (lo + hi) // 2
        if srow[mid] == srow[mid]:
            lo = mid + 1
        else:
            hi = mid
    return lo


class ArkVale(ScoreCurrent):
    """ArkVale eviction (pku-liang/ArkVale select.cu): evict the lowest current score among the
    unselected residents, never evicting the first `sink` pages or the `win` most recent completed
    pages while they are resident (they keep their slots). Deviation: pinned pages are not
    force-loaded; under the Quest selector a page is fetched only when selected, and forced loads
    would change the access stream. If only pinned pages are candidates, pinned pages are evicted
    in current-score order."""
    def __init__(self, K, sink=2, win=2, **kw):
        super().__init__(K, **kw)
        self.sink, self.win = sink, win

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        hp = eligible_pages(srow)
        self.pinned = set(range(self.sink)) | set(range(max(0, hp - self.win), hp))

    def key(self, p):
        return (p in self.pinned, *super().key(p))


def step_mass(pol):
    """Per-selected-page attention mass at this step: the trace's attn_mass if present, else the
    InfLLM-class proxy (softmax of Quest scores over the selected pages, temperature sqrt(128))."""
    if getattr(pol, "arow", None) is not None:
        return {p: float(m) for p, m in zip(pol.ranked, pol.arow)}
    top = pol.srow[pol.ranked[0]]
    w = [math.exp((pol.srow[p] - top) / pol.tau) for p in pol.ranked]
    tot = sum(w)
    return {p: x / tot for p, x in zip(pol.ranked, w)}


class InfLLMEq1(Policy):
    """InfLLM paper App. A Eq. 1: every resident s <- d*s + mass (mass 0 if not selected)."""
    needs_scores = True
    reset_selected = False

    def __init__(self, K, decay=0.1, head_dim=128, **kw):
        super().__init__(K, **kw)
        self.decay, self.tau = decay, math.sqrt(head_dim)
        self.f = {}

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        self.share = step_mass(self)
        for p in self.f:
            base = 0.0 if (self.reset_selected and p in self.share) else self.decay * self.f[p]
            self.f[p] = base + self.share.get(p, 0.0)

    def key(self, p):
        return self.f[p], self.last[p], self.hit[p], -p

    def dropped(self, p):
        super().dropped(p)
        del self.f[p]

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in misses:
            self.f[p] = self.share[p]


class InfLLMLruS(InfLLMEq1):
    """InfLLM public code, cache_strategy "lru-s": every resident decays (s <- d*s), then a selected
    page's score is replaced by this step's attention mass (its history is cleared)."""
    reset_selected = True


class SIEVE(Policy):
    """SIEVE (NSDI 2024): FIFO queue + visited bit; the hand moves from the oldest toward the newest,
    clears visited bits and evicts the first unvisited page. A step's hits set their bit before
    eviction; pages selected at this step are not candidates and the hand passes over them
    without clearing their bit."""
    def __init__(self, K, **kw):
        super().__init__(K, **kw)
        self.q = []          # oldest first
        self.visited = {}
        self.hand = 0

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        for p in ranked:
            if p in self.visited:
                self.visited[p] = True

    def evict(self, cands, n):
        out = []
        cands = set(cands)
        while len(out) < n:
            if self.hand >= len(self.q):
                self.hand = 0
            p = self.q[self.hand]
            if p not in cands:
                self.hand += 1
            elif self.visited[p]:
                self.visited[p] = False
                self.hand += 1
            else:
                out.append(p)
                cands.discard(p)
                del self.q[self.hand]      # the hand now points at the next newer page
                del self.visited[p]
        return out

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in sorted(misses):
            self.q.append(p)
            self.visited[p] = False


class S3FIFO(Policy):
    """S3-FIFO (SOSP 2023, Algorithm 1): small FIFO S (10% of B), main FIFO M, ghost FIFO G
    (B - |S target| entries), 2-bit frequency kept when a page moves from S to M. A step's hits raise
    their frequency before eviction; pages selected at this step are skipped (left in place) by the
    eviction scan, which takes the oldest unselected page of the queue."""
    needs_budget = True

    def __init__(self, K, B=None, small=0.1, **kw):
        super().__init__(K, **kw)
        self.cap = B
        self.s_cap = max(1, int(round(small * B)))
        self.S, self.M, self.G = OrderedDict(), OrderedDict(), OrderedDict()  # first = oldest
        self.freq = {}

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        for p in ranked:
            if p in self.freq:
                self.freq[p] = min(self.freq[p] + 1, 3)

    @staticmethod
    def _oldest(Q, cands):
        for p in Q:
            if p in cands:
                return p
        return None

    def _to_ghost(self, p):
        self.G[p] = None
        while len(self.G) > self.cap - self.s_cap:
            self.G.popitem(last=False)

    def evict(self, cands, n):
        cands = set(cands)
        out = []
        while len(out) < n:
            ps = self._oldest(self.S, cands) if len(self.S) >= self.s_cap else None
            if ps is not None:
                del self.S[ps]
                if self.freq[ps] > 1:
                    self.M[ps] = None
                else:
                    self._to_ghost(ps)
                    out.append(ps)
                    cands.discard(ps)
                    del self.freq[ps]
                continue
            pm = self._oldest(self.M, cands)
            if pm is None:  # only S holds candidates while S is below its target size
                pm = self._oldest(self.S, cands)
                del self.S[pm]
                self._to_ghost(pm)
                out.append(pm)
                cands.discard(pm)
                del self.freq[pm]
                continue
            if self.freq[pm] > 0:
                self.freq[pm] -= 1
                self.M.move_to_end(pm)
            else:
                del self.M[pm]
                out.append(pm)
                cands.discard(pm)
                del self.freq[pm]
        return out

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in sorted(misses):
            self.freq[p] = 0
            if p in self.G:
                del self.G[p]
                self.M[p] = None
            else:
                self.S[p] = None


class LazyEMA(ScoreEMA):
    """ScoreEMA refreshed only on steps with t % r == 0 (InertiaKV-style lazy update); newly
    admitted pages still start at their current value. alpha=1 is the frozen ranking (each page
    keeps the value it had when admitted)."""
    def __init__(self, K, alpha=0.5, r=1, **kw):
        super().__init__(K, alpha=alpha, **kw)
        self.r = r

    def begin(self, t, ranked, srow):
        if t % self.r == 0:
            super().begin(t, ranked, srow)
        else:
            Policy.begin(self, t, ranked, srow)
            self.rank_of = {p: r for r, p in enumerate(ranked, 1)}
            if self.signal == "allrank":
                from kvpolicy import _allrank_table
                self.table = _allrank_table(srow)


class Holt(Policy):
    """PRR-style Holt double exponential smoothing of the z signal (all residents every step):
    v <- beta*(x - l) + (1-beta)*v ; l <- alpha*x + (1-alpha)*l ; evict the lowest l + gamma*v.
    alpha weights the new observation (PRR convention; ScoreEMA's alpha weights the old value)."""
    needs_scores = True

    def __init__(self, K, alpha=0.5, beta=0.3, gamma=0.5, **kw):
        super().__init__(K, **kw)
        self.a, self.b, self.g = alpha, beta, gamma
        self.lv, self.tr = {}, {}

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        for p in self.lv:
            x = _z(srow, ranked, p)
            l0 = self.lv[p]
            self.tr[p] = self.b * (x - l0) + (1 - self.b) * self.tr[p]
            self.lv[p] = self.a * x + (1 - self.a) * l0

    def key(self, p):
        return self.lv[p] + self.g * self.tr[p], self.last[p], self.hit[p], -p

    def dropped(self, p):
        super().dropped(p)
        del self.lv[p], self.tr[p]

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in misses:
            self.lv[p] = _z(self.srow, self.ranked, p)
            self.tr[p] = 0.0


def powerlaw_mixture(gamma, horizon=1024, terms=10):
    """Non-negative weights w for sum_j w_j * r_j**a ~ (a+1)**-gamma over ages a = 0..horizon-1,
    with fixed rates r_j = 1 - 2**-j' (half-lives ~1, 2.4, 5.2, 11, 22, 44... steps: j' = 1, 2, ...)
    fitted by relative-error NNLS (active-set). Returns (rates, weights, max relative error)."""
    ages = np.arange(horizon, dtype=float)
    target = (ages + 1) ** -gamma
    rates = np.array([1 - 2.0 ** -(0.5 * j + 0.5) for j in range(2 * terms)])
    A = rates[None, :] ** ages[:, None] / target[:, None]
    b = np.ones(horizon)
    active = list(range(len(rates)))
    while True:
        w = np.zeros(len(rates))
        w[active] = np.linalg.lstsq(A[:, active], b, rcond=None)[0]
        neg = [j for j in active if w[j] < 0]
        if not neg:
            break
        active.remove(min(neg, key=lambda j: w[j]))
    err = float(np.max(np.abs(A @ w - b)))
    return rates, w, err


class PowerLawDecay(Policy):
    """Power-law age kernel instead of exponential decay (NoHairCache-style): the page value is the
    sum over steps since admission of z_tau * (t - tau + 1)**-gamma, computed with an exponential
    mixture (powerlaw_mixture) so that the state stays O(1) per resident."""
    needs_scores = True
    _cache = {}

    def __init__(self, K, gamma=1.0, **kw):
        super().__init__(K, **kw)
        if gamma not in self._cache:
            self._cache[gamma] = powerlaw_mixture(gamma)
        self.rates, self.w, self.err = self._cache[gamma]
        keep = [j for j in range(len(self.w)) if self.w[j] > 0]
        self.rates, self.w = [float(self.rates[j]) for j in keep], [float(self.w[j]) for j in keep]
        self.acc = {}

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        rates = self.rates
        for p, a in self.acc.items():
            x = _z(srow, ranked, p)
            for j, r in enumerate(rates):
                a[j] = r * a[j] + x

    def key(self, p):
        return sum(w * x for w, x in zip(self.w, self.acc[p])), self.last[p], self.hit[p], -p

    def dropped(self, p):
        super().dropped(p)
        del self.acc[p]

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in misses:
            self.acc[p] = [_z(self.srow, self.ranked, p)] * len(self.rates)


class CO2(ScoreEMA):
    """CO2-style factorial on top of ScoreEMA(z, all): a FIFO protection zone holding the
    floor(rho*B) most recently admitted residents, and a minimum residency of N steps after
    admission; the victim is the lowest-EMA page that is neither protected nor younger than N steps
    (if too few such candidates, the rest are the lowest-EMA remaining candidates)."""
    needs_budget = True

    def __init__(self, K, alpha=0.5, rho=0.0, N=0, B=None, **kw):
        super().__init__(K, alpha=alpha, **kw)
        self.rho, self.N, self.cap = rho, N, B
        self.adm = {}

    def evict(self, cands, n):
        cands = list(cands)
        nprot = int(self.rho * self.cap)
        prot = set()
        if nprot:
            prot = set(heapq.nlargest(nprot, self.adm, key=lambda p: (self.adm[p], p)))
        free = [p for p in cands if p not in prot and self.t - self.adm[p] >= self.N]
        out = heapq.nsmallest(n, free, key=self.key)
        if len(out) < n:
            chosen = set(out)
            out += heapq.nsmallest(n - len(out), [p for p in cands if p not in chosen], key=self.key)
        return out

    def dropped(self, p):
        super().dropped(p)
        del self.adm[p]

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in misses:
            self.adm[p] = self.t


V2_POLICIES = dict(random=RandomEvict, arkvale=ArkVale, infllm_eq1=InfLLMEq1, infllm_lrus=InfLLMLruS,
                   sieve=SIEVE, s3fifo=S3FIFO, lazy=LazyEMA, holt=Holt, powerlaw=PowerLawDecay, co2=CO2)
