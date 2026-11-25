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


class S3FIFO_v2orig(Policy):
    """replication-v2 version, kept unchanged for side-by-side reporting (followup-v3 A5: it stops scanning S
    once S drops below its target size, so it differs from Algorithm 1 when the small queue holds > 1 page;
    use S3FIFO). S3-FIFO (SOSP 2023, Algorithm 1): small FIFO S (10% of B), main FIFO M, ghost FIFO G
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


class S3FIFO(Policy):
    """S3-FIFO, SOSP 2023 Algorithm 1 (followup-v3 A5 fix). Small FIFO S, main FIFO M, ghost FIFO G; 2-bit
    frequency (capped at 3), kept when a page moves from S to M.
      EVICT:  if |S| >= small*B: EVICTS else EVICTM
      EVICTS: repeat while nothing was evicted and S has a candidate: t = tail of S; if freq(t) > 1, move t to
              M and, if M is then full (|M| >= (1-small)*B), EVICTM; else move t to G (evicted).
      EVICTM: repeat while nothing was evicted: t = tail of M; if freq(t) > 0: freq -= 1, reinsert at the
              head; else evict t.
    G keeps at most (1-small)*B entries (as many as M holds). Batched adaptation (same as the other
    policies): a step's hits raise their frequency before eviction; pages selected at this step are not
    candidates, so "tail" means the oldest unselected page of the queue and selected pages stay in
    place. EVICT is repeated until at least n pages are gone; like Algorithm 1 a single EVICT can remove
    two pages (one from M inside EVICTS and one from S), so a step may free more than n slots. If
    EVICTM finds no candidate in M (all of M selected), the oldest unselected page of S is evicted."""
    needs_budget = True

    def __init__(self, K, B=None, small=0.1, **kw):
        super().__init__(K, **kw)
        self.cap, self.small = B, small
        self.s_target = small * B
        self.m_cap = (1 - small) * B
        self.S, self.M, self.G = OrderedDict(), OrderedDict(), OrderedDict()  # first = oldest (tail)
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

    def _gone(self, p, out, cands):
        out.append(p)
        cands.discard(p)
        del self.freq[p]

    def _evict_m(self, cands, out):
        while True:
            pm = self._oldest(self.M, cands)
            if pm is None:
                ps = self._oldest(self.S, cands)
                if ps is None:
                    return False
                del self.S[ps]
                self.G[ps] = None
                self._trim_ghost()
                self._gone(ps, out, cands)
                return True
            if self.freq[pm] > 0:
                self.freq[pm] -= 1
                self.M.move_to_end(pm)
            else:
                del self.M[pm]
                self._gone(pm, out, cands)
                return True

    def _trim_ghost(self):
        while len(self.G) > self.m_cap:
            self.G.popitem(last=False)

    def _evict_s(self, cands, out):
        while True:
            ps = self._oldest(self.S, cands)
            if ps is None:
                return False
            del self.S[ps]
            if self.freq[ps] > 1:
                self.M[ps] = None
                if len(self.M) >= self.m_cap:
                    self._evict_m(cands, out)
            else:
                self.G[ps] = None
                self._trim_ghost()
                self._gone(ps, out, cands)
                return True

    def evict(self, cands, n):
        cands = set(cands)
        out = []
        while len(out) < n:
            if len(self.S) >= self.s_target and self._evict_s(cands, out):
                continue
            if len(out) >= n:
                break
            if not self._evict_m(cands, out):
                raise RuntimeError("S3-FIFO: no eviction candidate")
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


class HoltStd(Policy):
    """Standard Holt linear trend (followup-v3 A2), for comparison with the PRR-style Holt above:
    l_t = alpha*x + (1-alpha)*(l + v) ; v_t = beta*(l_t - l) + (1-beta)*v ; evict the lowest l + gamma*v.
    The trend is the change of the level (Delta level) and the level includes the trend; alpha weights the
    new observation. Starts at l = x, v = 0."""
    needs_scores = True

    def __init__(self, K, alpha=0.5, beta=0.3, gamma=0.5, **kw):
        super().__init__(K, **kw)
        self.a, self.b, self.g = alpha, beta, gamma
        self.lv, self.tr = {}, {}

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        for p in self.lv:
            x = _z(srow, ranked, p)
            l0, v0 = self.lv[p], self.tr[p]
            l1 = self.a * x + (1 - self.a) * (l0 + v0)
            self.tr[p] = self.b * (l1 - l0) + (1 - self.b) * v0
            self.lv[p] = l1

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
    sum over steps since admission of x_tau * (t - tau + 1)**-gamma, computed with an exponential
    mixture (powerlaw_mixture) so that the state stays O(1) per resident.

    Optional variants (followup-v3 A1/B1; the defaults are the replication-v2 behaviour):
      normalize  False: the plain (unnormalised) sum. True: divided by the sum of the kernel weights over the
                 page's residency, sum_tau (t - tau + 1)**-gamma (same mixture), i.e. a weighted mean.
      zero       where x = 0 lies: "kth" = the K-th selected score (x = z, kvpolicy._z); "median" / "min" = the
                 median / minimum of all eligible pages' scores at this step; x = (s - ref) / (top1 - kth).
                 With the unnormalised sum the zero point decides whether residency adds or subtracts value:
                 most unselected pages have z < 0, so with zero="kth" a page loses value every step it stays.
      shift      constant added to x (e.g. +1 moves the zero point to one (top1 - kth) unit below the K-th)."""
    needs_scores = True
    _cache = {}
    ZEROS = ("kth", "median", "min")

    def __init__(self, K, gamma=1.0, normalize=False, zero="kth", shift=0.0, **kw):
        super().__init__(K, **kw)
        if zero not in self.ZEROS:
            raise ValueError("bad zero")
        if gamma not in self._cache:
            self._cache[gamma] = powerlaw_mixture(gamma)
        self.rates, self.w, self.err = self._cache[gamma]
        keep = [j for j in range(len(self.w)) if self.w[j] > 0]
        self.rates, self.w = [float(self.rates[j]) for j in keep], [float(self.w[j]) for j in keep]
        self.normalize, self.zero, self.shift = normalize, zero, shift
        self.acc, self.norm = {}, {}

    def x(self, p):
        if self.zero == "kth":
            return _z(self.srow, self.ranked, p) + self.shift
        kth, top = self.srow[self.ranked[-1]], self.srow[self.ranked[0]]
        return (self.srow[p] - self.ref) / (top - kth + 1e-6) + self.shift

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        if self.zero != "kth":
            a = np.asarray(srow, dtype=np.float64)
            a = a[np.isfinite(a)]
            self.ref = float(np.median(a)) if self.zero == "median" else float(a.min())
        rates = self.rates
        for p, a in self.acc.items():
            x = self.x(p)
            for j, r in enumerate(rates):
                a[j] = r * a[j] + x
            if self.normalize:
                nm = self.norm[p]
                for j, r in enumerate(rates):
                    nm[j] = r * nm[j] + 1.0

    def value(self, p):
        v = sum(w * x for w, x in zip(self.w, self.acc[p]))
        if self.normalize:
            v /= sum(w * x for w, x in zip(self.w, self.norm[p]))
        return v

    def key(self, p):
        return self.value(p), self.last[p], self.hit[p], -p

    def dropped(self, p):
        super().dropped(p)
        del self.acc[p]
        self.norm.pop(p, None)

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in misses:
            self.acc[p] = [self.x(p)] * len(self.rates)
            if self.normalize:
                self.norm[p] = [1.0] * len(self.rates)


def powerlaw_direct(xs, gamma):
    """Reference for tests: sum_i xs[i] * (n - i)**-gamma for the signal history xs (oldest first), i.e. the
    exact power-law kernel value that PowerLawDecay approximates with an exponential mixture."""
    n = len(xs)
    return float(sum(x * (n - i) ** -gamma for i, x in enumerate(xs)))


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
                   sieve=SIEVE, s3fifo=S3FIFO, s3fifo_v2orig=S3FIFO_v2orig, lazy=LazyEMA, holt=Holt,
                   holt_std=HoltStd, powerlaw=PowerLawDecay, co2=CO2)


# ------------------------------------------------------------------ ArkVale, full version
# Rules from the official code (pku-liang/ArkVale @ 0c3b45e, read-only copy at ~/work/rkv-arkvale/ArkVale);
# file:line references are to that commit and are listed rule by rule in docs/rerun/s4/09_arkvale_full.md.
#   * every decoding step the n_sink first pages and the n_win last pages (the last one is the page being
#     written) take part in attention; top-k runs over the pages in between
#     (arkvale_cpp/src/select.cu:99-107, kv_cache.py:163-173).
#   * a selected page that is not on the GPU is recalled into the slot of the resident, non-sink,
#     non-window page with the LOWEST current estimated score (select.cu:126-143, `!select_min`).
#   * when a new page starts and the cache is full, the page that leaves the window is dropped from the GPU
#     whatever its score (kv_cache.py:162-194: slot budget-n_win is overwritten, cc2gp[e_cci] = -1); it can
#     be recalled later. While the cache has room, pages are appended (kv_cache.py:143-161).
#   * the first n_unlimited_layers layers have no budget (infer_state.py:57-62; benchmark default 2).
#   * estimated score of a page = mean over query heads of sum_d max(q_d*bmax_d, q_d*bmin_d) with the
#     cuboid-mean digest bmax/min = c +- mean_k |c - k| (infer_state.py:448-462, estimate.cu:43,
#     estimate.cu:551-552 mean over heads with n_groups=1).
# In our page accounting the page being written (the tail) is outside the cache for every policy, so
# "n_win window pages" are n_win - 1 completed pages here.

def forced_pages(hp, sink, recent):
    """Pages that must be resident at a step with hp completed pages: the first `sink` pages and the
    `recent` most recent completed pages (ascending page ids)."""
    return sorted(set(range(min(sink, hp))) | set(range(max(0, hp - recent), hp)))


def completed_pages(scores):
    """[T] number of completed pages per step (scores are finite on a prefix, NaN beyond)."""
    S = np.asarray(scores)
    return np.isfinite(S).sum(1)


def with_forced(ranked, scores, sink, recent):
    """Access stream with the forced pages merged in: [T, K + sink + recent] int64, -1 = padding.

    Row order is selected[:-1], forced pages not already selected (ascending), selected[-1], so that
    row[0] and row[-1] are still the selector's top-1 and K-th page: the z signal of ScoreEMA and
    PowerLawDecay (kvpolicy._z) is computed exactly as on the plain stream, and every policy, LRU and
    Belady see the same access sets. Use this for the fairness controls (forced pages for everybody)."""
    R = np.asarray(ranked)
    hps = completed_pages(scores)
    T, K = R.shape
    out = np.full((T, K + sink + recent), -1, dtype=np.int64)
    for t in range(T):
        sel = [int(p) for p in R[t] if p >= 0]
        have = set(sel)
        extra = [p for p in forced_pages(int(hps[t]), sink, recent) if p not in have]
        row = sel[:-1] + extra + sel[-1:]
        out[t, :len(row)] = row
    return out


def simulate_arkvale(ranked, scores, B, *, est=None, sink=1, recent=1, window_exit=True):
    """ArkVale's eviction on one layer stream (rules above). ranked [T, K]: the selector's pages (forced
    pages are added here); scores [T, P]: the selector's scores (used for the completed-page count);
    est [T, P]: the estimated scores that order evictions (default: scores). window_exit drops the page
    that leaves the recent window as ArkVale does (False = keep it until evicted by score).
    Returns dict(accesses, misses, compulsory, window_exits) with the same meaning as kvpolicy.simulate."""
    import kvpolicy as kp
    R = kp.check_ranked(ranked)
    S = np.asarray(scores, dtype=np.float64)
    kp.check_scores(R, S)
    E = S if est is None else np.asarray(est, dtype=np.float64)
    if E.shape != S.shape:
        raise ValueError(f"est shape {E.shape} != scores shape {S.shape}")
    hps = completed_pages(S)
    res, seen, last, hit = set(), set(), {}, {}
    acc = miss = exits = 0
    prev_win = set()
    for t, row in enumerate(R):
        hp = int(hps[t])
        sel = [int(p) for p in row if p >= 0]
        sinks = set(range(min(sink, hp)))
        win = set(range(max(0, hp - recent), hp)) - sinks
        if window_exit and len(res) >= B:  # kv_cache.py:143 (room left: append) vs :162 (full: overwrite)
            for p in prev_win - win - sinks:
                if p in res:
                    res.discard(p)
                    del last[p], hit[p]
                    exits += 1
        access = set(sel) | sinks | win
        if len(access) > B:
            raise ValueError("budget below the access set")
        misses = access - res
        n = len(res) + len(misses) - B
        if n > 0:
            e = E[t]
            cands = res - access
            if any(not math.isfinite(e[p]) for p in cands):
                raise ValueError("non-finite estimated score for a resident page")
            for v in heapq.nsmallest(n, cands, key=lambda p: (e[p], last[p], hit[p], -p)):
                res.discard(v)
                del last[v], hit[v]
        res |= access
        for p in access:
            last[p] = t
            hit[p] = p not in misses
        acc += len(access)
        miss += len(misses)
        seen |= access
        prev_win = win
    return dict(accesses=acc, misses=miss, compulsory=len(seen), window_exits=exits)


def unlimited_budget(B, L, n_unlimited, pages):
    """Equal-total-slots budget of the limited layers when the first n_unlimited of L layers keep all
    `pages` completed pages: floor((L*B - n_unlimited*pages) / (L - n_unlimited)); may be < K
    (infeasible at that budget)."""
    return (L * B - n_unlimited * pages) // (L - n_unlimited)
