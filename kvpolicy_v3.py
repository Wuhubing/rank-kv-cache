"""followup-v3 (mechanism-v3) policies. Registered into kvpolicy.POLICIES on import; nothing in kvpolicy.py or
kvpolicy_v2.py changes. Same semantics as kvpolicy: every step's top-K must be resident; a policy only picks which
unselected residents to evict. Ties are broken by the LRU key (last, hit, -page) as everywhere else.

B1  Kernel        value = sum over the residency of k(age) * x_tau, kernel exponential (rate a) or power law
                  ((age+1)^-gamma, exponential mixture), optionally normalised by sum k(age); zero point of x
                  (kth / median / min) and a constant shift, as kvpolicy_v2.PowerLawDecay.
    EMAAge        key = EMA(z) - lam * log(1 + age), age = steps since admission ("resid") or since last selected
                  ("since_sel").
    EMADeficit    key = EMA(z) + lam * D, D = sum over the residency of min(z, 0) (lam=inf: D alone).
    EMAUnsel      key = EMA(z) - lam * log(1 + u), u = steps since admission on which the page was not selected.
    TwoScale      key = w * EMA_fast(z) + (1 - w) * EMA_slow(z) (two normalised EMAs).
B3  Lookahead     finite-horizon Belady: the next use is known only if it is within H steps; pages not used within
                  H are evicted first, ordered among themselves by LRU or by EMA(z); if more victims are needed,
                  the known-near pages go by farthest next use. H=None is Belady.
B7  ScopedEMA     EMA updated each step for the selected residents plus M unselected residents: the M with the
                  highest ("top") or lowest ("bottom") current score; the rest keep their stale values.
    RefRankEMA    EMA of the rank among the R highest-scoring eligible pages: (R - r + 1)/R if r <= R else 0
                  (R = K: rank among the selected; R = None: all eligible pages).
C3  ShrinkEMA     EMA(z) whose newly admitted page starts at a shrunken value: (1 - c) * x + c * prior, prior = the
                  running mean of the layer's admission values (or 0), or a count-weighted mean (c = k / (k + n)).
Also: simulate_steps (per-step miss counts, for the phase analysis), and score_transform helpers.
"""
import heapq
import math

import numpy as np

import kvpolicy as kp
from kvpolicy import Policy, ScoreEMA, _z
from kvpolicy_v2 import powerlaw_mixture

ZEROS = ("kth", "median", "min")


class _ZeroPoint:
    """x(p) = (s_p - ref) / (s_top1 - s_K) + shift with ref = the K-th selected score (kth), or the median /
    minimum of all eligible pages' scores at this step."""

    def set_ref(self, srow):
        if self.zero == "kth":
            return
        a = np.asarray(srow, dtype=np.float64)
        a = a[np.isfinite(a)]
        self.ref = float(np.median(a)) if self.zero == "median" else float(a.min())

    def x(self, p):
        if self.zero == "kth":
            return _z(self.srow, self.ranked, p) + self.shift
        kth, top = self.srow[self.ranked[-1]], self.srow[self.ranked[0]]
        return (self.srow[p] - self.ref) / (top - kth + 1e-6) + self.shift


class Kernel(_ZeroPoint, Policy):
    needs_scores = True
    _mix = {}

    def __init__(self, K, kernel="exp", a=0.5, gamma=1.0, normalize=False, zero="kth", shift=0.0, **kw):
        super().__init__(K, **kw)
        if kernel not in ("exp", "power") or zero not in ZEROS:
            raise ValueError("bad kernel or zero")
        if kernel == "exp":
            self.rates, self.w = [float(a)], [1.0]
        else:
            if gamma not in self._mix:
                r, w, _ = powerlaw_mixture(gamma)
                keep = w > 0
                self._mix[gamma] = ([float(x) for x in r[keep]], [float(x) for x in w[keep]])
            self.rates, self.w = self._mix[gamma]
        self.normalize, self.zero, self.shift = normalize, zero, shift
        self.acc, self.norm = {}, {}

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        self.set_ref(srow)
        for p, acc in self.acc.items():
            x = self.x(p)
            for j, r in enumerate(self.rates):
                acc[j] = r * acc[j] + x
            if self.normalize:
                nm = self.norm[p]
                for j, r in enumerate(self.rates):
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


class EMAAge(ScoreEMA):
    def __init__(self, K, alpha=0.5, lam=0.0, clock="resid", **kw):
        super().__init__(K, alpha=alpha, **kw)
        if clock not in ("resid", "since_sel"):
            raise ValueError("bad clock")
        self.lam, self.clock = lam, clock
        self.adm = {}

    def key(self, p):
        age = self.t - (self.adm[p] if self.clock == "resid" else self.last[p])
        return self.ema[p] - self.lam * math.log1p(age), self.last[p], self.hit[p], -p

    def dropped(self, p):
        super().dropped(p)
        del self.adm[p]

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in misses:
            self.adm[p] = self.t


class EMADeficit(ScoreEMA):
    def __init__(self, K, alpha=0.5, lam=1.0, **kw):
        super().__init__(K, alpha=alpha, **kw)
        self.lam = lam
        self.D = {}

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        for p in self.D:
            self.D[p] += min(_z(srow, ranked, p), 0.0)

    def key(self, p):
        if math.isinf(self.lam):
            return self.D[p], self.ema[p], self.last[p], self.hit[p], -p
        return self.ema[p] + self.lam * self.D[p], self.last[p], self.hit[p], -p

    def dropped(self, p):
        super().dropped(p)
        del self.D[p]

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in misses:
            self.D[p] = min(_z(self.srow, self.ranked, p), 0.0)


class EMAUnsel(ScoreEMA):
    def __init__(self, K, alpha=0.5, lam=0.0, **kw):
        super().__init__(K, alpha=alpha, **kw)
        self.lam = lam
        self.u = {}

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        sel = set(ranked)
        for p in self.u:
            if p not in sel:
                self.u[p] += 1

    def key(self, p):
        return self.ema[p] - self.lam * math.log1p(self.u[p]), self.last[p], self.hit[p], -p

    def dropped(self, p):
        super().dropped(p)
        del self.u[p]

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in misses:
            self.u[p] = 0


class TwoScale(Policy):
    needs_scores = True

    def __init__(self, K, fast=0.3, slow=0.9, w=0.5, **kw):
        super().__init__(K, **kw)
        self.fa, self.sa, self.w = fast, slow, w
        self.f, self.s = {}, {}

    def begin(self, t, ranked, srow):
        super().begin(t, ranked, srow)
        for p in self.f:
            x = _z(srow, ranked, p)
            self.f[p] = self.fa * self.f[p] + (1 - self.fa) * x
            self.s[p] = self.sa * self.s[p] + (1 - self.sa) * x

    def key(self, p):
        return self.w * self.f[p] + (1 - self.w) * self.s[p], self.last[p], self.hit[p], -p

    def dropped(self, p):
        super().dropped(p)
        del self.f[p], self.s[p]

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in misses:
            self.f[p] = self.s[p] = _z(self.srow, self.ranked, p)


class Lookahead(ScoreEMA):
    """tie: "lru" or "ema" orders the pages whose next use is beyond the horizon (or never)."""
    needs_future = True

    def __init__(self, K, H=None, tie="lru", alpha=0.5, future=None, **kw):
        super().__init__(K, alpha=alpha, **kw)
        if tie not in ("lru", "ema"):
            raise ValueError("bad tie")
        self.H, self.tie, self.future = H, tie, future

    def evict(self, cands, n):
        far, near = [], []
        for p in cands:
            f = self.future[p]
            nxt = f[0] if f else math.inf
            if self.H is not None and nxt - self.t > self.H:
                far.append(p)
            else:
                near.append((nxt, p))
        if self.tie == "lru":
            k = Policy.key.__get__(self)
        else:
            k = self.key
        out = heapq.nsmallest(n, far, key=k)
        if len(out) < n:
            near.sort(key=lambda x: (-x[0], x[1]))
            out += [p for _, p in near[:n - len(out)]]
        return out

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in self.ranked:
            self.future[p].popleft()


class ScopedEMA(ScoreEMA):
    def __init__(self, K, alpha=0.5, M=None, which="top", **kw):
        kw.pop("mode", None)
        super().__init__(K, alpha=alpha, mode="all", **kw)
        if which not in ("top", "bottom"):
            raise ValueError("bad which")
        self.M, self.which = M, which
        self.upd = {}   # step of the last update of each resident (for the staleness read-out)

    def begin(self, t, ranked, srow):
        Policy.begin(self, t, ranked, srow)
        self.rank_of = {p: r for r, p in enumerate(ranked, 1)}
        if self.signal == "allrank":
            self.table = kp._allrank_table(srow)
        sel = set(ranked)
        if self.M is None:
            pages = list(self.ema)
        else:
            uns = [p for p in self.ema if p not in sel]
            pick = (heapq.nlargest if self.which == "top" else heapq.nsmallest)(
                self.M, uns, key=lambda p: (srow[p], -p)) if self.M > 0 else []
            pages = [p for p in self.ema if p in sel] + pick
        a = self.alpha
        for p in pages:
            self.ema[p] = a * self.ema[p] + (1 - a) * self.value(p)
            self.upd[p] = t

    def dropped(self, p):
        super().dropped(p)
        self.upd.pop(p, None)

    def admit(self, hits, misses):
        super().admit(hits, misses)
        for p in misses:
            self.upd[p] = self.t


class RefRankEMA(ScoreEMA):
    def __init__(self, K, alpha=0.5, R=None, **kw):
        kw.pop("signal", None)
        super().__init__(K, alpha=alpha, signal="z", **kw)
        self.R = R

    def begin(self, t, ranked, srow):
        a = np.asarray(srow, dtype=np.float64)
        n = int(np.isfinite(a).sum())
        R = n if self.R is None else min(self.R, n)
        order = np.argsort(-np.where(np.isfinite(a), a, -np.inf), kind="stable")[:R]
        self.rr = {int(p): r for r, p in enumerate(order, 1)}
        self.RR = max(R, 1)
        super().begin(t, ranked, srow)

    def value(self, p):
        r = self.rr.get(p)
        return 0.0 if r is None else (self.RR - r + 1) / self.RR


class ShrinkEMA(ScoreEMA):
    """prior: "zero" (shrink toward 0, the K-th score) or "mean" (running mean of this layer's admission
    values); c: fixed weight, or "count" for c = k / (k + 1) with k = c_count (pseudo-count; a new page has one
    sample)."""

    def __init__(self, K, alpha=0.5, c=0.5, prior="mean", **kw):
        super().__init__(K, alpha=alpha, **kw)
        self.c, self.prior = c, prior
        self.sum, self.cnt = 0.0, 0

    def admit(self, hits, misses):
        Policy.admit(self, hits, misses)
        pri = 0.0 if self.prior == "zero" or not self.cnt else self.sum / self.cnt
        c = self.c
        for p in misses:
            x = self.value(p)
            self.ema[p] = (1 - c) * x + c * pri
            self.sum += x
            self.cnt += 1


V3_POLICIES = dict(kernel=Kernel, ema_age=EMAAge, ema_deficit=EMADeficit, ema_unsel=EMAUnsel, twoscale=TwoScale,
                   lookahead=Lookahead, scoped_ema=ScopedEMA, refrank_ema=RefRankEMA, shrink_ema=ShrinkEMA)
kp.POLICIES.update(V3_POLICIES)


def simulate_steps(ranked, scores, B, name, attn=None, **params):
    """kvpolicy.simulate with per-step counts: returns (counts dict, misses per step int32 [T])."""
    R = kp.check_ranked(ranked)
    if scores is not None:
        scores = np.asarray(scores)
        kp.check_scores(R, scores)
    rows = [tuple(int(p) for p in r if p >= 0) for r in R]
    K = max(len(r) for r in rows)
    cls = kp.POLICIES[name]
    if name == "belady" or getattr(cls, "needs_future", False):
        params["future"] = kp.future_uses(rows)
    if name == "arc" or getattr(cls, "needs_budget", False):
        params["B"] = B
    pol = cls(K, **params)
    res, seen = set(), set()
    per = np.zeros(len(rows), np.int32)
    acc = 0
    for t, row in enumerate(rows):
        srow = scores[t].tolist() if cls.needs_scores else None
        if attn is not None:
            pol.arow = [float(x) for x in attn[t][:len(row)]]
        pol.begin(t, row, srow)
        sel = set(row)
        hits = sel & res
        misses = sel - hits
        n = len(res) + len(misses) - B
        if n > 0:
            for v in pol.evict(res - sel, n):
                res.discard(v)
                pol.dropped(v)
        res |= sel
        pol.admit(hits, misses)
        acc += len(sel)
        per[t] = len(misses)
        seen |= sel
    return dict(accesses=acc, misses=int(per.sum()), compulsory=len(seen)), per
