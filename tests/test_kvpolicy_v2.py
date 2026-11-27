"""replication-v2 baselines: invariants, reductions to existing policies, classic references."""
import random
from collections import OrderedDict

import numpy as np
import pytest

import kvpolicy as kp
import kvpolicy_v2 as k2


def synthetic(seed=0, T=120, P0=30, K=8, drift=0.15):
    """Quest-like trace: scores follow a slow random walk; ranked = top-K by score."""
    rng = np.random.default_rng(seed)
    P = P0 + T
    base = rng.normal(0, 1, P)
    S = np.full((T, P), np.nan, np.float32)
    R = np.zeros((T, K), np.int64)
    for t in range(T):
        n = P0 + t
        base[:n] += rng.normal(0, drift, n)
        S[t, :n] = base[:n] + rng.normal(0, 0.3, n)
        R[t] = np.argsort(-S[t, :n], kind="stable")[:K]
    return R, S


TRACES = [synthetic(s) for s in range(3)]
NEW = {"random": {}, "arkvale": {}, "infllm_eq1": {}, "infllm_lrus": {}, "sieve": {}, "s3fifo": {},
       "s3fifo_v2orig": {}, "lazy": dict(r=4), "holt": {}, "holt_std": {}, "powerlaw": {},
       "co2": dict(rho=0.25, N=4)}
VARIANTS = [dict(normalize=True), dict(zero="median"), dict(zero="min"), dict(shift=1.0),
            dict(normalize=True, zero="min"), dict(gamma=0.5, normalize=True)]


@pytest.mark.parametrize("name", sorted(NEW))
@pytest.mark.parametrize("B", [8, 12, 20])
def test_bounded_by_belady_and_counts(name, B):
    for R, S in TRACES:
        c = kp.simulate(R, S, B, name, **NEW[name])
        opt = kp.simulate(R, S, B, "belady")
        assert c["accesses"] == opt["accesses"] and c["compulsory"] == opt["compulsory"]
        assert opt["misses"] <= c["misses"] <= c["accesses"]


@pytest.mark.parametrize("kw", VARIANTS)
@pytest.mark.parametrize("B", [8, 20])
def test_powerlaw_variants_bounded(kw, B):
    for R, S in TRACES:
        c = kp.simulate(R, S, B, "powerlaw", **kw)
        opt = kp.simulate(R, S, B, "belady")
        assert c["accesses"] == opt["accesses"] and opt["misses"] <= c["misses"]


def same(a, b, B, ka=None, kb=None):
    for R, S in TRACES:
        assert kp.simulate(R, S, B, a, **(ka or {})) == kp.simulate(R, S, B, b, **(kb or {}))


@pytest.mark.parametrize("B", [9, 16])
def test_reductions(B):
    same("lazy", "ema", B, dict(alpha=0.5, r=1), dict(alpha=0.5))
    same("co2", "ema", B, dict(alpha=0.5, rho=0.0, N=0), dict(alpha=0.5))
    same("holt", "ema", B, dict(alpha=0.5, beta=0.0, gamma=0.0), dict(alpha=0.5))
    same("holt", "ema", B, dict(alpha=0.75, beta=0.3, gamma=0.0), dict(alpha=0.25))
    same("holt_std", "ema", B, dict(alpha=0.5, beta=0.0, gamma=0.0), dict(alpha=0.5))
    same("powerlaw", "powerlaw", B, dict(gamma=1.0), dict(gamma=1.0, normalize=False, zero="kth", shift=0.0))
    same("arkvale", "score", B, dict(sink=0, win=0))
    same("infllm_eq1", "infllm", B, dict(decay=0.9), dict(decay=0.9))


def test_lazy_alpha1_is_frozen_and_r_changes_result():
    R, S = TRACES[0]
    a = kp.simulate(R, S, 12, "lazy", alpha=1.0, r=1)
    b = kp.simulate(R, S, 12, "lazy", alpha=1.0, r=8)
    assert a == b  # alpha=1 never moves, refresh rate irrelevant
    # with alpha < 1 the refresh rate matters: r=1 is the EMA, a slower refresh gives different counts
    for R, S in TRACES:
        res = {r: kp.simulate(R, S, 12, "lazy", alpha=0.5, r=r)["misses"] for r in (1, 2, 8)}
        assert len(set(res.values())) > 1, res
        assert res[1] == kp.simulate(R, S, 12, "ema", alpha=0.5)["misses"]


def test_lazy_skips_updates_between_refreshes():
    R, S = TRACES[0]
    vals = []

    def cb(t, victims, pol):
        vals.append((t, dict(pol.ema)))
    kp.simulate(R, S, 12, "lazy", alpha=0.5, r=4, on_evict=cb)
    # between refresh steps the value of a page that stays resident does not change
    seen = 0
    for (t0, e0), (t1, e1) in zip(vals, vals[1:]):
        if t1 == t0 + 1 and t1 % 4 != 0:
            common = set(e0) & set(e1)
            assert all(e0[p] == e1[p] for p in common)
            seen += len(common)
    assert seen > 0


def test_lrus_differs_from_eq1_only_through_reset():
    R, S = TRACES[1]
    # with decay 0 both keep only this step's mass for selected pages and 0 for the rest
    assert kp.simulate(R, S, 12, "infllm_lrus", decay=0.0) == kp.simulate(R, S, 12, "infllm_eq1", decay=0.0)


def test_true_mass_is_used():
    # all mass on the K-th (last) selected page: the opposite of the softmax proxy, so results differ
    diffs = 0
    for R, S in TRACES:
        A = np.zeros(R.shape, np.float32)
        A[:, -1] = 1.0
        for B in (10, 12, 16):
            with_attn = kp.simulate(R, S, B, "infllm_eq1", attn=A, decay=0.5)
            proxy = kp.simulate(R, S, B, "infllm_eq1", decay=0.5)
            assert with_attn["accesses"] == proxy["accesses"]
            diffs += with_attn["misses"] != proxy["misses"]
    assert diffs > 0


def test_step_mass_reads_the_trace_row():
    R, S = TRACES[0]
    seen = []

    def cb(t, victims, pol):
        seen.append((list(pol.ranked), dict(pol.share)))
    A = np.tile(np.arange(R.shape[1], dtype=np.float32), (R.shape[0], 1))
    kp.simulate(R, S, 10, "infllm_eq1", attn=A, decay=0.5, on_evict=cb)
    assert seen
    for ranked, share in seen:
        assert share == {p: float(i) for i, p in enumerate(ranked)}


def test_holt_std_recursion():
    """Standard Holt: the level includes the trend and the trend is the change of the level."""
    R, S = TRACES[1]
    a, b = 0.4, 0.3
    track = {}

    class Rec(k2.HoltStd):
        def begin(self, t, ranked, srow):
            before = {p: (self.lv[p], self.tr[p]) for p in self.lv}
            super().begin(t, ranked, srow)
            for p, (l0, v0) in before.items():
                x = kp._z(srow, ranked, p)
                l1 = a * x + (1 - a) * (l0 + v0)
                assert self.lv[p] == pytest.approx(l1)
                assert self.tr[p] == pytest.approx(b * (l1 - l0) + (1 - b) * v0)
                track[p] = True

    kp.POLICIES["_rec"] = Rec
    try:
        kp.simulate(R, S, 12, "_rec", alpha=a, beta=b, gamma=0.5)
    finally:
        del kp.POLICIES["_rec"]
    assert track
    # unlike the PRR-style Holt, the trend enters the level, so gamma=0 is not the EMA when beta > 0
    assert any(kp.simulate(R2, S2, 12, "holt_std", alpha=0.5, beta=0.5, gamma=0.0) !=
               kp.simulate(R2, S2, 12, "ema", alpha=0.5) for R2, S2 in TRACES)


def test_random_seeds_reproducible():
    R, S = TRACES[2]
    assert kp.simulate(R, S, 12, "random", seed=3) == kp.simulate(R, S, 12, "random", seed=3)


def test_arkvale_never_evicts_pinned_when_others_available():
    R, S = TRACES[0]
    seen = []

    def cb(t, victims, pol):
        cands_unpinned = True
        for v in victims:
            seen.append(v in pol.pinned)
    kp.simulate(R, S, 16, "arkvale", on_evict=cb, sink=2, win=2)
    assert seen and not any(seen)


# ---- classic references for single-access streams (K = 1): our batched adaptation must agree ----

def sieve_ref(stream, B):
    q, vis, hand, miss = [], {}, 0, 0
    for x in stream:
        if x in vis:
            vis[x] = True
            continue
        miss += 1
        if len(q) == B:
            while True:
                if hand >= len(q):
                    hand = 0
                y = q[hand]
                if vis[y]:
                    vis[y] = False
                    hand += 1
                else:
                    del q[hand]
                    del vis[y]
                    break
        q.append(x)
        vis[x] = False
    return miss


def s3fifo_ref_v2orig(stream, B, small=0.1):
    s_cap = max(1, int(round(small * B)))
    S, M, G, f = OrderedDict(), OrderedDict(), OrderedDict(), {}
    miss = 0

    def ghost(x):
        G[x] = None
        while len(G) > B - s_cap:
            G.popitem(last=False)

    def evict_m():
        while True:
            t = next(iter(M))
            if f[t] > 0:
                f[t] -= 1
                M.move_to_end(t)
            else:
                del M[t], f[t]
                return

    def evict():
        if len(S) >= s_cap:
            while S:
                t = next(iter(S))
                del S[t]
                if f[t] > 1:
                    M[t] = None
                else:
                    ghost(t)
                    del f[t]
                    return
            evict_m()
        else:
            evict_m()

    for x in stream:
        if x in f:
            f[x] = min(f[x] + 1, 3)
            continue
        miss += 1
        while len(S) + len(M) >= B:
            evict()
        f[x] = 0
        if x in G:
            del G[x]
            M[x] = None
        else:
            S[x] = None
    return miss


def s3fifo_ref(stream, B, small=0.1):
    """S3-FIFO, SOSP 2023 Algorithm 1, written line by line for a single-access stream."""
    S, M, G, f = OrderedDict(), OrderedDict(), OrderedDict(), {}
    miss = 0

    def evict_m():
        evicted = False
        while not evicted and len(M) > 0:
            t = next(iter(M))            # tail = oldest
            if f[t] > 0:
                M.move_to_end(t)         # insert t to head of M
                f[t] -= 1
            else:
                del M[t], f[t]
                evicted = True

    def evict_s():
        evicted = False
        while not evicted and len(S) > 0:
            t = next(iter(S))
            if f[t] > 1:
                M[t] = None
                if len(M) >= (1 - small) * B:
                    evict_m()
            else:
                G[t] = None
                while len(G) > (1 - small) * B:
                    G.popitem(last=False)
                del f[t]
                evicted = True
            del S[t]

    def evict():
        if len(S) >= small * B:
            evict_s()
        else:
            evict_m()

    for x in stream:
        if x in S or x in M:
            f[x] = min(f[x] + 1, 3)
            continue
        miss += 1
        while len(S) + len(M) >= B:
            evict()
        if x in G:
            del G[x]
            M[x] = None
        else:
            S[x] = None
        f[x] = 0
    return miss


def skewed(seed, n=600, m=40):
    rng = random.Random(seed)
    return [int(rng.paretovariate(1.2)) % m for _ in range(n)]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("B", [2, 4, 10, 20, 50])
def test_sieve_and_s3fifo_match_classic_on_single_access(seed, B):
    stream = skewed(seed, n=1500, m=120)  # skewed stream with reuse
    R = np.array(stream)[:, None]
    assert kp.simulate(R, None, B, "sieve")["misses"] == sieve_ref(stream, B)
    assert kp.simulate(R, None, B, "s3fifo")["misses"] == s3fifo_ref(stream, B)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("B", [2, 4, 10])
def test_s3fifo_v2orig_matches_its_reference_at_small_budgets(seed, B):
    stream = skewed(seed)
    R = np.array(stream)[:, None]
    assert kp.simulate(R, None, B, "s3fifo_v2orig")["misses"] == s3fifo_ref_v2orig(stream, B)


def test_s3fifo_v2orig_departs_from_algorithm_1_with_a_larger_small_queue():
    diffs = 0
    for seed in range(40):
        stream = skewed(seed, n=3000, m=60)
        R = np.array(stream)[:, None]
        diffs += kp.simulate(R, None, 20, "s3fifo_v2orig")["misses"] != s3fifo_ref(stream, 20)
        assert kp.simulate(R, None, 20, "s3fifo")["misses"] == s3fifo_ref(stream, 20)
    assert diffs > 0


def _history(R, S, gamma, **kw):
    """Run PowerLawDecay and record, for every resident at every eviction, the mixture value and the signal
    history since admission."""
    hist, out = {}, []

    def cb(t, victims, pol):
        for p in pol.acc:
            out.append((pol.value(p), list(hist[p]), pol))

    class Rec(k2.PowerLawDecay):
        def begin(self, t, ranked, srow):
            super().begin(t, ranked, srow)
            for p in self.acc:
                hist[p].append(self.x(p))

        def admit(self, hits, misses):
            super().admit(hits, misses)
            for p in misses:
                hist[p] = [self.x(p)]

    kp.POLICIES["_rec"] = Rec
    try:
        kp.simulate(R, S, 12, "_rec", on_evict=cb, gamma=gamma, **kw)
    finally:
        del kp.POLICIES["_rec"]
    return out


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("kw", [{}, dict(zero="min"), dict(normalize=True)])
def test_powerlaw_matches_direct_sum(gamma, kw):
    R, S = TRACES[0]
    rows = _history(R, S, gamma, **kw)
    assert rows
    for v, xs, pol in rows:
        exact = k2.powerlaw_direct(xs, gamma)
        bound = pol.err * k2.powerlaw_direct([abs(x) for x in xs], gamma) + 1e-9
        if kw.get("normalize"):
            wsum = k2.powerlaw_direct([1.0] * len(xs), gamma)
            exact /= wsum
            # numerator and denominator each within err (relative): |v - exact| <= 2 err/(1-err) * mean|x|
            bound = 2 * pol.err / (1 - pol.err) * k2.powerlaw_direct([abs(x) for x in xs], gamma) / wsum + 1e-9
        assert abs(v - exact) <= bound


def test_powerlaw_direct_reference():
    assert k2.powerlaw_direct([1.0, 2.0, 3.0], 1.0) == pytest.approx(1 / 3 + 2 / 2 + 3 / 1)


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
def test_powerlaw_fit(gamma):
    rates, w, err = k2.powerlaw_mixture(gamma)
    assert (w >= 0).all()
    assert err < 0.25  # max relative error over ages 0..1023 (reported in results)
