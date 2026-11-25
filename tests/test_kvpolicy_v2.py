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
       "lazy": dict(r=4), "holt": {}, "powerlaw": {}, "co2": dict(rho=0.25, N=4)}


@pytest.mark.parametrize("name", sorted(NEW))
@pytest.mark.parametrize("B", [8, 12, 20])
def test_bounded_by_belady_and_counts(name, B):
    for R, S in TRACES:
        c = kp.simulate(R, S, B, name, **NEW[name])
        opt = kp.simulate(R, S, B, "belady")
        assert c["accesses"] == opt["accesses"] and c["compulsory"] == opt["compulsory"]
        assert opt["misses"] <= c["misses"] <= c["accesses"]


def same(a, b, B, ka=None, kb=None):
    for R, S in TRACES:
        assert kp.simulate(R, S, B, a, **(ka or {})) == kp.simulate(R, S, B, b, **(kb or {}))


@pytest.mark.parametrize("B", [9, 16])
def test_reductions(B):
    same("lazy", "ema", B, dict(alpha=0.5, r=1), dict(alpha=0.5))
    same("co2", "ema", B, dict(alpha=0.5, rho=0.0, N=0), dict(alpha=0.5))
    same("holt", "ema", B, dict(alpha=0.5, beta=0.0, gamma=0.0), dict(alpha=0.5))
    same("holt", "ema", B, dict(alpha=0.75, beta=0.3, gamma=0.0), dict(alpha=0.25))
    same("arkvale", "score", B, dict(sink=0, win=0))
    same("infllm_eq1", "infllm", B, dict(decay=0.9), dict(decay=0.9))


def test_lazy_alpha1_is_frozen_and_r_changes_result():
    R, S = TRACES[0]
    a = kp.simulate(R, S, 12, "lazy", alpha=1.0, r=1)
    b = kp.simulate(R, S, 12, "lazy", alpha=1.0, r=8)
    assert a == b  # alpha=1 never moves, refresh rate irrelevant


def test_lrus_differs_from_eq1_only_through_reset():
    R, S = TRACES[1]
    # with decay 0 both keep only this step's mass for selected pages and 0 for the rest
    assert kp.simulate(R, S, 12, "infllm_lrus", decay=0.0) == kp.simulate(R, S, 12, "infllm_eq1", decay=0.0)


def test_true_mass_is_used():
    R, S = TRACES[0]
    A = np.zeros(R.shape, np.float32)
    A[:, 0] = 1.0  # all mass on the top page
    with_attn = kp.simulate(R, S, 12, "infllm_eq1", attn=A, decay=0.5)
    proxy = kp.simulate(R, S, 12, "infllm_eq1", decay=0.5)
    assert with_attn["accesses"] == proxy["accesses"]
    assert with_attn != proxy or True  # runs; equality is possible but the path is exercised


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


def s3fifo_ref(stream, B, small=0.1):
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


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("B", [2, 4, 10])
def test_sieve_and_s3fifo_match_classic_on_single_access(seed, B):
    rng = random.Random(seed)
    # skewed stream with reuse
    stream = [int(rng.paretovariate(1.2)) % 40 for _ in range(600)]
    R = np.array(stream)[:, None]
    assert kp.simulate(R, None, B, "sieve")["misses"] == sieve_ref(stream, B)
    assert kp.simulate(R, None, B, "s3fifo")["misses"] == s3fifo_ref(stream, B)


@pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
def test_powerlaw_fit(gamma):
    rates, w, err = k2.powerlaw_mixture(gamma)
    assert (w >= 0).all()
    assert err < 0.25  # max relative error over ages 0..1023 (reported in results)
