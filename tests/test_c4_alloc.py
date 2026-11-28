"""C4 simulators on synthetic multi-layer traces."""
import numpy as np

import kvpolicy as kp
from analysis_v3 import c4_alloc as c4
from test_kvpolicy_v2 import synthetic


def trace(L=3, seed=0):
    layers = [synthetic(seed=seed * 10 + l, T=80, P0=60, K=8) for l in range(L)]
    R = np.stack([x[0] for x in layers], 1)
    S = np.stack([x[1] for x in layers], 1).astype(np.float32)
    return R, S


def test_online_without_moves_is_per_layer_ema():
    R, S = trace()
    m, budget = c4.online_run(R, S, 12, 0.5, N=10 ** 6, s=2, decay=0.0)
    ref = sum(kp.simulate(R[:, l], S[:, l], 12, "ema", alpha=0.5)["misses"] for l in range(3))
    assert m == ref and budget == [12, 12, 12]


def test_online_moves_keep_total_and_floor():
    R, S = trace(seed=1)
    m, budget = c4.online_run(R, S, 12, 0.5, N=4, s=2, decay=0.5)
    assert sum(budget) == 36 and min(budget) >= 8


def test_pool_bounds():
    R, S = trace(seed=2)
    T, L, K = R.shape
    comp = sum(len({int(p) for p in R[:, l].ravel()}) for l in range(L))
    m_big, acc = c4.pool_run(R, S, 10 ** 6, "lru")
    assert m_big == comp and acc == T * L * K
    bel, _ = c4.pool_run(R, S, 36, "belady")
    lru, _ = c4.pool_run(R, S, 36, "lru")
    ema, _ = c4.pool_run(R, S, 36, "ema", signal="allrank", beta=0.5)
    assert bel <= lru and bel <= ema


def test_pool_belady_single_layer_matches_belady():
    R, S = trace(L=1, seed=3)
    m, _ = c4.pool_run(R, S, 12, "belady")
    assert m == kp.simulate(R[:, 0], None, 12, "belady")["misses"]
