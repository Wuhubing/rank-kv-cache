import numpy as np
import pytest

import kvpolicy as kp
from replay import replay_stream


def synthetic(T=120, P=60, K=8, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=P)
    ranked, scores = [], []
    for t in range(T):
        s = base + 0.7 * rng.normal(size=P)
        s[min(P, 20 + t // 3):] = np.nan  # pages become eligible over time
        valid = np.flatnonzero(~np.isnan(s))
        order = valid[np.argsort(-s[valid], kind="stable")][:K]
        ranked.append(order)
        scores.append(s)
    return np.array(ranked), np.array(scores, dtype=np.float32)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lru_and_belady_match_replay(seed):
    R, S = synthetic(seed=seed)
    steps = [tuple(int(p) for p in r) for r in R]
    for name in ("lru", "belady", "fifo"):
        ref = replay_stream(steps, 12, name)
        got = kp.simulate(R, S, 12, name)
        assert got["misses"] == ref["misses"] and got["accesses"] == ref["accesses"]


@pytest.mark.parametrize("delta,beta", [(0, 0), (0, 2), (4, 2), (8, 2)])
def test_rank_keeps_original_replay_tie_break(delta, beta):
    steps = [[6, 4, 0], [3, 1, 4], [3, 1, 6], [3, 0, 2], [6, 1, 0],
             [1, 2, 4], [0, 5, 6], [1, 4, 0], [5, 1, 3]]
    expected = replay_stream(steps, 4, "score", delta=delta, beta=beta)
    assert kp.simulate(steps, None, 4, "rank", delta=delta, beta=beta)["misses"] == expected["misses"]


def test_dev_auc_counts_equal_scores_as_half():
    from efficient_dev import auc

    assert auc([1, 1], [True, False]) == 0.5
    assert auc([1, 1], [False, True]) == 0.5
    assert np.isnan(auc([1, 1], [False, False]))


@pytest.mark.parametrize("seed", [0, 3])
def test_belady_is_lower_bound(seed):
    R, S = synthetic(seed=seed)
    opt = kp.simulate(R, S, 12, "belady")["misses"]
    for name in kp.POLICIES:
        if name == "belady":
            continue
        assert kp.simulate(R, S, 12, name)["misses"] >= opt, name
    assert kp.simulate(R, S, 12, "linear", coef=[1, -1, -1, 0, 1, 0, 0])["misses"] >= opt


def test_budget_equal_to_k_and_large():
    R, S = synthetic()
    for name in kp.POLICIES:
        a = kp.simulate(R, S, 8, name)
        b = kp.simulate(R, S, 60, name)
        assert b["misses"] == b["compulsory"]
        assert a["misses"] >= b["misses"]
    with pytest.raises(ValueError):
        kp.simulate(R, S, 7, "lru")


def test_score_policy_needs_scores():
    R, _ = synthetic()
    with pytest.raises(ValueError):
        kp.simulate(R, None, 12, "score")


def test_hybrid_zero_weight_is_lru():
    R, S = synthetic(seed=5)
    assert kp.simulate(R, S, 12, "hybrid", w=0)["misses"] == kp.simulate(R, S, 12, "lru")["misses"]


def test_allocate_respects_total_and_gains():
    grid = [8, 12, 16]
    curves = {0: {8: 100, 12: 50, 16: 40}, 1: {8: 10, 12: 9, 16: 9}}
    a = kp.allocate(curves, 2, 24, grid)
    assert a == {0: 16, 1: 8}
    assert sum(a.values()) <= 24


def test_bootstrap_interval_contains_point():
    lo, hi = kp.bootstrap([10, 12, 9, 11], [8, 9, 8, 9], [6, 6, 5, 7], reps=500)
    point = (42 - 34) / (42 - 24)
    assert lo <= point <= hi


def test_efficient_dev_smoke(tmp_path, monkeypatch):
    import sys
    import efficient_dev as ed
    rng = np.random.default_rng(0)
    for i in range(3):
        T, P, K, L = 60, 40, 8, 2
        R = np.zeros((T, L, K), np.int16)
        S = np.full((T, L, P), np.nan, np.float16)
        for l in range(L):
            base = rng.normal(size=P)
            for t in range(T):
                s = base + 0.6 * rng.normal(size=P)
                R[t, l] = np.argsort(-s)[:K]
                S[t, l] = s
        np.savez(tmp_path / f"d{i}.npz", ranked=R, scores=S, rid=f"t/d{i}")
    monkeypatch.setattr(ed, "GRID", [8, 10, 12, 16])
    monkeypatch.setattr(ed, "CURVE", [10, 12])
    monkeypatch.setattr(ed, "PRIMARY", 10)
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path), "--out", str(tmp_path / "o"), "--procs", "2"])
    # learned/bypass loops use fixed budgets 128/256; shrink them for the smoke test
    monkeypatch.setattr(ed, "LEARN_BUDGETS", (10, 12))
    ed.main()
    import csv as _csv
    rows = list(_csv.DictReader(open(tmp_path / "o" / "summary.csv")))
    assert any(r["policy"] == "belady" and abs(float(r["G"]) - 1) < 1e-9 for r in rows)
    assert any(r["policy"].startswith("linear-dagger") for r in rows)


def test_ema_variants_and_infllm_run_and_bound():
    R, S = synthetic(seed=7)
    opt = kp.simulate(R, S, 12, "belady")["misses"]
    for kw in (dict(mode="selected"), dict(signal="raw"), dict(signal="rank"),
               dict(signal="rank", mode="selected")):
        assert kp.simulate(R, S, 12, "ema", alpha=0.5, **kw)["misses"] >= opt
    assert kp.simulate(R, S, 12, "infllm", decay=0.9)["misses"] >= opt
    with pytest.raises(ValueError):
        kp.simulate(R, S, 12, "ema", mode="bogus")


def test_ema_alpha_zero_all_z_equals_current_score():
    # alpha=0 with every resident updated is the current normalised score, whose order equals the
    # current raw score order within a step
    R, S = synthetic(seed=8)
    a = kp.simulate(R, S, 12, "ema", alpha=0.0)["misses"]
    b = kp.simulate(R, S, 12, "score")["misses"]
    assert a == b


def test_lru_tie_variants_and_decayed_sum():
    R, S = synthetic(seed=9)
    opt = kp.simulate(R, S, 12, "belady")["misses"]
    assert kp.simulate(R, S, 12, "lru", tie="miss_first")["misses"] == kp.simulate(R, S, 12, "lru")["misses"]
    for tie in ("hit_first", "id"):
        assert kp.simulate(R, S, 12, "lru", tie=tie)["misses"] >= opt
    for v in ("softmax", "raw", "indicator"):
        for d in (0.0, 0.1, 0.9):
            assert kp.simulate(R, S, 12, "dsum", decay=d, v=v)["misses"] >= opt
    # d=0 with raw values is the current-score policy
    assert kp.simulate(R, S, 12, "dsum", decay=0.0, v="raw")["misses"] == kp.simulate(R, S, 12, "score")["misses"]


def test_allocate_dp_is_optimal_and_no_worse_than_greedy():
    import itertools
    rng = np.random.default_rng(1)
    grid = [8, 12, 16, 24]
    curves = {l: dict(zip(grid, sorted(rng.integers(0, 100, 4))[::-1])) for l in range(3)}
    total = 44
    dp = kp.allocate_dp(curves, 3, total, grid)
    brute = min((sum(curves[l][g] for l, g in enumerate(c)), c)
                for c in itertools.product(grid, repeat=3) if sum(c) <= total)
    assert sum(curves[l][dp[l]] for l in range(3)) == brute[0]
    gr = kp.allocate(curves, 3, total, grid)
    assert sum(curves[l][dp[l]] for l in range(3)) <= sum(curves[l][gr[l]] for l in range(3))
    pr = kp.allocate_proportional({0: 1, 1: 0, 2: 3}, 3, total, grid)
    assert sum(pr.values()) <= total and pr[2] >= pr[1]


def test_allrank_values_and_ties():
    t = kp._allrank_table([0.1, float("nan"), 0.5, 0.5, 0.9])
    assert list(t) == [0.1, 0.5, 0.5, 0.9]
    assert kp._allrank(t, 0.1) == 0.0
    assert kp._allrank(t, 0.9) == 1.0
    assert kp._allrank(t, 0.5) == 0.5  # two tied pages occupy ranks 1 and 2 of 0..3 -> mean 1.5 / 3
    assert kp._allrank(t, float("nan")) == 0.0
    assert kp._allrank(kp._allrank_table([2.0, float("nan")]), 2.0) == 1.0


def _ema_trajectory(R, S, B, **kw):
    """Misses plus the EMA value of every resident after every step."""
    traj = []

    class Spy(kp.ScoreEMA):
        def admit(self, hits, misses):
            super().admit(hits, misses)
            traj.append(dict(self.ema))
    kp.POLICIES["_spy"] = Spy
    try:
        c = kp.simulate(R, S, B, "_spy", **kw)
    finally:
        del kp.POLICIES["_spy"]
    return c["misses"], traj


MONOTONE = (lambda s: np.exp(2.0 * s), lambda s: 5.0 * s - 3.0, lambda s: s ** 3 + 0.1 * s)


@pytest.mark.parametrize("seed", [0, 4, 11])
def test_allrank_invariant_to_monotone_transforms(seed):
    R, S = synthetic(seed=seed)
    S = S.astype(np.float64)
    for alpha in (0.0, 0.5, 0.9):
        base = _ema_trajectory(R, S, 12, alpha=alpha, signal="allrank")
        for f in MONOTONE:
            got = _ema_trajectory(R, f(S), 12, alpha=alpha, signal="allrank")
            assert got[0] == base[0]
            assert len(got[1]) == len(base[1])
            for a, b in zip(got[1], base[1]):
                assert a.keys() == b.keys()
                assert all(abs(a[p] - b[p]) < 1e-12 for p in a)


def test_z_is_not_invariant_to_monotone_transforms():
    changed = False
    for seed in (0, 4, 11):
        R, S = synthetic(seed=seed)
        S = S.astype(np.float64)
        base = kp.simulate(R, S, 12, "ema", alpha=0.5, signal="z")["misses"]
        changed |= any(kp.simulate(R, f(S), 12, "ema", alpha=0.5, signal="z")["misses"] != base
                       for f in MONOTONE[:1] + MONOTONE[2:])
    assert changed


@pytest.mark.parametrize("seed", [1, 8])
def test_allrank_alpha_zero_equals_current_score(seed):
    # allrank is a strictly increasing function of the current score within a step, with equal
    # scores mapped to equal values, so with alpha=0 its eviction order is the raw-score order
    R, S = synthetic(seed=seed)
    for B in (8, 12, 20):
        assert (kp.simulate(R, S, B, "ema", alpha=0.0, signal="allrank")["misses"]
                == kp.simulate(R, S, B, "score")["misses"])


def test_allrank_differs_from_selected_rank_and_bounds():
    R, S = synthetic(seed=7)
    opt = kp.simulate(R, S, 12, "belady")["misses"]
    for mode in ("all", "selected"):
        assert kp.simulate(R, S, 12, "ema", alpha=0.5, signal="allrank", mode=mode)["misses"] >= opt
    # selected-rank gives every unselected page 0: its EMA among unselected pages carries no
    # ordering from the current step, unlike allrank
    _, rank_traj = _ema_trajectory(R, S, 12, alpha=0.0, signal="rank")
    sel = set(int(p) for p in R[-1])
    assert all(v == 0.0 for p, v in rank_traj[-1].items() if p not in sel)
    _, all_traj = _ema_trajectory(R, S, 12, alpha=0.0, signal="allrank")
    unsel = [v for p, v in all_traj[-1].items() if p not in sel]
    assert len(set(unsel)) > 1
