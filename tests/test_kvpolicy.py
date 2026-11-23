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
