"""ArkVale full version (kvpolicy_v2.simulate_arkvale, with_forced, unlimited_budget).

Each test is tied to a rule of the official code (pku-liang/ArkVale @ 0c3b45e); see
docs/rerun/s4/09_arkvale_full.md for the file:line table.
"""
import numpy as np
import pytest

import kvpolicy as kp
import kvpolicy_v2 as k2


def synthetic(seed=0, T=150, P0=30, K=8, drift=0.15, every=1):
    """Quest-like trace: one page completes every `every` steps; ranked = top-K by score; est = noisy
    second estimate of the same pages."""
    rng = np.random.default_rng(seed)
    P = P0 + T // every + 1
    base = rng.normal(0, 1, P)
    S = np.full((T, P), np.nan, np.float64)
    E = np.full((T, P), np.nan, np.float64)
    R = np.zeros((T, K), np.int64)
    for t in range(T):
        n = P0 + t // every
        base[:n] += rng.normal(0, drift, n)
        S[t, :n] = base[:n] + rng.normal(0, 0.3, n)
        E[t, :n] = S[t, :n] + rng.normal(0, 0.5, n)
        R[t] = np.argsort(-S[t, :n], kind="stable")[:K]
    return R, S, E


TRACES = [synthetic(s, every=e) for s in range(3) for e in (1, 3)]


def test_forced_pages():
    assert k2.forced_pages(10, 1, 1) == [0, 9]
    assert k2.forced_pages(10, 2, 3) == [0, 1, 7, 8, 9]
    assert k2.forced_pages(2, 2, 2) == [0, 1]       # overlap is merged
    assert k2.forced_pages(0, 1, 1) == []


@pytest.mark.parametrize("sink,recent", [(1, 1), (2, 2), (4, 2)])
def test_with_forced_keeps_selector_ends_and_adds_forced(sink, recent):
    for R, S, _ in TRACES:
        F = k2.with_forced(R, S, sink, recent)
        hps = k2.completed_pages(S)
        for t in range(len(R)):
            row = [p for p in F[t] if p >= 0]
            assert row[0] == R[t, 0] and row[-1] == R[t, -1]   # kvpolicy._z unchanged
            assert set(row) == set(R[t]) | set(k2.forced_pages(int(hps[t]), sink, recent))
            assert len(row) == len(set(row))
        kp.check_ranked(F)
        kp.check_scores(kp.check_ranked(F), S)


def test_with_forced_z_signal_identical():
    R, S, _ = TRACES[0]
    F = k2.with_forced(R, S, 1, 1)
    for t in range(len(R)):
        srow = S[t].tolist()
        for p in range(int(k2.completed_pages(S)[t])):
            assert kp._z(srow, list(F[t][F[t] >= 0]), p) == kp._z(srow, list(R[t]), p)


@pytest.mark.parametrize("B", [12, 16, 24])
@pytest.mark.parametrize("sink,recent", [(1, 1), (2, 2)])
def test_without_window_exit_equals_score_policy_on_forced_stream(B, sink, recent):
    """With window_exit off, ArkVale eviction = evict the lowest current estimate among unaccessed
    residents (select.cu:126-143) = kvpolicy ScoreCurrent on the forced stream with est as scores."""
    for R, S, E in TRACES:
        a = k2.simulate_arkvale(R, S, B, est=E, sink=sink, recent=recent, window_exit=False)
        F = k2.with_forced(R, S, sink, recent)
        b = kp.simulate(F, E, B, "score")
        assert (a["accesses"], a["misses"], a["compulsory"]) == (b["accesses"], b["misses"], b["compulsory"])
        assert a["window_exits"] == 0


@pytest.mark.parametrize("B", [12, 16, 24])
def test_counts_bounded_by_belady_on_same_stream(B):
    for R, S, E in TRACES:
        a = k2.simulate_arkvale(R, S, B, est=E, sink=1, recent=1)
        F = k2.with_forced(R, S, 1, 1)
        opt = kp.simulate(F, None, B, "belady")
        lru = kp.simulate(F, None, B, "lru")
        assert a["accesses"] == opt["accesses"] == lru["accesses"]
        assert a["compulsory"] == opt["compulsory"]
        assert opt["misses"] <= a["misses"]


def test_window_exit_drops_page_leaving_window_when_full():
    """kv_cache.py:162-194: page leaving the window is dropped (if the cache is full), then recalled
    (a miss) if it is selected at that step."""
    T, P0 = 4, 6
    S = np.full((T, P0 + T), np.nan)
    for t in range(T):
        S[t, :P0 + t] = np.arange(P0 + t, dtype=float)       # newest page has the highest score
    R = np.array([[2], [2], [2], [2]])
    # step 0: hp=6, access {0 sink, 5 recent, 2} -> 3 misses; B=3 is full
    # step 1: hp=7, page 5 leaves the window -> dropped; access {0, 6, 2}: 1 miss (page 6)
    # step 2: page 6 dropped, 7 enters -> 1 miss; step 3 likewise
    a = k2.simulate_arkvale(R, S, 3, sink=1, recent=1)
    assert a == dict(accesses=12, misses=6, compulsory=6, window_exits=3)
    # selecting the page that just left the window costs a recall
    R2 = np.array([[2], [5], [6], [7]])
    a2 = k2.simulate_arkvale(R2, S, 4, sink=1, recent=1)
    # step0 {0,5,2}=3 misses (res 3 < B=4, no exit at step 1); step1 {0,6,5}: 6 miss, res {0,5,2,6} full
    # step2: 6 leaves window, full -> dropped; access {0,7,6}: 7 and 6 miss -> victim lowest of {5,2}=2
    # step3: 7 leaves -> dropped; access {0,8,7}: 2 misses
    assert a2["window_exits"] == 2 and a2["misses"] == 3 + 1 + 2 + 2
    b2 = k2.simulate_arkvale(R2, S, 4, sink=1, recent=1, window_exit=False)
    assert b2["misses"] < a2["misses"]


def arkvale_port(S, E, B, sink, recent, k):
    """Slot-level port of ArkVale's decode bookkeeping with its own top-k selection, in our page
    accounting (the page being written is outside the cache; a newly completed page enters the window
    and counts as a miss, as in simulate_arkvale). Structure follows the code:
      sink / window / middle slots (select.cu:104-107 copies sink and window ids into the output);
      new page with a full cache -> the oldest window page is overwritten (kv_cache.py:162-194),
      otherwise appended and the old window page becomes a middle slot (kv_cache.py:143-161);
      top-k over the pages between sinks and window (select.cu:99-101);
      the n_recalls missing top-k pages replace the n_recalls lowest-scored middle slots, chosen all at
      once from the slot contents before the swap (select.cu:126-143)."""
    T = S.shape[0]
    hps = np.isfinite(S).sum(1)
    sinks, win, mid = [], [], []
    acc = miss = exits = 0
    for t in range(T):
        hp = int(hps[t])
        e = E[t]
        if t == 0:   # cold start, as for every replayed policy: the first access set is loaded
            sinks = list(range(min(sink, hp)))
            win = [p for p in range(max(0, hp - recent), hp) if p not in sinks]
        else:
            for p in range(int(hps[t - 1]), hp):
                full = len(sinks) + len(win) + len(mid) >= B
                win.append(p)
                miss += 1
                if len(win) > recent:
                    old = win.pop(0)
                    if full:   # cache was full: the old window slot is overwritten
                        exits += 1
                    else:
                        mid.append(old)
        pinned = set(sinks) | set(win)
        cand = [p for p in range(hp) if p not in pinned]
        top = sorted(cand, key=lambda p: -e[p])[:k]
        recalls = [p for p in top if p not in mid]
        room = B - len(sinks) - len(win) - len(mid)
        n_vict = max(0, len(recalls) - room)
        victims = sorted(mid, key=lambda p: e[p])[:n_vict]
        mid = [p for p in mid if p not in victims] + recalls
        assert len(sinks) + len(win) + len(mid) <= B
        if t == 0:
            miss += len(sinks) + len(win)
        miss += len(recalls)
        acc += len(set(sinks) | set(win) | set(top))
    return dict(accesses=acc, misses=miss, exits=exits)


@pytest.mark.parametrize("B", [10, 14, 20])
@pytest.mark.parametrize("sink,recent,k", [(1, 1, 5), (2, 2, 4)])
def test_matches_slot_level_port_of_official_code(B, sink, recent, k):
    for _, S, E in TRACES:
        hps = np.isfinite(S).sum(1)
        R = np.zeros((len(S), k), np.int64)
        for t in range(len(S)):
            pinned = set(k2.forced_pages(int(hps[t]), sink, recent))
            cand = [p for p in range(int(hps[t])) if p not in pinned]
            R[t] = sorted(cand, key=lambda p: -E[t, p])[:k]
        a = k2.simulate_arkvale(R, S, B, est=E, sink=sink, recent=recent)
        ref = arkvale_port(S, E, B, sink, recent, k)
        assert (a["accesses"], a["misses"], a["window_exits"]) == (ref["accesses"], ref["misses"], ref["exits"])


def test_budget_below_access_set_raises():
    R, S, E = TRACES[0]
    with pytest.raises(ValueError):
        k2.simulate_arkvale(R, S, 9, sink=1, recent=1)


def test_unlimited_budget():
    # 28 layers at B=256, 2 unlimited layers with 1500 pages: (7168 - 3000) // 26 = 160
    assert k2.unlimited_budget(256, 28, 2, 1500) == 160
    assert k2.unlimited_budget(96, 28, 2, 1500) < 64          # infeasible at K=64
    assert k2.unlimited_budget(100, 10, 0, 999) == 100
