"""Independent contracts and strict reproductions of defects at 24e5018.

Run: python -m pytest -q validation/test_contracts.py
The defects F1-F9 reproduced here were strict xfails at 24e5018; they are fixed in #5/#7,
so these are now ordinary regression tests.
"""
import itertools
import json
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import efficient_dev as ed
import checks
import heldout
import kvpolicy as kp

ROOT = Path(__file__).resolve().parents[1]


def trace(seed=0, steps=60, pages=24, width=5):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(steps, pages)).astype(np.float32)
    ranked = np.argsort(-scores, axis=1, kind="stable")[:, :width]
    return ranked, scores


def exact_minimum(rows, budget):
    """Enumerate legal cache states, independently of eviction heuristics."""
    @lru_cache(None)
    def solve(t, resident):
        if t == len(rows):
            return 0
        selected, resident = set(rows[t]), set(resident)
        cost = len(selected - resident)
        union = selected | resident
        if len(union) <= budget:
            return cost + solve(t + 1, tuple(sorted(union)))
        retain = budget - len(selected)
        return cost + min(solve(t + 1, tuple(sorted(selected | set(extra))))
                          for extra in itertools.combinations(sorted(union - selected), retain))
    return solve(0, ())


@pytest.mark.parametrize("budget", [2, 3, 4])
def test_belady_matches_enumerated_optimum(budget):
    choices = list(itertools.combinations(range(4), 2))
    for rows in itertools.product(choices, repeat=4):
        assert kp.simulate(rows, None, budget, "belady")["misses"] == exact_minimum(rows, budget)


@pytest.mark.parametrize("policy", [p for p in kp.POLICIES if p != "belady"])
def test_online_policies_cannot_observe_future(policy):
    ranked, scores = trace()
    def decisions(r, s):
        log = []
        kp.simulate(r, s, 8, policy, on_evict=lambda t, victims, _: log.append((t, tuple(victims))))
        return log
    assert decisions(ranked[:30], scores[:30]) == [x for x in decisions(ranked, scores) if x[0] < 30]


def scalar_ema(ranked, scores, budget, alpha):
    """Simple full-sort reference, with explicit protected sets and per-page state."""
    state, seen, decisions = {}, set(), []
    total = 0
    for t, row in enumerate(ranked.tolist()):
        values = [float(x) for x in scores[t]]
        z = [(x - values[row[-1]]) / (values[row[0]] - values[row[-1]] + 1e-6) for x in values]
        selected = set(row)
        hits = selected.intersection(state)
        missing = selected - hits
        for p in state:
            state[p][0] = alpha * state[p][0] + (1 - alpha) * z[p]
        n = max(0, len(state) + len(missing) - budget)
        if n:
            victims = sorted(set(state) - selected,
                             key=lambda p: (state[p][0], state[p][1], state[p][2], -p))[:n]
            decisions.append((t, tuple(victims)))
            for p in victims:
                del state[p]
        for p in row:
            state[p] = [state[p][0] if p in hits else z[p], t, p in hits]
        assert selected <= state.keys() and len(state) <= budget
        seen.update(row)
        total += len(missing)
    return total, decisions, len(seen)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.95])
def test_ema_matches_independent_state_reference(seed, alpha):
    ranked, scores = trace(seed)
    expected, evictions, compulsory = scalar_ema(ranked, scores, 8, alpha)
    actual = []
    c = kp.simulate(ranked, scores, 8, "ema", alpha=alpha,
                    on_evict=lambda t, victims, _: actual.append((t, tuple(victims))))
    assert (c["misses"], actual, c["compulsory"]) == (expected, evictions, compulsory)


def test_frozen_split_and_allocations_are_internally_consistent():
    split = json.loads((ROOT / "splits/longctx_v1.json").read_text())
    frozen = json.loads((ROOT / "splits/heldout_frozen_v1.json").read_text())
    dev = [p for p in split["prompts"] if p["split"] == "dev"]
    hold = [p for p in split["prompts"] if p["split"] == "heldout"]
    assert (len(dev), len(hold)) == (14, 21)
    assert not {p["rid"] for p in dev} & {p["rid"] for p in hold}
    assert not {p["ctx"] for p in dev} & {p["ctx"] for p in hold}
    assert set(frozen["dev_docs"]) == {p["rid"].replace("/", "__") + ".npz" for p in dev}
    for budget, allocation in frozen["alloc"].items():
        assert len(allocation) == 28 and min(allocation) >= 64
        assert sum(allocation) <= 28 * int(budget)


def test_frozen_alpha_controls_evaluation():
    frozen = {"alpha": {"96": 0.123}, "linear": {}}
    assert heldout.configs(96, frozen)["ema"][1]["alpha"] == 0.123


class InlinePool:
    """Run real evaluation jobs serially; avoids process setup for small fixtures."""
    def __init__(self, *args):
        pass

    def map(self, fn, jobs):
        return list(map(fn, jobs))


def test_renamed_development_trace_is_rejected(tmp_path, monkeypatch):
    split = json.loads((ROOT / "splits/longctx_v1.json").read_text())
    dev = next(p["rid"] for p in split["prompts"] if p["split"] == "dev")
    ranked, scores = trace(steps=2, pages=100, width=64)
    traces = tmp_path / "traces"
    traces.mkdir()
    np.savez(traces / "renamed.npz", ranked=ranked[:, None], scores=scores[:, None], rid=dev)
    monkeypatch.setattr(heldout, "CURVE", [96])
    monkeypatch.setattr(heldout, "Pool", InlinePool)
    args = SimpleNamespace(traces=traces, out=tmp_path / "out", procs=1,
                           frozen=ROOT / "splits/heldout_frozen_v1.json")
    with pytest.raises((ValueError, AssertionError), match="dev|held|split|document"):
        heldout.evaluate(args)


@pytest.mark.parametrize("labels", [[0, 0, 1, 1], [1, 1, 0, 0]])
def test_auc_ties_are_half_credit_and_order_invariant(labels):
    assert ed.auc(np.ones(4), labels) == 0.5


@pytest.mark.parametrize("ranked", [np.array([[0, 0]]), np.array([[0.5, 1.5]])])
def test_invalid_ranked_pages_are_rejected(ranked):
    with pytest.raises(ValueError):
        kp.simulate(ranked, np.array([[2.0, 1.0]]), 2, "ema")


def test_nonfinite_selected_score_is_rejected(tmp_path):
    path = tmp_path / "invalid.npz"
    np.savez(path, ranked=np.array([[[0, 1]]]), scores=np.array([[[np.nan, 1.0]]]), rid="d")
    with pytest.raises(ValueError):
        kp.load(path)


def test_allocator_rejects_infeasible_budget():
    curves = {0: {8: 10, 16: 1}, 1: {8: 10, 16: 1}}
    with pytest.raises(ValueError):
        kp.allocate(curves, 2, 8, [8, 16])


def test_zero_weights_produce_feasible_allocation():
    allocation = kp.allocate_proportional({0: 0, 1: 0}, 2, 24, [8, 12, 16])
    assert len(allocation) == 2 and sum(allocation.values()) <= 24


def test_zero_weight_rank_policy_preserves_lru_baseline():
    ranked = np.array([[0, 2], [1, 2], [3, 0], [2, 3]])
    assert kp.simulate(ranked, None, 3, "rank", delta=0, beta=0) == kp.simulate(ranked, None, 3, "lru")


def test_partial_check_resume_schedules_missing_document(tmp_path, monkeypatch):
    traces, result = tmp_path / "traces", tmp_path / "result"
    traces.mkdir()
    result.mkdir()
    for doc in ("a", "b"):
        (traces / f"{doc}.npz").touch()
    header = "doc,layer,config,B,accesses,misses,compulsory\n"
    (result / "misses.csv").write_text(header + "a,0,lru,96,100,20,10\nb,0,lru,96,100,20,10\n")
    (result / "misses_ext.csv").write_text(header + "a,0,ema:0.4,96,100,18,10\n")
    submitted = []
    class Pool:
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def imap_unordered(self, fn, jobs):
            submitted.extend(jobs)
            return iter([])
    class StopAfterScheduling(Exception):
        pass
    def stop(_):
        raise StopAfterScheduling
    monkeypatch.setattr(checks, "Pool", Pool)
    monkeypatch.setattr(checks, "EXTRA", {"ema:0.4": ("ema", {"alpha": 0.4})})
    monkeypatch.setattr(checks, "doc_meta", stop)
    monkeypatch.setattr(sys, "argv", ["checks", str(traces), "--res", str(result), "--budgets", "96"])
    with pytest.raises(StopAfterScheduling):
        checks.main()
    assert any(Path(path).stem == "b" and 96 in budgets for path, _, budgets in submitted)


def test_uniform_belady_is_not_global_allocation_lower_bound():
    busy = np.array([[p] for p in (0, 1, 2) * 6])
    stable = np.zeros((18, 1), dtype=int)
    uniform_oracle = sum(kp.simulate(r, None, 2, "belady")["misses"] for r in (busy, stable))
    allocated_lru = kp.simulate(busy, None, 3, "lru")["misses"] + kp.simulate(stable, None, 1, "lru")["misses"]
    assert allocated_lru < uniform_oracle  # Same total of four slots; different allocation.
