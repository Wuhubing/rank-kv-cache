"""Regressions for the review findings in PR #4 (validation/REPORT.md, F1-F9)."""
import numpy as np
import pytest

import kvpolicy as kp


# F6: trace schema is validated at the ingestion boundary.
@pytest.mark.parametrize("ranked", [np.array([[0, 0]]), np.array([[0.5, 1.5]]),
                                    np.array([[0, 5]]), np.array([[-2, 1]]), np.array([0, 1])])
def test_invalid_ranked_pages_are_rejected(ranked):
    with pytest.raises(ValueError):
        kp.simulate(ranked, np.array([[2.0, 1.0]]), 2, "ema")


def test_padding_and_valid_float_ids_are_accepted():
    S = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    a = kp.simulate(np.array([[0, 1], [2, -1]]), S, 2, "ema")
    b = kp.simulate(np.array([[0.0, 1.0], [2.0, -1.0]]), S, 2, "ema")
    assert a == b and a["misses"] == 3


def test_nonfinite_selected_score_is_rejected(tmp_path):
    path = tmp_path / "invalid.npz"
    np.savez(path, ranked=np.array([[[0, 1]]]), scores=np.array([[[np.nan, 1.0]]]), rid="d")
    with pytest.raises(ValueError, match="non-finite"):
        kp.load(path)


def test_nonfinite_resident_score_is_rejected(tmp_path):
    # page 0 is selected at t=0 and then resident; EMA reads its score at t=1
    path = tmp_path / "invalid.npz"
    S = np.array([[[1.0, 0.5, np.nan]], [[np.nan, 2.0, 1.0]]])
    np.savez(path, ranked=np.array([[[0, 1]], [[1, 2]]]), scores=S, rid="d")
    with pytest.raises(ValueError, match="non-finite"):
        kp.load(path)


def test_unselected_future_pages_may_be_nan(tmp_path):
    path = tmp_path / "ok.npz"
    S = np.array([[[1.0, 0.5, np.nan]], [[0.1, 2.0, 1.0]]], dtype=np.float16)
    np.savez(path, ranked=np.array([[[0, 1]], [[1, 2]]], dtype=np.int16), scores=S, rid="t/d")
    R, S2, rid = kp.load(path)
    assert rid == "t/d" and S2.dtype == np.float32 and R.shape == (2, 1, 2)


@pytest.mark.parametrize("bad", [
    dict(ranked=np.array([[[0, 1]]]), scores=np.ones((1, 2, 2)), rid="d"),        # L mismatch
    dict(ranked=np.array([[0, 1]]), scores=np.ones((1, 2)), rid="d"),             # not 3-D
    dict(ranked=np.array([[[0.0, 1.0]]]), scores=np.ones((1, 1, 2)), rid="d"),    # float ids
    dict(ranked=np.array([[[0, 3]]]), scores=np.ones((1, 1, 2)), rid="d"),        # id >= P
    dict(ranked=np.array([[[1, 1]]]), scores=np.ones((1, 1, 2)), rid="d"),        # duplicate
    dict(ranked=np.array([[[0, 1]]]), scores=np.ones((1, 1, 2)), rid=""),         # empty rid
    dict(ranked=np.array([[[0, 1]]]), scores=np.ones((1, 1, 2))),                 # no rid
])
def test_malformed_npz_is_rejected(tmp_path, bad):
    path = tmp_path / "bad.npz"
    np.savez(path, **bad)
    with pytest.raises(ValueError):
        kp.load(path)


# F9: allocators reject infeasible inputs and handle all-zero weights.
CURVES = {0: {8: 10, 16: 1}, 1: {8: 10, 16: 1}}


@pytest.mark.parametrize("fn", [kp.allocate, kp.allocate_dp])
def test_allocator_rejects_infeasible_budget(fn):
    with pytest.raises(ValueError, match="minimum feasible"):
        fn(CURVES, 2, 8, [8, 16])


def test_proportional_rejects_infeasible_budget():
    with pytest.raises(ValueError, match="minimum feasible"):
        kp.allocate_proportional({0: 1, 1: 1}, 2, 8, [8, 16])


@pytest.mark.parametrize("grid", [[16, 8], [8, 8], [0, 8], [8.5, 16], []])
def test_allocator_rejects_bad_grid(grid):
    with pytest.raises(ValueError):
        kp.allocate(CURVES, 2, 32, grid)


def test_allocator_rejects_incomplete_curves():
    with pytest.raises(ValueError, match="curves"):
        kp.allocate({0: {8: 10, 16: 1}}, 2, 32, [8, 16])


@pytest.mark.parametrize("weights", [{0: 0, 1: 0}, {0: -1, 1: -3}])
def test_zero_weights_produce_feasible_allocation(weights):
    allocation = kp.allocate_proportional(weights, 2, 24, [8, 12, 16])
    assert allocation == {0: 8, 1: 8}


@pytest.mark.parametrize("fn", [kp.allocate, kp.allocate_dp])
def test_feasible_allocation_stays_within_total(fn):
    a = fn(CURVES, 2, 24, [8, 16])
    assert sum(a.values()) <= 24 and set(a.values()) <= {8, 16}
