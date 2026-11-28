"""Learned policies (C2 linear; B3 trees when sklearn is present) on synthetic traces."""
import numpy as np
import pytest

import kvpolicy as kp
from analysis_v3 import learn
from test_kvpolicy_v2 import synthetic


@pytest.fixture
def trace(tmp_path):
    layers = [synthetic(seed=l, T=90, P0=100, K=8) for l in range(2)]
    R = np.stack([x[0] for x in layers], 1).astype(np.int16)
    S = np.stack([x[1] for x in layers], 1).astype(np.float16)
    p = tmp_path / "t.npz"
    np.savez(p, ranked=R, scores=S, rid=np.array("t"))
    return str(p)


def test_behaviour_is_lru_and_features_finite(trace):
    d = learn.collect(trace, 12, "linear", None)
    R, S, _ = kp.load(trace)
    assert d["misses"] == sum(kp.simulate(R[:, l], None, 12, "lru")["misses"] for l in range(2))
    assert d["X"].shape[1] == len(learn.FEATS) and np.isfinite(d["X"]).all()
    assert len(d["y"]) == len(d["X"]) == len(d["ext"])


def test_linear_fit_runs_and_is_bounded(trace):
    d = learn.collect(trace, 12, "linear", None)
    coef, meta = learn.fit("linear", d["X"], d["y"], d["ext"], seed=0)
    assert len(coef) == len(learn.FEATS) + 1
    m = learn.collect(trace, 12, "linear", coef)["misses"]
    R, S, _ = kp.load(trace)
    opt = sum(kp.simulate(R[:, l], None, 12, "belady")["misses"] for l in range(2))
    assert m >= opt


def test_trees_if_available(trace):
    pytest.importorskip("sklearn")
    d = learn.collect(trace, 12, "gbdt", None)
    model, meta = learn.fit("gbdt", d["X"], d["y"], d["ext"], seed=0)
    assert learn.collect(trace, 12, "gbdt", model)["misses"] > 0
