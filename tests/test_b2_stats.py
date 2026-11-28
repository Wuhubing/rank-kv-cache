"""B2 statistics on small hand-made inputs."""
import numpy as np
import pytest

from analysis_v3 import b2_stats as b2
from test_kvpolicy_v2 import synthetic


def test_percentiles_mid_rank_and_nan():
    S = np.array([[1.0, 3.0, 2.0, np.nan], [2.0, 2.0, 1.0, 5.0]])
    P = b2.percentiles(S)
    assert P[0, :3].tolist() == [0.0, 1.0, 0.5] and np.isnan(P[0, 3])
    assert P[1].tolist() == pytest.approx([0.5, 0.5, 0.0, 1.0])


def test_layer_stats_ranges():
    R, S = synthetic(seed=3, T=100, P0=40, K=8)
    st, sel = b2.layer_stats(R, S, 8)
    assert 0 < st["overlap"] <= 1
    assert sel.sum() == (R >= 0).sum()
    for L in b2.LAGS:
        if L < 100:
            assert -1 <= st[f"ac_lag{L}"] <= 1
    assert st["ac_lag1"] >= st["ac_lag64"]          # a random walk decorrelates with the lag
    assert 0 <= st["reuse_le1"] <= st["reuse_le256"] <= 1


def test_job_rows(tmp_path):
    layers = [synthetic(seed=l, T=80, P0=90, K=8) for l in range(2)]
    R = np.stack([x[0] for x in layers], 1).astype(np.int16)
    S = np.stack([x[1] for x in layers], 1).astype(np.float16)
    np.savez(tmp_path / "t.npz", ranked=R, scores=S, rid=np.array("t"))
    rows = b2.job(("syn", str(tmp_path / "t.npz"), [1.5, 2]))
    assert len(rows) == 2
    r = rows[0]
    assert r["bel_B12"] <= r["lru_B12"] and r["bel_B16"] <= r["lru_B16"]
    assert 0 <= r["ret_le1"] <= r["ret_le256"] <= 1
