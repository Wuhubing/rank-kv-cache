"""D3 noise models on synthetic rows."""
import numpy as np

from analysis_v3 import noise as nz
from test_kvpolicy_v2 import synthetic


def test_models_keep_shape_and_none_is_identity():
    R, S = synthetic(seed=2, T=60, P0=30, K=8)
    rng = np.random.default_rng(0)
    assert np.array_equal(np.nan_to_num(nz.noisy(S, "none", rng)), np.nan_to_num(S.astype(np.float64)))
    pool = [np.random.default_rng(1).standard_normal((4, 60))]
    for m in ("iid:0.3", "ar0.9:0.3", "mono2", "emp"):
        X = nz.noisy(S, m, np.random.default_rng(0), pool)
        assert X.shape == S.shape
        fin = np.isfinite(S)
        assert np.isfinite(X[fin]).all()


def test_ar_noise_is_time_correlated():
    S = np.zeros((400, 50)) + np.arange(50)[None, :]
    X = nz.noisy(S, "ar0.9:1.0", np.random.default_rng(0)) - S
    a, b = X[:-1].ravel(), X[1:].ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.8
    Y = nz.noisy(S, "iid:1.0", np.random.default_rng(0)) - S
    assert abs(np.corrcoef(Y[:-1].ravel(), Y[1:].ravel())[0, 1]) < 0.1


def test_mono_preserves_order_mostly():
    R, S = synthetic(seed=4, T=40, P0=30, K=8)
    t = 30
    n = int(np.isfinite(S[t]).sum())
    a = np.argsort(np.argsort(S[t, :n]))
    rho = {}
    for m in ("mono0.5", "mono2"):
        X = nz.noisy(S, m, np.random.default_rng(0))
        rho[m] = np.corrcoef(a, np.argsort(np.argsort(X[t, :n])))[0, 1]
    # k=2 compresses the low scores, so the small perturbation reorders them more
    assert rho["mono0.5"] > 0.95 and 0.5 < rho["mono2"] < rho["mono0.5"]


def test_counterfactual_topk():
    R, S = synthetic(seed=5, T=30, P0=30, K=8)
    Rn = nz.topk(S.astype(np.float64), S, 8)
    assert (Rn == R).all()
