"""followup-v3 policies: reductions to existing policies, bounds, direct references."""
import math

import numpy as np
import pytest

import kvpolicy as kp
import kvpolicy_v2 as k2
import kvpolicy_v3 as k3
from test_kvpolicy_v2 import TRACES


def same(a, b, B, ka=None, kb=None):
    for R, S in TRACES:
        assert kp.simulate(R, S, B, a, **(ka or {})) == kp.simulate(R, S, B, b, **(kb or {})), (a, ka, b, kb)


ALL = {"kernel": [dict(kernel="exp", a=0.5), dict(kernel="power", gamma=1.0, normalize=True, zero="median")],
       "ema_age": [dict(lam=0.5), dict(lam=0.5, clock="since_sel")], "ema_deficit": [dict(lam=1.0), dict(lam=math.inf)],
       "ema_unsel": [dict(lam=0.5)], "twoscale": [dict(fast=0.2, slow=0.9, w=0.5)],
       "lookahead": [dict(H=4), dict(H=16, tie="ema")], "scoped_ema": [dict(M=2), dict(M=2, which="bottom")],
       "refrank_ema": [dict(R=16)], "shrink_ema": [dict(c=0.5), dict(c=0.5, prior="zero")]}


@pytest.mark.parametrize("name", sorted(ALL))
@pytest.mark.parametrize("B", [8, 12, 20])
def test_bounded_by_belady(name, B):
    for kw in ALL[name]:
        for R, S in TRACES:
            c = kp.simulate(R, S, B, name, **kw)
            opt = kp.simulate(R, S, B, "belady")
            assert c["accesses"] == opt["accesses"] and c["compulsory"] == opt["compulsory"]
            assert opt["misses"] <= c["misses"] <= c["accesses"]


@pytest.mark.parametrize("B", [9, 16])
def test_reductions(B):
    for norm in (False, True):
        for zero in k3.ZEROS:
            same("kernel", "powerlaw", B, dict(kernel="power", gamma=1.0, normalize=norm, zero=zero),
                 dict(gamma=1.0, normalize=norm, zero=zero))
    same("ema_age", "ema", B, dict(alpha=0.5, lam=0.0), dict(alpha=0.5))
    same("ema_unsel", "ema", B, dict(alpha=0.5, lam=0.0), dict(alpha=0.5))
    same("ema_deficit", "ema", B, dict(alpha=0.5, lam=0.0), dict(alpha=0.5))
    same("twoscale", "ema", B, dict(fast=0.4, slow=0.4, w=0.3), dict(alpha=0.4))
    same("lookahead", "belady", B, dict(H=None))
    same("lookahead", "lru", B, dict(H=0, tie="lru"))
    same("lookahead", "ema", B, dict(H=0, tie="ema", alpha=0.5), dict(alpha=0.5))
    same("scoped_ema", "ema", B, dict(alpha=0.5, M=None), dict(alpha=0.5))
    same("scoped_ema", "ema", B, dict(alpha=0.5, M=0), dict(alpha=0.5, mode="selected"))
    same("scoped_ema", "ema", B, dict(alpha=0.5, M=10 ** 6), dict(alpha=0.5))
    same("refrank_ema", "ema", B, dict(alpha=0.5, R=8), dict(alpha=0.5, signal="rank"))  # K = 8, float32 scores
    same("shrink_ema", "ema", B, dict(alpha=0.5, c=0.0), dict(alpha=0.5))


def test_lookahead_monotone_on_average():
    tot = {H: 0 for H in (0, 4, 16, 64, None)}
    for R, S in TRACES:
        for H in tot:
            tot[H] += kp.simulate(R, S, 12, "lookahead", H=H)["misses"]
    assert tot[None] <= tot[64] <= tot[4] <= tot[0]


def test_exp_kernel_direct_value():
    """Kernel(exp) value equals sum a^age x (and the normalised mean) recorded along the run."""
    R, S = TRACES[0]
    hist, rows = {}, []

    class Rec(k3.Kernel):
        def begin(self, t, ranked, srow):
            super().begin(t, ranked, srow)
            for p in self.acc:
                hist[p].append(self.x(p))

        def admit(self, hits, misses):
            super().admit(hits, misses)
            for p in misses:
                hist[p] = [self.x(p)]

    def cb(t, victims, pol):
        for p in pol.acc:
            rows.append((pol.value(p), list(hist[p]), pol.normalize))

    kp.POLICIES["_rec"] = Rec
    try:
        for norm in (False, True):
            kp.simulate(R, S, 12, "_rec", on_evict=cb, kernel="exp", a=0.7, normalize=norm, zero="min")
    finally:
        del kp.POLICIES["_rec"]
    assert rows
    for v, xs, norm in rows:
        n = len(xs)
        num = sum(x * 0.7 ** (n - 1 - i) for i, x in enumerate(xs))
        den = sum(0.7 ** (n - 1 - i) for i in range(n))
        assert v == pytest.approx(num / den if norm else num, rel=1e-9, abs=1e-9)


def test_deficit_accumulates_negative_part():
    R, S = TRACES[1]
    seen = []

    def cb(t, victims, pol):
        seen.extend(pol.D.values())
    kp.simulate(R, S, 12, "ema_deficit", on_evict=cb, lam=1.0)
    assert seen and max(seen) <= 0.0 and min(seen) < 0.0


def test_scoped_ema_staleness_tracks_updates():
    R, S = TRACES[2]
    stale = []

    def cb(t, victims, pol):
        stale.extend(t - pol.upd[p] for p in pol.ema)
    kp.simulate(R, S, 12, "scoped_ema", on_evict=cb, M=0)
    assert max(stale) > 0
    stale.clear()
    kp.simulate(R, S, 12, "scoped_ema", on_evict=cb, M=None)
    assert max(stale) == 0


def test_shrink_toward_zero_lowers_new_pages():
    R, S = TRACES[0]
    out = {}
    for c in (0.0, 0.9):
        vals = []

        class Rec(k3.ShrinkEMA):
            def admit(self, hits, misses):
                super().admit(hits, misses)
                vals.extend(abs(self.ema[p]) for p in misses)
        kp.POLICIES["_rec"] = Rec
        try:
            kp.simulate(R, S, 12, "_rec", c=c, prior="zero")
        finally:
            del kp.POLICIES["_rec"]
        out[c] = np.mean(vals)
    assert out[0.9] < out[0.0]


@pytest.mark.parametrize("name,kw", [("lru", {}), ("ema", dict(alpha=0.5)), ("belady", {}),
                                     ("lookahead", dict(H=8))])
def test_simulate_steps_matches_simulate(name, kw):
    for R, S in TRACES:
        c, per = k3.simulate_steps(R, S, 12, name, **kw)
        assert c == kp.simulate(R, S, 12, name, **kw)
        assert per.sum() == c["misses"] and len(per) == len(R)
