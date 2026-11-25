"""collect/long.py: same selection as notebooks/long_trace_capture.py, brute-force selector check,
and the optional read-outs (attn_mass, true_rho) leave the selection and the output unchanged."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from collect import long as cl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def notebook_module():
    spec = importlib.util.spec_from_file_location("ltc", ROOT / "notebooks/long_trace_capture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_backend(be, steps, *, kvh=2, g=3, dim=8, ps=4, start=9, seed=0, L=2):
    """Drive a selector through `steps` decode steps on random q/k/v; returns outputs per step."""
    gen = torch.Generator().manual_seed(seed)
    T = steps
    pmax = (start + T) // ps + 1
    be.RANK = torch.full((T, L, be.top_k), -1, dtype=torch.int16)
    be.SCORE = torch.full((T, L, pmax), float("nan"), dtype=torch.float16)
    if getattr(be, "record_attn_mass", False):
        be.ATTN = torch.full((T, L, be.top_k), float("nan"), dtype=torch.float16)
    if getattr(be, "record_true_attn_rank", False):
        be.RHO = torch.full((T, L), float("nan"))
    keys = torch.randn(L, 1, kvh, start + T, dim, generator=gen)
    vals = torch.randn(L, 1, kvh, start + T, dim, generator=gen)
    outs = []
    for t in range(T):
        be.step = t
        for layer in range(L):
            q = torch.randn(1, kvh * g, 1, dim, generator=gen)
            n = start + t + 1
            o, _ = be(SimpleNamespace(layer_idx=layer), q, keys[layer, :, :, :n], vals[layer, :, :, :n],
                      None, scaling=dim ** -0.5)
            outs.append(o)
    return outs, keys, vals


@pytest.mark.parametrize("top_k", [1, 3, 64])
def test_same_selection_as_notebook(top_k):
    nb = notebook_module()
    a, b = nb.QuestPages(4, top_k), cl.QuestPages(4, top_k)
    start = 4 * top_k + 5  # the long-document selector assumes at least K complete pages
    oa, _, _ = run_backend(a, 20, seed=top_k, start=start)
    ob, _, _ = run_backend(b, 20, seed=top_k, start=start)
    assert torch.equal(a.RANK, b.RANK)
    assert torch.equal(a.SCORE.isnan(), b.SCORE.isnan())
    assert torch.equal(a.SCORE.nan_to_num(), b.SCORE.nan_to_num())
    for x, y in zip(oa, ob):
        assert torch.equal(x, y)


def test_brute_force_selector():
    torch.manual_seed(7)
    q = torch.randn(1, 4, 1, 4)
    k = torch.randn(1, 2, 13, 4)
    v = torch.randn_like(k)
    module = SimpleNamespace(layer_idx=0)
    for top_k in (1, 2):
        for length in (9, 12, 13):
            be = cl.QuestPages(page_size=4, top_k=top_k)
            be.RANK = torch.full((1, 1, top_k), -1, dtype=torch.int16)
            be.SCORE = torch.full((1, 1, 4), float("nan"), dtype=torch.float16)
            be.step = 0
            out, _ = be(module, q, k[:, :, :length], v[:, :, :length], None, scaling=0.5)
            n = (length - 1) // 4
            scores = []
            for p in range(n):
                keys = k[0, :, p*4:(p+1)*4].repeat_interleave(2, dim=0)
                scores.append((q[0] * keys).amax(dim=1).sum(-1).mean().item())
            ids = sorted(range(n), key=lambda p: -scores[p])[:top_k]
            got = [x for x in be.RANK[0, 0].tolist() if x >= 0]
            assert got == ids
            torch.testing.assert_close(be.SCORE[0, 0, :n].float(), torch.tensor(scores),
                                       rtol=1e-3, atol=1e-3)
            positions = [i for p in sorted(ids) for i in range(p*4, p*4+4)] + list(range(4*n, length))
            keys = k[:, :, positions].repeat_interleave(2, dim=1)
            values = v[:, :, positions].repeat_interleave(2, dim=1)
            reference = ((q @ keys.transpose(-1, -2) * 0.5).softmax(-1) @ values).transpose(1, 2)
            torch.testing.assert_close(out, reference)


def test_recording_does_not_change_selection():
    plain = cl.QuestPages(4, 3)
    rec = cl.QuestPages(4, 3, record_attn_mass=True, record_true_attn_rank=True)
    oa, _, _ = run_backend(plain, 25, seed=11, start=17)
    ob, keys, _ = run_backend(rec, 25, seed=11, start=17)
    assert torch.equal(plain.RANK, rec.RANK)
    assert torch.equal(plain.SCORE.nan_to_num(), rec.SCORE.nan_to_num())
    for x, y in zip(oa, ob):
        assert torch.equal(x, y)
    # attention mass: one value per selected page, in (0, H]; total over pages <= H (= kvh * g = 6)
    m = rec.ATTN.float()
    sel = rec.RANK >= 0
    assert torch.isfinite(m[sel]).all() and (m[sel] > 0).all()
    assert (m.nan_to_num().sum(-1) <= 6 + 1e-2).all()
    assert torch.isfinite(rec.RHO).all() and (rec.RHO.abs() <= 1 + 1e-6).all()


def test_attn_mass_matches_explicit_softmax():
    torch.manual_seed(3)
    kvh, g, dim, ps = 2, 2, 4, 4
    q = torch.randn(1, kvh*g, 1, dim)
    k = torch.randn(1, kvh, 18, dim)
    v = torch.randn_like(k)
    be = cl.QuestPages(ps, 2, record_attn_mass=True)
    be.RANK = torch.full((1, 1, 2), -1, dtype=torch.int16)
    be.SCORE = torch.full((1, 1, 5), float("nan"), dtype=torch.float16)
    be.ATTN = torch.full((1, 1, 2), float("nan"), dtype=torch.float16)
    be.step = 0
    be(SimpleNamespace(layer_idx=0), q, k, v, None, scaling=0.5)
    ranked = be.RANK[0, 0].long().tolist()
    hp = (18 - 1) // ps
    pos = [i for p in sorted(ranked) for i in range(p*ps, p*ps+ps)] + list(range(hp*ps, 18))
    kk = k[0][:, pos].repeat_interleave(g, 0)
    w = ((q[0, :, 0][:, None] @ kk.transpose(-1, -2))[:, 0] * 0.5).softmax(-1)
    for j, p in enumerate(ranked):
        idx = [pos.index(i) for i in range(p*ps, p*ps+ps)]
        assert abs(w[:, idx].sum().item() - be.ATTN[0, 0, j].item()) < 5e-3


def test_spearman():
    a = torch.tensor([1., 2., 3., 4.])
    assert cl.spearman(a, a) == pytest.approx(1)
    assert cl.spearman(a, -a) == pytest.approx(-1)
    # ties: scipy.stats.spearmanr([1,1,2,3],[1,2,3,4]) = 0.9486832980505138
    assert cl.spearman(torch.tensor([1., 1., 2., 3.]), a) == pytest.approx(0.9486832980505138)
