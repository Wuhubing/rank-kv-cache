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
    if getattr(be, "record_arkvale_digest", False):
        be.DIGEST = {n: torch.full((T, L, pmax), float("nan"), dtype=torch.float16) for n in cl.DIGESTS}
    if getattr(be, "record_attn_all", False):
        be.ATTN_ALL = torch.full((T, L, pmax), float("nan"), dtype=torch.float16)
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


def test_attn_all_does_not_change_selection_and_matches_dense_softmax():
    plain = cl.QuestPages(4, 3)
    rec = cl.QuestPages(4, 3, record_attn_all=True)
    oa, _, _ = run_backend(plain, 25, seed=12, start=17)
    ob, keys, _ = run_backend(rec, 25, seed=12, start=17)
    assert torch.equal(plain.RANK, rec.RANK)
    assert torch.equal(plain.SCORE.nan_to_num(), rec.SCORE.nan_to_num())
    assert torch.equal(plain.SCORE.isnan(), rec.SCORE.isnan())
    for x, y in zip(oa, ob):
        assert torch.equal(x, y)
    # finite exactly where a score exists (completed pages), mean over heads of the page's dense mass
    assert torch.equal(rec.ATTN_ALL.isnan(), rec.SCORE.isnan())
    tot = rec.ATTN_ALL.float().nan_to_num().sum(-1)
    assert ((tot > 0) & (tot <= 1 + 1e-2)).all()


def test_attn_all_value_matches_explicit_softmax():
    torch.manual_seed(5)
    kvh, g, dim, ps = 2, 3, 4, 4
    q = torch.randn(1, kvh*g, 1, dim)
    k = torch.randn(1, kvh, 23, dim)
    v = torch.randn_like(k)
    be = cl.QuestPages(ps, 2, record_attn_all=True)
    be.RANK = torch.full((1, 1, 2), -1, dtype=torch.int16)
    be.SCORE = torch.full((1, 1, 6), float("nan"), dtype=torch.float16)
    be.ATTN_ALL = torch.full((1, 1, 6), float("nan"), dtype=torch.float16)
    be.step = 0
    be(SimpleNamespace(layer_idx=0), q, k, v, None, scaling=0.5)
    hp = (23 - 1) // ps
    kk = k[0].repeat_interleave(g, 0)                                  # [H, 23, d]
    w = ((q[0, :, 0][:, None] @ kk.transpose(-1, -2))[:, 0] * 0.5).softmax(-1)   # [H, 23]
    for p in range(hp):
        ref = w[:, p*ps:(p+1)*ps].sum(-1).mean().item()
        assert abs(be.ATTN_ALL[0, 0, p].item() - ref) < 2e-3
    assert torch.isnan(be.ATTN_ALL[0, 0, hp:]).all()


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


# ---------------------------------------------------------------- ArkVale digests (--record-arkvale-digest)

def official_cuboid_mean(keys):
    """Literal port of ArkVale InferState._summarize_keys (infer_state.py:448-462) for one page:
    keys [page, kvh, d] -> (maxs, mins) [kvh, d]."""
    maxs = keys.max(dim=0).values
    mins = keys.min(dim=0).values
    centers = (maxs + mins) / 2
    dists = (centers[None] - keys).abs().mean(dim=0)
    return centers + dists, centers - dists


def official_estimate(q, maxs, mins):
    """ArkVale estimate.cu:43 (o += max(q*k, q*v) with k = maxs, v = mins) for every query head, then
    the mean over query heads (estimate.cu:551-552 with n_groups = 1). q [H, d]; maxs/mins [kvh, d]."""
    g = q.shape[0] // maxs.shape[0]
    per_head = [torch.maximum(q[h] * maxs[h // g], q[h] * mins[h // g]).sum() for h in range(q.shape[0])]
    return torch.stack(per_head).mean()


def paper_bounds(keys, name):
    """Paper section 5.2, Eqs. 1-4, written out per page. keys [page, kvh, d]. Returns a function of q [H, d]."""
    lo, hi = keys.min(0).values, keys.max(0).values
    c = (lo + hi) / 2
    dev = keys - c[None]
    if name.startswith("cuboid"):
        ad = dev.abs()
        r = ad.mean(0) if name == "cuboid_mean" else (ad.min(0).values + ad.max(0).values) / 2
        return lambda q: official_estimate(q, c + r, c - r)
    l2 = dev.norm(dim=-1)                                   # [page, kvh]
    R = {"sphere_max": l2.max(0).values, "sphere_center": (l2.min(0).values + l2.max(0).values) / 2,
         "sphere_mean": l2.mean(0)}[name]
    def f(q):
        g = q.shape[0] // c.shape[0]
        return torch.stack([q[h] @ c[h // g] + R[h // g] * q[h].norm() for h in range(q.shape[0])]).mean()
    return f


def test_digest_recording_does_not_change_selection():
    plain = cl.QuestPages(4, 3)
    rec = cl.QuestPages(4, 3, record_arkvale_digest=True)
    oa, _, _ = run_backend(plain, 25, seed=5, start=17)
    ob, _, _ = run_backend(rec, 25, seed=5, start=17)
    assert torch.equal(plain.RANK, rec.RANK)
    assert torch.equal(plain.SCORE.isnan(), rec.SCORE.isnan())
    assert torch.equal(plain.SCORE.nan_to_num(), rec.SCORE.nan_to_num())
    for x, y in zip(oa, ob):
        assert torch.equal(x, y)
    for n in cl.DIGESTS:  # same layout as scores: finite exactly where scores are
        assert torch.equal(rec.DIGEST[n].isnan(), rec.SCORE.isnan())


def test_digest_values_match_official_code_and_paper():
    torch.manual_seed(1)
    kvh, g, dim, ps, length = 2, 3, 8, 4, 23
    q = torch.randn(1, kvh * g, 1, dim)
    k = torch.randn(1, kvh, length, dim)
    v = torch.randn_like(k)
    be = cl.QuestPages(ps, 2, record_arkvale_digest=True)
    hp = (length - 1) // ps
    be.RANK = torch.full((1, 1, 2), -1, dtype=torch.int16)
    be.SCORE = torch.full((1, 1, hp), float("nan"), dtype=torch.float16)
    be.DIGEST = {n: torch.full((1, 1, hp), float("nan"), dtype=torch.float16) for n in cl.DIGESTS}
    be.step = 0
    be(SimpleNamespace(layer_idx=0), q, k, v, None, scaling=0.5)
    qh = q[0, :, 0]
    for p in range(hp):
        keys = k[0, :, p*ps:(p+1)*ps].transpose(0, 1)            # [page, kvh, d]
        mx, mn = official_cuboid_mean(keys)
        ref = official_estimate(qh, mx, mn).item()
        assert abs(be.DIGEST["cuboid_mean"][0, 0, p].item() - ref) < 2e-2 * max(1, abs(ref))
        for n in cl.DIGESTS:
            ref = paper_bounds(keys, n)(qh).item()
            assert abs(be.DIGEST[n][0, 0, p].item() - ref) < 2e-2 * max(1, abs(ref)), n
        # cuboid-max (raw AABB) is the Quest score
        ref = official_estimate(qh, keys.max(0).values, keys.min(0).values).item()
        assert abs(be.SCORE[0, 0, p].item() - ref) < 2e-2 * max(1, abs(ref))


def test_digest_stats_incremental_equals_one_shot():
    """Pages completed over many steps (stats appended page by page) give the same digest statistics
    as computing all pages at once."""
    torch.manual_seed(2)
    kvh, g, dim, ps, start, T = 2, 2, 8, 4, 9, 30
    k = torch.randn(1, kvh, start + T, dim)
    v = torch.randn_like(k)
    be = cl.QuestPages(ps, 2, record_arkvale_digest=True)
    pmax = (start + T) // ps + 1
    be.RANK = torch.full((T, 1, 2), -1, dtype=torch.int16)
    be.SCORE = torch.full((T, 1, pmax), float("nan"), dtype=torch.float16)
    be.DIGEST = {n: torch.full((T, 1, pmax), float("nan"), dtype=torch.float16) for n in cl.DIGESTS}
    for t in range(T):
        be.step = t
        n = start + t + 1
        be(SimpleNamespace(layer_idx=0), torch.randn(1, kvh * g, 1, dim), k[:, :, :n], v[:, :, :n], None,
           scaling=0.5)
    hp = (start + T - 1) // ps
    kk = k[0, :, :hp * ps].reshape(kvh, hp, ps, dim).float()
    ref = cl.page_digest_stats(kk, kk.amin(2), kk.amax(2))
    st = be.dstats[0]
    assert set(st) == {"c", "r_mean", "r_center", "R_max", "R_center", "R_mean"}
    for name in st:
        assert st[name].shape[1] == hp
        torch.testing.assert_close(st[name], ref[name])


def test_arkvale_selector():
    torch.manual_seed(4)
    kvh, g, dim, ps, length, K = 2, 2, 8, 4, 39, 5
    q = torch.randn(1, kvh * g, 1, dim)
    k = torch.randn(1, kvh, length, dim)
    v = torch.randn_like(k)
    hp = (length - 1) // ps                                     # 9 completed pages
    be = cl.QuestPages(ps, K, selector="arkvale-mean", arkvale_sink=1, arkvale_recent=1)
    be.RANK = torch.full((1, 1, K), -1, dtype=torch.int16)
    be.SCORE = torch.full((1, 1, hp), float("nan"), dtype=torch.float16)
    be.step = 0
    out, _ = be(SimpleNamespace(layer_idx=0), q, k, v, None, scaling=0.5)
    qh = q[0, :, 0]
    scores = []
    for p in range(hp):
        mx, mn = official_cuboid_mean(k[0, :, p*ps:(p+1)*ps].transpose(0, 1))
        scores.append(official_estimate(qh, mx, mn).item())
    forced = [0, hp - 1]
    cand = [p for p in range(hp) if p not in forced]
    top = sorted(cand, key=lambda p: -scores[p])[:K - 2]
    got = be.RANK[0, 0].tolist()
    assert got[:K - 2] == top and got[K - 2:] == [-1, -1]
    torch.testing.assert_close(be.SCORE[0, 0].float(), torch.tensor(scores), rtol=2e-3, atol=2e-3)
    attended = sorted(top + forced)
    pos = [i for p in attended for i in range(p*ps, p*ps+ps)] + list(range(hp*ps, length))
    kk = k[:, :, pos].repeat_interleave(g, 1)
    vv = v[:, :, pos].repeat_interleave(g, 1)
    ref = ((q @ kk.transpose(-1, -2) * 0.5).softmax(-1) @ vv).transpose(1, 2)
    torch.testing.assert_close(out, ref)


def test_bad_selector():
    with pytest.raises(ValueError):
        cl.QuestPages(4, 2, selector="nope")


def test_parquet_source_builds_the_longbench_prompt():
    from collect import sources

    class Tok:
        def apply_chat_template(self, msg, **kw):
            return list(msg[0]["content"].encode())
    p = dict(task="gov_report", rid="govreport/test/x", file="f", row=0, field="report", source_spec="a@b")
    row = {"context": "Hello report"}
    got = sources.build_turns(Tok(), "parquet", p, row)
    assert got == [list(cl.USER["gov_report"].format(context="Hello report").encode())]
    assert sources.context_hash("parquet", row) == cl.context_hash("longbench", row)
