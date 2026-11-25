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


# F1: the held-out driver takes every policy parameter from the frozen file.
import copy
import hashlib
import json
from pathlib import Path

import heldout

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "splits/heldout_frozen_v1.json"


def test_frozen_alpha_controls_evaluation():
    frozen = {"alpha": {"96": 0.123}, "linear": {}}
    assert heldout.configs(96, frozen)["ema"][1]["alpha"] == 0.123


def test_committed_frozen_file_validates_and_matches_constants():
    frozen, sha, L = heldout.load_frozen(FROZEN)
    assert L == 28 and sha == hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    for B in heldout.CURVE:
        assert heldout.configs(B, frozen)["ema"][1]["alpha"] == heldout.ALPHA[B]


def _mutated(fn):
    frozen = json.loads(FROZEN.read_text())
    fn(frozen)
    return frozen


@pytest.mark.parametrize("mutate", [
    lambda f: f["alpha"].pop("96"),
    lambda f: f["alpha"].update({"96": 1.5}),
    lambda f: f["alpha"].update({"96": "0.5"}),
    lambda f: f["alloc"]["96"].pop(),
    lambda f: f["alloc"]["96"].__setitem__(0, 65),
    lambda f: f["alloc"].update({"96": [256] * 28}),
    lambda f: f["linear"]["96"].pop(),
    lambda f: f["linear"]["96"].__setitem__(0, float("nan")),
    lambda f: f.pop("dev_docs"),
    lambda f: f["dev_docs"].append(f["dev_docs"][0]),
])
def test_invalid_frozen_parameters_are_rejected(mutate):
    with pytest.raises(ValueError):
        heldout.check_frozen(_mutated(mutate))


def test_configs_without_frozen_alpha_raise():
    with pytest.raises(ValueError, match="alpha"):
        heldout.configs(96, {"alpha": {}, "linear": {}})


# F2: held-out traces are validated against the split by internal document id.
from types import SimpleNamespace

SPLIT = json.loads((ROOT / "splits/longctx_v1.json").read_text())


class InlinePool:
    jobs = []

    def __init__(self, *args):
        pass

    def map(self, fn, jobs):
        InlinePool.jobs.extend(jobs)
        return list(map(fn, jobs))


def _write_trace(folder, rid, name=None, layers=28, width=64, steps=2, pages=100, **extra):
    rng = np.random.default_rng(abs(hash(rid)) % 2**32)
    S = rng.normal(size=(steps, layers, pages)).astype(np.float16)
    R = np.argsort(-S.astype(np.float32), axis=2, kind="stable")[:, :, :width].astype(np.int16)
    np.savez(folder / (name or heldout.trace_name(rid)), ranked=R, scores=S, rid=rid, **extra)


def _rids(subset):
    return [p["rid"] for p in SPLIT["prompts"] if p["split"] == subset]


def _eval(tmp_path, monkeypatch, frozen=FROZEN):
    monkeypatch.setattr(heldout, "CURVE", [96])
    monkeypatch.setattr(heldout, "Pool", InlinePool)
    InlinePool.jobs = []
    args = SimpleNamespace(traces=tmp_path / "traces", out=tmp_path / "out", procs=1, frozen=frozen)
    heldout.evaluate(args)
    return args


@pytest.fixture
def traces(tmp_path):
    folder = tmp_path / "traces"
    folder.mkdir()
    return folder


def test_renamed_development_trace_is_rejected(traces, tmp_path, monkeypatch):
    for rid in _rids("heldout")[1:]:
        _write_trace(traces, rid)
    _write_trace(traces, _rids("dev")[0], name="renamed.npz")
    with pytest.raises(ValueError, match="dev document"):
        _eval(tmp_path, monkeypatch)


def test_single_renamed_dev_trace_is_rejected(traces, tmp_path, monkeypatch):
    _write_trace(traces, _rids("dev")[0], name="renamed.npz", layers=1)
    with pytest.raises(ValueError, match="dev|held|split|document"):
        _eval(tmp_path, monkeypatch)


@pytest.mark.parametrize("case", ["missing", "duplicate", "renamed", "unknown", "layers", "width", "ctx"])
def test_incomplete_or_malformed_heldout_folder_is_rejected(traces, tmp_path, monkeypatch, case):
    rids = _rids("heldout")
    for rid in rids[1:]:
        _write_trace(traces, rid)
    first = rids[0]
    if case == "duplicate":
        _write_trace(traces, rids[1], name="copy.npz")
    elif case == "renamed":
        _write_trace(traces, first, name="other.npz")
    elif case == "unknown":
        _write_trace(traces, "gov_report/not-in-split")
    elif case == "layers":
        _write_trace(traces, first, layers=27)
    elif case == "width":
        _write_trace(traces, first, width=32)
    elif case == "ctx":
        _write_trace(traces, first, ctx="0000000000000000")
    with pytest.raises(ValueError):
        _eval(tmp_path, monkeypatch)


def test_valid_heldout_folder_uses_frozen_alpha_and_records_provenance(traces, tmp_path, monkeypatch):
    for rid in _rids("heldout"):
        _write_trace(traces, rid)
    frozen = json.loads(FROZEN.read_text())
    frozen["alpha"]["96"] = 0.123
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(frozen))
    args = _eval(tmp_path, monkeypatch, frozen=path)
    ema = [kw["alpha"] for _, name, kw, _ in InlinePool.jobs if name == "ema"]
    assert ema and set(ema) == {0.123}
    prov = json.loads((args.out / "provenance.json").read_text())
    assert prov["frozen_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert prov["configs"]["96"]["ema"][1]["alpha"] == 0.123
    assert prov["docs"] == sorted(_rids("heldout"))
    rows = (args.out / "misses.csv").read_text().splitlines()
    assert len({r.split(",")[1] for r in rows[1:]}) == 21


def test_freeze_rejects_heldout_trace_in_dev_folder(traces, tmp_path):
    for rid in _rids("dev")[1:]:
        _write_trace(traces, rid)
    _write_trace(traces, _rids("heldout")[0], name=heldout.trace_name(_rids("dev")[0]))
    args = SimpleNamespace(traces=traces, out=tmp_path / "frozen.json", procs=1)
    with pytest.raises(ValueError, match="heldout document"):
        heldout.freeze(args)
