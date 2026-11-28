"""analysis_v3.run on synthetic traces: replay rows match kvpolicy.simulate, summaries are consistent."""
import argparse
import csv

import numpy as np
import pytest

import kvpolicy as kp
from analysis_v3 import experiments, run
from test_kvpolicy_v2 import synthetic


def write_traces(folder, n=4, L=2, attn=False):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        layers = [synthetic(seed=10 * i + l, T=90, P0=100, K=8) for l in range(L)]
        R = np.stack([x[0] for x in layers], 1).astype(np.int16)
        S = np.stack([x[1] for x in layers], 1).astype(np.float16)
        extra = {}
        if attn:
            A = np.where(np.isfinite(S), np.abs(S.astype(np.float32)) + 0.01, np.nan).astype(np.float16)
            extra["attn_all"] = A
        np.savez(folder / f"d{i}.npz", ranked=R, scores=S, rid=np.array(f"doc{i}"), **extra)


def test_replay_and_summarize(tmp_path):
    write_traces(tmp_path / "tr")
    a = argparse.Namespace(exp="b3", traces=tmp_path / "tr", out=tmp_path / "out", procs=2, budgets=[10, 16],
                           misses=None)
    run.cmd_replay(a)
    rows = list(csv.DictReader(open(tmp_path / "out/misses.csv")))
    e = experiments.get("b3")
    assert len(rows) == 4 * len(e.configs) * 2 * 2
    R, S, rid = kp.load(str(tmp_path / "tr/d1.npz"))
    r = [x for x in rows if x["doc"] == rid and x["config"] == "look_lru_H16" and x["B"] == "10" and x["layer"] == "1"]
    assert int(r[0]["misses"]) == kp.simulate(R[:, 1], S[:, 1], 10, "lookahead", H=16, tie="lru", alpha=0.5)["misses"]
    M = run.read_misses([tmp_path / "out/misses.csv"])
    e.budgets = [10, 16]
    per, srows, con, docs = run.summarize(e, M, tmp_path / "out")
    g = {(x["policy"], x["B"]): x for x in srows}
    assert g["lru", 10]["G"] == 0 and g["belady", 10]["G"] == 1
    assert g["look_lru_HNone", 10]["G"] == pytest.approx(1.0)
    assert g["look_lru_H1", 16]["G_lo"] <= g["look_lru_H1", 16]["G"] <= g["look_lru_H1", 16]["G_hi"]
    assert "ema_z(lodo)" in {x["policy"] for x in srows}


def test_attn_signal_replaces_scores(tmp_path):
    write_traces(tmp_path / "tr", n=1, attn=True)
    p = str(tmp_path / "tr/d0.npz")
    R, A, _ = run.load(p, "attn_all")
    _, S, _ = run.load(p, "scores")
    assert not np.array_equal(np.nan_to_num(A), np.nan_to_num(S))
    out = run.run_job((p, "b4", "ema_raw:0.5", [12]))
    assert out[0][5] == kp.simulate(R[:, 0], A[:, 0], 12, "ema", alpha=0.5, signal="raw")["misses"]


def test_lodo_and_contrast():
    M = {("a", 96): {"x": np.array([0, 10, 0]), "y": np.array([0, 31, 0]), "z": np.array([0, 31, 0])},
         ("b", 96): {"x": np.array([0, 50, 0]), "y": np.array([0, 30, 0]), "z": np.array([0, 30, 0])}}
    m, ch = run.lodo(M, ["x", "y", "z"], ["a", "b"], 96)
    assert ch == ["b", "a", "a"] and list(m) == [50, 31, 31]
    per = {("lru", 96): np.array([100., 100]), ("belady", 96): np.array([0., 0]), ("p", 96): np.array([70., 70]),
           ("q", 96): np.array([80., 80])}
    c = run.contrast(per, run.boot_idx(2), "p", "q", 96)
    assert c["dG"] == pytest.approx(0.1) and c["lo"] == pytest.approx(0.1) and c["docs_a_fewer"] == 2
