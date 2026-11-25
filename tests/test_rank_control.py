"""End-to-end check of validation/rank_control_dev.py on synthetic schema-v2 NPZs."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import kvpolicy as kp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import rank_control_dev as rc  # noqa: E402
from test_kvpolicy import synthetic  # noqa: E402


def write_docs(folder, n=3, L=2):
    for i in range(n):
        layers = [synthetic(T=80, P=50, K=8, seed=10 * i + l) for l in range(L)]
        R = np.stack([r for r, _ in layers], 1).astype(np.int16)
        S = np.stack([s for _, s in layers], 1).astype(np.float16)
        np.savez(folder / f"d{i}.npz", ranked=R, scores=S, rid=f"syn/d{i}")


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_rank_control_pipeline(tmp_path):
    write_docs(tmp_path)
    out = tmp_path / "o"
    rc.main([str(tmp_path), "--out", str(out), "--workers", "2", "--budgets", "10", "12",
             "--alphas", "0.3", "0.7", "--no-split-check", "--archive", str(tmp_path / "none")])
    misses = read(out / "misses.csv")
    summary = read(out / "summary.csv")
    meta = json.loads((out / "summary.json").read_text())
    assert {r["signal"] for r in summary} == {"z", "allrank", "rank"}
    assert {int(r["B"]) for r in summary} == {10, 12}
    assert len(meta["traces"]) == 3 and meta["archive_rows_matched"] == 0
    assert {r["contrast"] for r in read(out / "contrast.csv")} == {"allrank-z", "allrank-rank", "z-rank"}

    # z configs are exactly the main policy (ScoreEMA defaults) used by efficient_dev / heldout
    R, S, rid = kp.load(tmp_path / "d1.npz")
    for row in misses:
        if row["doc"] == rid and row["config"] == "ema:0.3" and row["B"] == "10":
            l = int(row["layer"])
            assert int(row["misses"]) == kp.simulate(R[:, l], S[:, l], 10, "ema", alpha=0.3)["misses"]

    # LODO picks are recomputable from misses.csv with the efficient_dev rule
    tot = {}
    for r in misses:
        if r["B"] == "12":
            tot[r["config"], r["doc"]] = tot.get((r["config"], r["doc"]), 0) + int(r["misses"])
    docs = sorted({d for _, d in tot})
    for r in read(out / "lodo.csv"):
        if r["B"] != "12":
            continue
        cfgs = [f"{rc.SIGNALS[r['signal']]}:{a}" for a in (0.3, 0.7)]
        train = [d for d in docs if d != r["doc"]]
        best = min(cfgs, key=lambda c: (sum(tot[c, d] for d in train), c))
        assert r["chosen"] == best and int(r["misses"]) == tot[best, r["doc"]]


def test_rank_control_archive_check(tmp_path):
    write_docs(tmp_path, n=2, L=1)
    archive = tmp_path / "arch"
    archive.mkdir()
    R, S, rid = kp.load(tmp_path / "d0.npz")
    c = kp.simulate(R[:, 0], S[:, 0], 10, "ema", alpha=0.3, signal="rank")
    rows = [("doc", "layer", "config", "B", "accesses", "misses", "compulsory"),
            (rid, 0, "ema_rank:0.3", 10, c["accesses"], c["misses"], c["compulsory"])]
    with open(archive / "misses_ext.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    args = [str(tmp_path), "--budgets", "10", "--alphas", "0.3", "--no-split-check", "--archive", str(archive)]
    rc.main(args + ["--out", str(tmp_path / "ok")])
    assert json.loads((tmp_path / "ok" / "summary.json").read_text())["archive_rows_matched"] == 1
    rows[1] = rows[1][:5] + (c["misses"] + 1, c["compulsory"])
    with open(archive / "misses_ext.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(ValueError, match="differs from archived"):
        rc.main(args + ["--out", str(tmp_path / "bad")])


def test_rank_control_rejects_non_dev_ids(tmp_path):
    write_docs(tmp_path, n=1, L=1)
    with pytest.raises(ValueError, match="Unexpected"):
        rc.main([str(tmp_path), "--out", str(tmp_path / "o")])
