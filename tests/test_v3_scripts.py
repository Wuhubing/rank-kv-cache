"""Smaller followup-v3 scripts on synthetic inputs: F1 survey helpers, E summary, F2 freeze choice, D5 / D6 summaries,
C5 cost table, B4 signal read-out, instrument summary."""
import csv
import gzip
import json
import subprocess
import sys

import numpy as np
import pytest

from test_kvpolicy_v2 import synthetic


def test_survey_helpers():
    from analysis_v2 import survey_v3 as sv
    a = "the quick brown fox jumps over the lazy dog and runs far away"
    assert sv.shingles(a) == sv.shingles(a.upper())
    assert len(sv.shingles(a)) == len(a.split()) - 7
    assert sv.order_key("x") == sv.order_key("x") != sv.order_key("y")
    assert sv.ctx_hash("abc") == sv.ctx_hash("abc") and len(sv.ctx_hash("abc")) == 16


def _fake_e(folder):
    folder.mkdir()
    (folder / "meta.json").write_text(json.dumps({"args": {"warmup": 2}}))
    runs = []
    rng = np.random.default_rng(0)
    for rnd in range(3):
        for pol, base, m in (("lru", 50.0, 10), ("ema", 49.0, 9), ("belady", 48.0, 7), ("allmiss", 60.0, 64),
                             ("nofetch", 45.0, 0)):
            lat = (base + rng.normal(0, 0.5, 12)).tolist()
            runs.append(dict(R=2, round=rnd, policy=pol, delay_ns_per_byte=0.0, lat_ms=lat,
                             step_misses=[m] * 12, page_step=[i % 4 == 0 for i in range(12)], misses=m * 10,
                             tokens_per_s=2 * 10 / (sum(lat[2:]) / 1e3), ranks_equal_prep=True,
                             tokens_equal_prep=True, before={"loadavg1": 1.0, "other_pids": []},
                             after={"loadavg1": 1.0, "other_pids": []}))
    (folder / "batch2.json").write_text(json.dumps(dict(R=2, budget=96, page_bytes=65536, runs=runs)))


def test_e_summarize(tmp_path):
    _fake_e(tmp_path / "cfg")
    subprocess.run([sys.executable, "-m", "analysis_v3.e_summarize", str(tmp_path / "cfg"), "--out", str(tmp_path / "o")],
                   check=True, capture_output=True)
    rows = {r["policy"]: r for r in csv.DictReader(open(tmp_path / "o/throughput.csv"))}
    assert float(rows["lru"]["ratio_vs_lru"]) == pytest.approx(1.0)
    assert float(rows["belady"]["ratio_vs_lru"]) > 1.0
    d2 = list(csv.DictReader(open(tmp_path / "o/d2_model.csv")))[0]
    assert float(d2["transfer_max_ms"]) > 0


def test_freeze_choose():
    from analysis_v3.freeze_v3 import choose
    M = {("a", 96): {"x": 10, "y": 31, "z": 31}, ("b", 96): {"x": 50, "y": 30, "z": 30}}
    pooled, lodo = choose(M, ["a", "b"], 96)
    assert pooled == "a" and lodo == ["b", "a", "a"]


def _misses_gz(path, rows):
    with gzip.open(path, "wt") as f:
        w = csv.writer(f)
        w.writerow(["doc", "layer", "config", "B", "accesses", "misses", "compulsory"])
        w.writerows(rows)


def test_d5_summary(tmp_path):
    rows, stats = [], []
    rng = np.random.default_rng(1)
    for d in range(14):
        for l in range(6):
            base = 100 + rng.integers(0, 10)
            for c, m in (("lru", base), ("belady", base - 40), ("ema_fixed:0.5", base - 12), ("prot_lru_rho0.1", base + 3),
                         ("pin_ema_s1r1", base - 14)):
                rows.append((f"d{d}", l, c, 96, 1000, m, 5))
            stats.append(dict(doc=f"d{d}", layer=l, overlap=0.5 + 0.05 * l))
    _misses_gz(tmp_path / "m.csv.gz", rows)
    with open(tmp_path / "s.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc", "layer", "overlap"])
        w.writeheader()
        w.writerows(stats)
    subprocess.run([sys.executable, "-m", "analysis_v3.d5_summary", "--misses", str(tmp_path / "m.csv.gz"), "--stats",
                    str(tmp_path / "s.csv"), "--out", str(tmp_path / "o")], check=True, capture_output=True)
    out = {(r["config"], r["layers"]): r for r in csv.DictReader(open(tmp_path / "o/summary.csv"))}
    assert float(out["prot_lru_rho0.1", "all"]["dG"]) < 0 and float(out["pin_ema_s1r1", "all"]["dG"]) > 0


def test_d6_rule(tmp_path):
    from analysis_v3 import d6_alpha as d6
    syn = tmp_path / "syn"
    syn.mkdir()
    rows = []
    stats = {}
    for reg, ac in (("a", 0.9), ("b", 0.6), ("c", 0.3)):
        stats[reg] = {"0": {"ac_lag16": ac}}
        for ratio in (1.5, 4.0):
            best = (1 - ac) * 0.5 + 0.1 * np.log2(ratio)
            for al in d6.GRID[:-1]:
                rows.append(dict(regime=reg, B_over_K=ratio, policy="ema", param=al, misses=0,
                                 G=1 - abs(al - best)))
    with open(syn / "curves_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (syn / "regimes.json").write_text(json.dumps({"stats": stats}))
    c, pts = d6.fit_rule(syn)
    assert len(pts) == 6 and c[1] > 0 and c[2] > 0
    assert d6.rule(c, 0.9, 1.5) <= d6.rule(c, 0.3, 4.0)


def test_b4_signal(tmp_path):
    from analysis_v3 import b4_signal
    layers = [synthetic(seed=l, T=80, P0=60, K=8) for l in range(2)]
    R = np.stack([x[0] for x in layers], 1).astype(np.int16)
    S = np.stack([x[1] for x in layers], 1).astype(np.float16)
    A = np.where(np.isfinite(S), np.exp(S.astype(np.float32)), np.nan).astype(np.float16)
    np.savez(tmp_path / "t.npz", ranked=R, scores=S, attn_all=A, rid=np.array("t"))
    rows = b4_signal.job(str(tmp_path / "t.npz"))
    assert len(rows) == 2
    for r in rows:
        assert r[2] > 0.99          # attention is a monotone function of the score here
        assert 0.5 < r[4] <= 1 and abs(r[4] - r[5]) < 0.02


def test_d1_iso_interpolation():
    from analysis_v3.d1_iso import iso
    grid, lru = [64, 72, 80], [300, 200, 100]
    assert iso(grid, lru, 200) == 72 and iso(grid, lru, 150) == 76
    assert np.isnan(iso(grid, lru, 400)) and iso(grid, lru, 50) == float("inf")
