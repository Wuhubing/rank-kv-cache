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


def _fake_e_warm(folder, gz=True):
    """Runs whose warm-up steps miss far more than the timed steps; 'misses' = total over all steps (as in e_bench)."""
    folder.mkdir()
    (folder / "meta.json").write_text(json.dumps({"args": {"warmup": 2}}))
    runs = []
    for rnd in range(2):
        for pol, base, m in (("lru", 50.0, 10 + rnd), ("ema", 49.0, 8), ("allmiss", 60.0, 64), ("nofetch", 45.0, 1)):
            sm = [500, 400] + [m] * 10
            lat = [base] * 12
            others = [123] if (pol == "ema" and rnd == 1) else []
            runs.append(dict(R=2, round=rnd, policy=pol, delay_ns_per_byte=0.0, lat_ms=lat, step_misses=sm,
                             page_step=[i % 4 == 0 for i in range(12)], misses=sum(sm), tokens_per_s=100.0,
                             ranks_equal_prep=None if pol == "nofetch" else True,
                             tokens_equal_prep=None if pol == "nofetch" else True,
                             peak_decode=2 ** 30 * (3 + rnd), host_bytes=2 ** 31,
                             before={"loadavg1": 1.0, "other_pids": []},
                             after={"loadavg1": 2.5 if rnd else 1.0, "other_pids": others}))
    b = dict(R=2, budget=96, page_bytes=65536, runs=runs)
    if gz:
        with gzip.open(folder / "batch2.json.gz", "wt") as f:
            json.dump(b, f)
    else:
        (folder / "batch2.json").write_text(json.dumps(b))


@pytest.mark.parametrize("gz", [True, False])
def test_e_summarize_timed_steps_only(tmp_path, gz):
    """R11-1: misses per step = mean of step_misses[warmup:] per run, then mean over rounds (warm-up excluded)."""
    from analysis_v3 import e_summarize
    _fake_e_warm(tmp_path / "cfg", gz=gz)
    e_summarize.main([str(tmp_path / "cfg"), "--out", str(tmp_path / "o")])
    rows = {r["policy"]: r for r in csv.DictReader(open(tmp_path / "o/throughput.csv"))}
    assert float(rows["lru"]["misses_per_step"]) == pytest.approx(10.5)
    assert float(rows["ema"]["misses_per_step"]) == pytest.approx(8.0)
    assert float(rows["allmiss"]["misses_per_step"]) == pytest.approx(64.0)
    d2 = list(csv.DictReader(open(tmp_path / "o/d2_model.csv")))[0]
    assert float(d2["lru_misses_per_step"]) == pytest.approx(10.5)
    assert float(d2["ms_per_missed_page"]) == pytest.approx(15.0 / 64)
    assert float(rows["ema"]["upper_bound_gain_ms"]) == pytest.approx(2.5 * 15.0 / 64)
    # R11-2: runs with another process, clean flag, load = max(before, after)
    assert rows["ema"]["other_procs_seen"] == "1" and rows["ema"]["clean"] == "False"
    assert rows["lru"]["other_procs_seen"] == "0" and rows["lru"]["clean"] == "True"
    assert float(rows["lru"]["max_load"]) == pytest.approx(2.5)
    # R11-5: unchecked -> n/a; memory columns in GiB (max over rounds)
    assert rows["nofetch"]["ranks_equal"] == "n/a" and rows["nofetch"]["tokens_equal"] == "n/a"
    assert rows["lru"]["ranks_equal"] == "True"
    assert float(rows["lru"]["peak_decode_gib"]) == pytest.approx(4.0)
    assert float(rows["lru"]["host_tier_gib"]) == pytest.approx(2.0)


def test_e_bench_wait_clean(monkeypatch):
    """R11-6: wait_clean polls until no other process is on the card, and gives up after wait_max."""
    from analysis_v3 import e_bench
    states = iter([[7], [7], []])
    monkeypatch.setattr(e_bench, "gpu_state", lambda: {"other_pids": next(states)})
    monkeypatch.setattr(e_bench.time, "sleep", lambda s: None)
    assert e_bench.wait_clean(100, poll=1) >= 0
    monkeypatch.setattr(e_bench, "gpu_state", lambda: {"other_pids": [7]})
    t = iter(range(0, 10000, 50))
    monkeypatch.setattr(e_bench.time, "time", lambda: next(t))
    assert e_bench.wait_clean(100, poll=1) >= 100


def _fake_runs(folder, pols, rounds=3, steps=12, warm=2, timing=False):
    """lat = base + 0.01 * round (so paired differences are exact); breakdown phase 2 = policy time."""
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(json.dumps({"args": {"warmup": warm}}))
    runs = []
    for rnd in range(rounds):
        for pol, (base, m, pol_ms) in pols.items():
            lat = [base + 0.01 * rnd] * steps
            brk = [[0.5, 1.0, pol_ms, 0.2, 3.0, 0.0, 0.7]] * steps if timing else None
            runs.append(dict(R=2, round=rnd, policy=pol, delay_ns_per_byte=0.0, lat_ms=lat,
                             step_misses=[999] * warm + [m] * (steps - warm), page_step=[False] * steps,
                             misses=m * steps, tokens_per_s=1.0, ranks_equal_prep=True, tokens_equal_prep=True,
                             breakdown_ms=brk, mem_phases=dict(prefill_forward=2 ** 30, migration=2 ** 31,
                                                               decode_start=2 ** 29, decode_warmup=2 ** 30,
                                                               decode_steady=3 * 2 ** 30),
                             before={"loadavg1": 1.0, "other_pids": []}, after={"loadavg1": 1.0, "other_pids": []}))
    (folder / "batch2.json").write_text(json.dumps(dict(R=2, budget=96, page_bytes=65536, runs=runs)))


def test_r11_netgain(tmp_path):
    from analysis_v3 import r11_netgain
    # per page = (60 - 45) / 64 ms; EMA saves 4 misses -> 0.9375 ms; overhead 1.4 - 0.9 = 0.5 ms; predicted 0.4375
    _fake_runs(tmp_path / "tp", {"lru": (50.0, 10, 0), "ema": (49.5625, 6, 0), "allmiss": (60.0, 64, 0),
                                 "nofetch": (45.0, 1, 0)})
    _fake_runs(tmp_path / "tm", {"lru": (50.0, 10, 0.9), "ema": (50.0, 6, 1.4)}, timing=True)
    r11_netgain.main(["--pair", "x", str(tmp_path / "tp"), "2", str(tmp_path / "tm"), "--out", str(tmp_path / "o")])
    r = list(csv.DictReader(open(tmp_path / "o/netgain.csv")))[0]
    assert float(r["saved_transfer_ms"]) == pytest.approx(4 * 15 / 64)
    assert float(r["overhead_ms"]) == pytest.approx(0.5)
    assert float(r["predicted_net_ms"]) == pytest.approx(4 * 15 / 64 - 0.5)
    assert float(r["measured_ms"]) == pytest.approx(0.4375)
    assert r["consistent"] == "True"
    mem = {m["policy"]: m for m in csv.DictReader(open(tmp_path / "o/memory.csv"))}
    assert float(mem["lru"]["migration"]) == pytest.approx(2.0) and float(mem["ema"]["decode_steady"]) == pytest.approx(3.0)
    ph = list(csv.DictReader(open(tmp_path / "o/phases.csv")))
    assert len(ph) == 6 and float(ph[0]["gather"]) == pytest.approx(0.7)


def test_r11_netgain_without_timing(tmp_path):
    from analysis_v3 import r11_netgain
    _fake_runs(tmp_path / "tp", {"lru": (50.0, 10, 0), "ema": (49.5625, 6, 0), "allmiss": (60.0, 64, 0),
                                 "nofetch": (45.0, 1, 0)})
    r11_netgain.main(["--pair", "x", str(tmp_path / "tp"), "2", "-", "--out", str(tmp_path / "o")])
    r = list(csv.DictReader(open(tmp_path / "o/netgain.csv")))[0]
    assert float(r["implied_overhead_ms"]) == pytest.approx(4 * 15 / 64 - 0.4375)
    assert r["overhead_ms"] == "nan" and r["consistent"] == ""
