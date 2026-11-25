"""Statistics helpers of analysis_v2.s4_dev on synthetic data (followup-v3 A6)."""
import math
from collections import defaultdict

import numpy as np
import pytest

import kvpolicy as kp
from analysis_v2 import s4_dev as s4


def table(cells):
    """cells: {(cfg, B): {doc: misses}} -> s4_dev's M layout (accesses 1000, compulsory 10)."""
    M = defaultdict(lambda: defaultdict(lambda: np.zeros(3, np.int64)))
    for key, per in cells.items():
        for d, m in per.items():
            M[key][d] = np.array([1000, m, 10])
    return M


def test_g_of():
    assert s4.g_of(100, 100, 50) == 0.0
    assert s4.g_of(100, 50, 50) == 1.0
    assert s4.g_of(100, 75, 50) == 0.5
    assert s4.g_of(100, 125, 50) == -0.5
    assert math.isnan(s4.g_of(80, 70, 80))


def test_lodo_picks_on_the_other_documents(monkeypatch):
    monkeypatch.setitem(s4.FAMILIES, "fam", ["a", "b"])
    # doc x prefers a strongly, y and z prefer b slightly: leaving x out picks b; leaving y or z out picks a
    M = table({("a", 96): {"x": 10, "y": 31, "z": 31}, ("b", 96): {"x": 50, "y": 30, "z": 30}})
    misses, chosen = s4.lodo(M, ["x", "y", "z"], "fam", 96)
    assert chosen == ["b", "a", "a"]
    assert list(misses) == [50, 31, 31]


def test_lodo_ties_broken_by_name(monkeypatch):
    monkeypatch.setitem(s4.FAMILIES, "fam", ["b", "a"])
    M = table({("a", 96): {"x": 5, "y": 5}, ("b", 96): {"x": 5, "y": 5}})
    _, chosen = s4.lodo(M, ["x", "y"], "fam", 96)
    assert chosen == ["a", "a"]


def test_iso_capacity_interpolates_on_the_fine_lru_grid():
    docs = ["x", "y"]
    cells = {("lru", b): {"x": 3000 - 2 * b, "y": 3000 - 2 * b} for b in s4.FINE_LRU}
    M = table(cells)
    # total LRU misses at B is 2 * (3000 - 2B); 130 is off the grid (interpolated)
    assert s4.iso_capacity(M, docs, 2 * (3000 - 2 * 128)) == pytest.approx(128)
    assert s4.iso_capacity(M, docs, 2 * (3000 - 2 * 130)) == pytest.approx(130)
    assert math.isnan(s4.iso_capacity(M, docs, 10 ** 6))          # more misses than LRU at the smallest B
    assert s4.iso_capacity(M, docs, 0) == float("inf")             # fewer than LRU at the largest B


def test_boot_idx_is_deterministic_and_paired():
    a, b = s4.boot_idx(14), s4.boot_idx(14)
    assert a.shape == (s4.REPS, 14) and (a == b).all()
    assert a.min() >= 0 and a.max() < 14


def test_summarize_rows_point_estimate_and_ci():
    rng = np.random.default_rng(1)
    n = 12
    lru = rng.integers(900, 1100, n).astype(float)
    opt = lru - rng.integers(300, 400, n)
    pol = lru - 0.3 * (lru - opt) + rng.normal(0, 5, n)
    docs = [f"d{i}" for i in range(n)]
    M = table({("lru", 96): dict(zip(docs, lru.astype(int)))})
    per = {("lru", 96): lru, ("belady", 96): opt, ("pol", 96): pol}
    rows = {r["policy"]: r for r in s4.summarize_rows(per, docs, M, [96])}
    r = rows["pol"]
    assert r["G"] == pytest.approx(s4.g_of(lru.sum(), pol.sum(), opt.sum()))
    assert r["G_lo"] <= r["G"] <= r["G_hi"]
    assert 0.25 < r["G_lo"] and r["G_hi"] < 0.35
    assert r["docs_better_than_lru"] == n
    assert rows["lru"]["G"] == 0.0 and rows["belady"]["G"] == 1.0


def test_kvpolicy_bootstrap_degenerate_and_covering():
    lo, hi = kp.bootstrap([100] * 5, [70] * 5, [0] * 5)
    assert lo == pytest.approx(0.3) and hi == pytest.approx(0.3)
    rng = np.random.default_rng(0)
    lru = rng.integers(900, 1100, 20)
    opt = lru - 300
    pol = lru - rng.integers(50, 130, 20)
    lo, hi = kp.bootstrap(lru, pol, opt)
    g = (lru.sum() - pol.sum()) / (lru.sum() - opt.sum())
    assert lo < g < hi


def test_audit_compare_reports_missing_reference_rows():
    from analysis_v2.audit_v2 import compare
    ind = {("d", 0, "lru", 96): (10, 3, 2), ("d", 1, "lru", 96): (10, 4, 2), ("d", 2, "lru", 96): (10, 4, 2)}
    ref = {("d", 0, "lru", 96): (10, 3, 2), ("d", 1, "lru", 96): (10, 5, 2)}
    bad, missing, checked = compare(ref, ind)
    assert checked == 2
    assert bad == [(("d", 1, "lru", 96), (10, 5, 2), (10, 4, 2))]
    assert missing == [("d", 2, "lru", 96)]


def test_fix_v3_read_later_file_replaces(tmp_path):
    from analysis_v2.fix_v3 import read
    hdr = "doc,layer,config,B,accesses,misses,compulsory\n"
    (tmp_path / "a.csv").write_text(hdr + "d,0,lru,96,10,5,1\nd,1,lru,96,10,5,1\nd,0,score,96,10,4,1\n")
    (tmp_path / "b.csv").write_text(hdr + "d,0,lru,96,10,5,1\nd,1,lru,96,10,5,1\n")
    M = read([tmp_path / "a.csv", tmp_path / "b.csv"])
    assert M["lru", 96]["d"][1] == 10 and M["score", 96]["d"][1] == 4
