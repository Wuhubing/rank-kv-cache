"""validation/alloc_oracle.py on synthetic NPZs: same-allocation LRU/Bélády anchors."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import kvpolicy as kp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
import alloc_oracle as ao  # noqa: E402
from test_rank_control import write_docs  # noqa: E402

ALLOC = [10, 12, 14]


def setup(tmp_path, n=3):
    write_docs(tmp_path, n=n, L=3)
    rows, alloc_rows = [], []
    for i in range(n):
        R, S, rid = kp.load(tmp_path / f"d{i}.npz")
        for name, budgets, label in (("lru", 12, "lru"), ("belady", 12, "belady"),
                                     ("ema", dict(enumerate(ALLOC)), "alloc_ema")):
            for l, c in kp.doc_counts(R, S, budgets, name, alpha=0.5).items() if name == "ema" else \
                    kp.doc_counts(R, None, budgets, name).items():
                rows.append((label, rid, l, 12, c["accesses"], c["misses"], c["compulsory"]))
        lru_a = sum(c["misses"] for c in kp.doc_counts(R, None, dict(enumerate(ALLOC)), "lru").values())
        alloc_rows.append(("lru", 12, rid, "lru", lru_a, sum(ALLOC), " ".join(map(str, ALLOC))))
        alloc_rows.append(("best_hybrid", 12, rid, "x", lru_a - 5, 36, "12 12 12"))
    with open(tmp_path / "misses.csv", "w", newline="") as f:
        csv.writer(f).writerows([("config", "doc", "layer", "B", "accesses", "misses", "compulsory")] + rows)
    with open(tmp_path / "alloc.csv", "w", newline="") as f:
        csv.writer(f).writerows([("base", "B", "doc", "config", "misses", "slots", "per_layer")] + alloc_rows)
    (tmp_path / "frozen.json").write_text(json.dumps({"alloc": {"12": ALLOC}}))


def test_heldout_same_alloc_anchors(tmp_path):
    setup(tmp_path)
    out = ao.main(["heldout", str(tmp_path), "--out", str(tmp_path / "o"), "--misses", str(tmp_path / "misses.csv"),
                   "--frozen", str(tmp_path / "frozen.json"), "--workers", "2"])
    (r,) = out
    assert r["label"] == "alloc_ema" and r["n_docs"] == 3
    assert r["belady_same_alloc"] <= r["lru_same_alloc"]
    lu, pol = r["lru_uniform"], r["policy_misses"]
    assert r["G_uniform_anchors"] == pytest.approx((lu - pol) / (lu - r["belady_uniform"]))
    assert r["G_vs_alloc_belady"] == pytest.approx((lu - pol) / (lu - r["belady_same_alloc"]))
    assert r["G_within_alloc"] == pytest.approx(
        (r["lru_same_alloc"] - pol) / (r["lru_same_alloc"] - r["belady_same_alloc"]))
    # Bélády at the same per-layer budgets, recomputed directly
    tot = 0
    for i in range(3):
        R, _, _ = kp.load(tmp_path / f"d{i}.npz")
        tot += sum(c["misses"] for c in kp.doc_counts(R, None, dict(enumerate(ALLOC)), "belady").values())
    assert tot == r["belady_same_alloc"]
    assert (tmp_path / "o" / "alloc_anchors_by_doc.csv").exists()


def test_dev_uniform_allocation_reduces_to_uniform_anchors(tmp_path):
    setup(tmp_path)
    out = {r["label"]: r for r in ao.main(["dev", str(tmp_path), "--out", str(tmp_path / "o"),
                                           "--misses", str(tmp_path / "misses.csv"),
                                           "--alloc", str(tmp_path / "alloc.csv"), "--workers", "2"])}
    lru, hyb = out["alloc+lru"], out["alloc+best_hybrid"]
    assert lru["G_within_alloc"] == pytest.approx(0.0)
    assert hyb["belady_same_alloc"] == hyb["belady_uniform"] and hyb["lru_same_alloc"] == hyb["lru_uniform"]
    assert hyb["G_vs_alloc_belady"] == pytest.approx(hyb["G_uniform_anchors"])


def test_dev_rejects_inconsistent_lru_allocation(tmp_path):
    setup(tmp_path)
    rows = list(csv.reader(open(tmp_path / "alloc.csv")))
    rows[1][4] = str(int(rows[1][4]) + 1)
    with open(tmp_path / "alloc.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(ValueError, match="differ from replay"):
        ao.main(["dev", str(tmp_path), "--out", str(tmp_path / "o"), "--misses", str(tmp_path / "misses.csv"),
                 "--alloc", str(tmp_path / "alloc.csv")])
