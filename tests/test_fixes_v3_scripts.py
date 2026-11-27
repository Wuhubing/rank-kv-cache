"""followup-v3 part A scripts: provenance collection and the degenerate-generation split."""
import csv
import gzip
import json
import subprocess
import sys

import numpy as np


def test_provenance_strip_and_identity(tmp_path):
    from analysis_v2 import provenance_v3 as pv
    m = pv.strip({"rid": "a", "generated": [1, 2, 3]})
    assert m == {"rid": "a", "generated_len": 3, "generated_sha256": m["generated_sha256"]} and len(m["generated_sha256"]) == 64
    for d in ("x", "y"):
        (tmp_path / d).mkdir()
        np.savez(tmp_path / d / "t.npz", ranked=np.arange(6).reshape(3, 2), scores=np.ones((3, 2)), rid=np.array("t"))
        (tmp_path / d / "t.json").write_text(json.dumps({"rid": "t", "generated": [1, 2]}))
    assert pv.identity(tmp_path, "x", "y")[0].endswith("1 documents compared, 0 differ")
    np.savez(tmp_path / "y" / "t.npz", ranked=np.arange(6).reshape(3, 2) + 1, scores=np.ones((3, 2)), rid=np.array("t"))
    assert pv.identity(tmp_path, "x", "y")[0].endswith("1 differ")


def test_degenerate_split(tmp_path):
    rows = []
    rng = np.random.default_rng(0)
    for d in range(4):
        for l in range(2):
            lru = 100 + int(rng.integers(0, 5))
            for c, m in (("lru", lru), ("belady", lru - 40), ("score", lru - 8), ("ema:0.5", lru - 11), ("ema:0.3", lru - 10),
                         ("ema_allrank:0.5", lru - 10), ("infllm_eq1:0.1", lru - 1), ("infllm_lrus:0.1", lru - 1)):
                for B in (80, 96):
                    rows.append((f"doc{d}", l, c, B, 1000, m, 10))
    with gzip.open(tmp_path / "m.csv.gz", "wt") as f:
        w = csv.writer(f)
        w.writerow(["doc", "layer", "config", "B", "accesses", "misses", "compulsory"])
        w.writerows(rows)
    with open(tmp_path / "man.jsonl", "w") as f:
        for d in range(4):
            f.write(json.dumps({"rid": f"doc{d}", "stop": "loop" if d == 0 else "eos"}) + "\n")
    subprocess.run([sys.executable, "-m", "analysis_v2.degenerate_v3", "--out", str(tmp_path / "o"), "--setting", "s",
                    str(tmp_path / "m.csv.gz"), str(tmp_path / "man.jsonl")], check=True, capture_output=True)
    out = list(csv.DictReader(open(tmp_path / "o/summary.csv")))
    sub = {(r["subset"], r["policy"], r["B"]): r for r in out}
    assert sub["all", "ema(lodo)", "96"]["n_docs"] == "4"
    assert sub["without_loop_stops", "ema(lodo)", "96"]["n_docs"] == "3"
    assert sub["loop_stops_only", "score", "96"]["n_docs"] == "1"
