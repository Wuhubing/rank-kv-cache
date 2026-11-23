"""Trace schema v2: optional selector scores next to the v1 ranking."""
import gzip
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analysis import analyze
from replay import check_scores, compare, read_trace

ROOT = Path(__file__).resolve().parents[1]
V1_KEYS = ("request_id", "layer_id", "decode_step", "ranked_page_ids")


def synthetic_rows(seed=3, requests=2, layers=3, steps=12, top_k=4, decimals=3):
    """Rows like the notebook writes: page scores over all completed pages, top-K ranking."""
    rng = random.Random(seed)
    rows = []
    for r in range(requests):
        for t in range(steps):
            for layer in range(layers):
                n = min(6 + t, 40)
                every = [round(rng.gauss(0, 3), decimals) for _ in range(n)]
                ranked = sorted(range(n), key=lambda p: (-every[p], p))[:top_k]
                rows.append(dict(request_id=f"task/band/doc{r}", layer_id=layer, decode_step=t,
                                 ranked_page_ids=ranked,
                                 ranked_scores=[every[p] for p in ranked], page_scores=every))
    return rows


def write(path, rows, gz=False):
    text = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as out:
            out.write(text)
    else:
        Path(path).write_text(text, encoding="utf-8")


class SchemaV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.rows = synthetic_rows()
        write(self.dir / "v2.jsonl.gz", self.rows, gz=True)
        write(self.dir / "v1.jsonl", [{k: r[k] for k in V1_KEYS} for r in self.rows])

    def tearDown(self):
        self.tmp.cleanup()

    def test_v2_projection_matches_v1(self):
        v1 = read_trace(self.dir / "v1.jsonl")
        v2 = read_trace(self.dir / "v2.jsonl.gz")
        self.assertEqual(v1, v2)
        self.assertEqual(compare(v1, [4, 6, 8], ["lru", "full", "belady"], delta=8, beta=2),
                         compare(v2, [4, 6, 8], ["lru", "full", "belady"], delta=8, beta=2))

    def test_scores_returned_per_step(self):
        streams, scores = read_trace(self.dir / "v2.jsonl.gz", scores=True)
        self.assertEqual(streams.keys(), scores.keys())
        row = self.rows[0]
        ranked, every = scores[row["request_id"], row["layer_id"]][0]
        self.assertEqual(len(ranked), len(row["ranked_page_ids"]))
        self.assertEqual(len(every), len(row["page_scores"]))
        for a, b in zip(every, row["page_scores"]):
            self.assertAlmostEqual(a, b, places=3)
        _, v1_scores = read_trace(self.dir / "v1.jsonl", scores=True)
        self.assertTrue(all(s is None for steps in v1_scores.values() for s in steps))

    def test_topk_only_rows(self):
        row = dict(self.rows[5])
        del row["page_scores"]
        ranked, every = check_scores(row["ranked_page_ids"], row)
        self.assertIsNone(every)
        self.assertEqual(len(ranked), 4)

    def test_mixed_v1_and_v2_streams(self):
        rows = [dict(r) for r in self.rows]
        for r in rows:
            if r["request_id"].endswith("doc1"):
                del r["ranked_scores"], r["page_scores"]
        write(self.dir / "mixed.jsonl", rows)
        streams, scores = read_trace(self.dir / "mixed.jsonl", scores=True)
        self.assertEqual(streams, read_trace(self.dir / "v1.jsonl"))
        self.assertIsNone(scores["task/band/doc1", 0][0])
        self.assertIsNotNone(scores["task/band/doc0", 0][0])

    def test_validation(self):
        base = self.rows[30]
        pages = base["ranked_page_ids"]

        def broken(**change):
            row = dict(base, **change)
            return {k: v for k, v in row.items() if v is not None}

        bad = [
            broken(ranked_scores=base["ranked_scores"][:-1]),               # misaligned
            broken(ranked_scores=base["ranked_scores"][::-1]),              # increasing
            broken(ranked_scores=["x"] * len(pages)),                       # not numbers
            broken(ranked_scores=[float("nan")] * len(pages)),              # not finite
            broken(ranked_scores=None),                                     # page_scores alone
            broken(page_scores=base["page_scores"][:max(pages)]),           # too short
            broken(page_scores=[s + 1 if i == pages[0] else s
                                for i, s in enumerate(base["page_scores"])]),  # disagrees
            broken(page_scores=base["page_scores"] + [1e6]),                # unselected above
        ]
        for i, row in enumerate(bad):
            with self.subTest(i=i), self.assertRaises(ValueError):
                check_scores(row["ranked_page_ids"], row)
            path = self.dir / f"bad{i}.jsonl"
            write(path, [row])
            with self.subTest(i=i, via="read_trace"), self.assertRaisesRegex(ValueError, "line 1"):
                read_trace(path)

    def test_ties_at_the_boundary_are_valid(self):
        row = dict(request_id="r", layer_id=0, decode_step=0, ranked_page_ids=[1, 0],
                   ranked_scores=[2.0, 1.0], page_scores=[1.0, 2.0, 1.0])
        self.assertIsNotNone(check_scores(row["ranked_page_ids"], row))

    def test_analysis_identical_on_v1_and_v2(self):
        v1 = analyze(read_trace(self.dir / "v1.jsonl"), (1, 2), samples=("selected", "candidate"),
                     budget=6, replicates=50)
        v2 = analyze(read_trace(self.dir / "v2.jsonl.gz"), (1, 2), samples=("selected", "candidate"),
                     budget=6, replicates=50)
        self.assertEqual(v1, v2)

    def test_replay_cli_reads_gzip_v2(self):
        out = subprocess.run([sys.executable, str(ROOT / "replay.py"), str(self.dir / "v2.jsonl.gz"),
                              "--budgets", "6", "--no-save"], capture_output=True, text=True,
                             cwd=ROOT, check=True).stdout
        ref = subprocess.run([sys.executable, str(ROOT / "replay.py"), str(self.dir / "v1.jsonl"),
                              "--budgets", "6", "--no-save"], capture_output=True, text=True,
                             cwd=ROOT, check=True).stdout
        self.assertEqual(out, ref)

    def test_handwritten_v1_example_still_reads(self):
        self.assertIn(("toy", 0), read_trace(ROOT / "examples/handwritten.jsonl"))


if __name__ == "__main__":
    unittest.main()
