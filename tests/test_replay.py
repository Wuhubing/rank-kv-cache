import itertools
import json
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

from replay import compare, read_trace, replay_stream

ROOT = Path(__file__).resolve().parents[1]


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.streams = read_trace(ROOT / "examples/handwritten.jsonl")
        self.steps = self.streams["toy", 0]

    def test_hand_calculated(self):
        rows = compare(self.streams, [2, 3, 4])
        self.assertEqual([r["misses"] for r in rows], [5, 5, 5, 4, 4, 4])
        for r in rows:
            self.assertEqual((r["accesses"], r["compulsory_misses"], r["repeat_accesses"]), (8, 4, 4))
            self.assertEqual(r["repeat_miss_rate"], (r["misses"] - 4) / 4)

    def test_equal_budget_and_unlimited(self):
        sets = [set(s) for s in self.steps]
        expected = len(sets[0]) + sum(len(b-a) for a, b in zip(sets, sets[1:]))
        for policy in ("lru", "belady"):
            self.assertEqual(replay_stream(self.steps, 2, policy)["misses"], expected)
            self.assertEqual(replay_stream(self.steps, 99, policy)["repeat_misses"], 0)
            with self.assertRaises(ValueError):
                replay_stream(self.steps, 1, policy)

    def test_hits_ahead_of_fills_and_rank_invariance(self):
        # At t=2 page 1 hits and 2 fills. At t=3 both are eligible. LRU retains 1.
        steps = [(0, 1), (1, 3), (1, 2), (3, 4), (1, 4)]
        self.assertEqual(replay_stream(steps, 3, "lru")["misses"], 5)
        for policy in ("lru", "belady"):
            self.assertEqual(replay_stream(steps, 3, policy),
                             replay_stream([s[::-1] for s in steps], 3, policy))

    def test_page_id_tie(self):
        # Among equally recent fills 0 and 1, prefer to retain smaller ID 0.
        self.assertEqual(replay_stream([(1, 0), (2, 3), (0, 2)], 3, "lru")["misses"], 4)

    def test_cold_start_per_layer_and_request(self):
        streams = {("r1", 0): [(7,)], ("r1", 1): [(7,)], ("r2", 0): [(7,)]}
        for row in compare(streams, [1]):
            self.assertEqual(row["compulsory_misses"], 3)
            self.assertIsNone(row["repeat_miss_rate"])

    def test_schema_validation(self):
        row = dict(request_id="a", layer_id=0, decode_step=0, ranked_page_ids=[1])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.jsonl"
            for patch in ({"ranked_page_ids": [0, 0]}, {"decode_step": 1}, {"layer_id": -1},
                          {"ranked_page_ids": [True]}, {"request_id": ""}):
                with self.subTest(patch=patch), self.assertRaises(ValueError):
                    path.write_text(json.dumps({**row, **patch}))
                    read_trace(path)

    def test_oracle_against_exhaustive_admissions(self):
        # Independent optimum enumerates every legal eviction for every four-step trace.
        choices = list(itertools.combinations(range(4), 2))
        for steps in itertools.product(choices, repeat=4):
            @lru_cache(None)
            def solve(t, resident):
                if t == len(steps):
                    return 0
                want = set(steps[t])
                old = resident - want
                keep = min(len(old), 3 - len(want))
                return len(want - resident) + min(
                    solve(t + 1, frozenset(want.union(subset)))
                    for subset in itertools.combinations(old, keep))
            self.assertEqual(replay_stream(steps, 3, "belady")["misses"], solve(0, frozenset()))

    def test_standalone_notebook_matches_local_replay(self):
        notebook = json.loads((ROOT / "notebooks/first_trace.ipynb").read_text())
        cell, = [c for c in notebook["cells"] if c["id"] == "replay"]
        local = (ROOT / "replay.py").read_text().split("\ndef main():")[0]
        self.assertEqual("".join(cell["source"]).strip(), local.strip())
