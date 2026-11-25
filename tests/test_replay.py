import itertools
import json
import random
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path

from replay import compare, iter_replay, parse_policy, read_trace, replay_stream

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

    def test_score_with_zero_weights_matches_lru_step_by_step(self):
        # Every legal four-step trace over five pages with two or three selections per step.
        choices = [c for n in (2, 3) for c in itertools.permutations(range(5), n)]
        rng = random.Random(0)
        traces = [list(self.steps), [(1, 0), (2, 3), (0, 2)]]
        traces += [[rng.choice(choices) for _ in range(rng.randint(1, 12))] for _ in range(400)]
        for steps in traces:
            for budget in (3, 4, 5):
                lru = list(iter_replay(steps, budget, "lru"))
                for top_k in (None, 1, 8):
                    self.assertEqual(list(iter_replay(steps, budget, "score", top_k=top_k)), lru)

    def test_fifo_ignores_hits(self):
        # Page 0 is hit at t=2. LRU then evicts 1, FIFO evicts 0 and misses it at t=4.
        steps = [(0,), (1,), (0,), (2,), (0,)]
        self.assertEqual(replay_stream(steps, 2, "lru")["misses"], 3)
        self.assertEqual(replay_stream(steps, 2, "fifo")["misses"], 4)

    def test_rank_only_evicts_worst_last_rank(self):
        # Page 1 ranks first at t=0. LRU breaks the tie toward evicting larger ID 1.
        steps = [(1, 0), (2, 3), (1, 2)]
        self.assertEqual(replay_stream(steps, 3, "lru")["misses"], 5)
        self.assertEqual(replay_stream(steps, 3, "rank")["misses"], 4)

    def test_score_rank_bonus_and_frequency(self):
        # K=4, so only rank 1 is top quartile. delta keeps page 6, which LRU evicts.
        steps = [(6, 5), (7,), (8,), (6,)]
        self.assertEqual(replay_stream(steps, 3, "lru", top_k=4)["misses"], 5)
        self.assertEqual(replay_stream(steps, 3, "score", delta=1, top_k=4)["misses"], 4)
        # K=8 puts ranks 1-2 in the top quartile, so pages 5 and 6 tie and 6 leaves.
        self.assertEqual(replay_stream(steps, 3, "score", delta=1, top_k=8)["misses"], 5)
        # Page 0 was selected at t=0 and t=1. At t=3, beta=2 scores it 1+4 against 2+2
        # for the fresher page 1. beta=1 would tie at 3 and fall back to LRU.
        steps = [(0,), (0,), (1,), (2,), (0,)]
        self.assertEqual(replay_stream(steps, 2, "lru")["misses"], 4)
        self.assertEqual(replay_stream(steps, 2, "score", beta=2)["misses"], 3)
        self.assertEqual(replay_stream(steps, 2, "score", beta=1)["misses"], 4)
        with self.assertRaises(ValueError):
            replay_stream(steps, 2, "score", beta=-1)

    def test_oracle_lower_bounds_every_policy(self):
        choices = list(itertools.combinations(range(4), 2))
        policies = ["lru", "fifo", "rank", "rf", "rrank", "full", "score:8:2"]
        for steps in itertools.product(choices, repeat=4):
            rows = compare({("x", 0): list(steps)}, [3], [*policies, "belady"], delta=2, beta=1)
            oracle = rows[-1]["misses"]
            self.assertTrue(all(r["misses"] >= oracle for r in rows))

    def test_policy_specs(self):
        self.assertEqual(parse_policy("rf", 3, 2), ("rf", "score", 0, 2))
        self.assertEqual(parse_policy("rrank", 3, 2), ("rrank", "score", 3, 0))
        self.assertEqual(parse_policy("full", 3, 2), ("full", "score", 3, 2))
        self.assertEqual(parse_policy("score:4:0.5"), ("score:4:0.5", "score", 4, 0.5))
        for bad in ("mru", "score:1", "score:a:b"):
            with self.assertRaises(ValueError):
                parse_policy(bad)
        rows = compare(self.streams, [2], ["fifo", "full"], delta=1, beta=1)
        self.assertEqual([(r["policy"], r["delta"], r["top_k"]) for r in rows],
                         [("fifo", None, 2), ("full", 1, 2)])

    def test_standalone_notebook_matches_local_replay(self):
        notebook = json.loads((ROOT / "notebooks/first_trace.ipynb").read_text())
        cell, = [c for c in notebook["cells"] if c["id"] == "replay"]
        local = (ROOT / "replay.py").read_text().split("\ndef main():")[0]
        self.assertEqual("".join(cell["source"]).strip(), local.strip())
