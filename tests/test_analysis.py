import random
import unittest
from collections import Counter
from pathlib import Path

from analysis import analyze, observations, quartile, recency_bin, stratified_effect, tally
from replay import read_trace

ROOT = Path(__file__).resolve().parents[1]


def synthetic(seed, keep, rank_by=None, requests=30, steps=40, pages=48, k=8):
    """Each step keeps pages from the previous step with probability keep(rank, freq).

    rank_by=None ranks each selection randomly. Otherwise pages are ranked by
    rank_by(freq), highest first, so rank can be confounded with frequency.
    """
    rng = random.Random(seed)
    streams = {}
    for r in range(requests):
        trace = []
        for t in range(steps):
            def freq(p):
                return sum(p in trace[s] for s in range(max(0, t - 8), t))
            chosen = []
            if trace:
                previous = trace[-1]
                prior = len(trace) - 1
                for rank, page in enumerate(previous, 1):
                    c = sum(page in trace[s] for s in range(max(0, prior - 8), prior))
                    if rng.random() < keep(rank, c):
                        chosen.append(page)
            pool = [p for p in range(pages) if p not in chosen]
            chosen += rng.sample(pool, k - len(chosen))
            if rank_by is None:
                rng.shuffle(chosen)
            else:
                chosen.sort(key=lambda p: (-rank_by(freq(p)), rng.random()))
            trace.append(tuple(chosen))
        streams[f"synthetic/{r}", 0] = trace
    return streams


def pooled(streams, horizon=1, replicates=400):
    effects, _, _ = analyze(streams, (1, 2, 4, 8), replicates=replicates, seed=1)
    return next(r for r in effects if r["group"] == "all" and r["horizon"] == horizon)


class AnalysisTests(unittest.TestCase):
    def test_hand_features(self):
        rows = list(observations(read_trace(ROOT / "examples/handwritten.jsonl"), (1, 2)))
        got = [(r["decode_step"], r["page_id"], r["rank"], r["quartile"], r["recency"],
                r["frequency"], r["reuse_1"], r["reuse_2"]) for r in rows]
        self.assertEqual(got, [(0, 0, 1, "top", None, 0, False, False),
                               (0, 1, 2, "bottom", None, 0, True, True),
                               (1, 1, 1, "top", 1, 1, True, True),
                               (1, 2, 2, "bottom", None, 0, False, False)])
        # Steps 2 and 3 lack two later steps, and no step has eight.
        self.assertEqual(list(observations(read_trace(ROOT / "examples/handwritten.jsonl"))), [])

    def test_frequency_window_and_bins(self):
        steps = [(5,)] * 12 + [(1, 2)] * 9
        rows = [r for r in observations({("a", 0): steps}, (8,)) if r["page_id"] == 5]
        self.assertEqual([r["frequency"] for r in rows], [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8])
        self.assertEqual([recency_bin(x) for x in (None, 1, 2, 3, 4, 5, 8, 9)],
                         ["new", "1", "2", "3-4", "3-4", "5-8", "5-8", ">8"])
        self.assertEqual([quartile(r, 64) for r in (1, 16, 17, 48, 49, 64)],
                         ["top", "top", "middle", "middle", "bottom", "bottom"])
        self.assertIsNone(quartile(1, 1))

    def test_stratified_weights(self):
        # Stratum A: 1/2 vs 0/2 over 4 pages. Stratum B: 3/4 vs 3/4 over 8 pages. C lacks bottom.
        tables = [{"A": [2, 1, 2, 0], "B": [4, 3, 4, 3]}, {"C": [5, 5, 0, 0]}]
        effect, used, top, bottom = stratified_effect(tables)
        self.assertAlmostEqual(effect, 4 * 0.5 / 12)
        self.assertEqual((used, top, bottom), (2, 9 / 11, 3 / 6))

    def test_detects_injected_rank_signal(self):
        streams = synthetic(3, lambda rank, c: 0.8 if rank <= 2 else 0.3)
        row = pooled(streams)
        self.assertGreater(row["ci_low"], 0.2)
        self.assertEqual(row["clusters"], 30)

    def test_no_signal_interval_covers_zero(self):
        for seed in (4, 5):
            row = pooled(synthetic(seed, lambda rank, c: 0.5))
            with self.subTest(seed=seed):
                self.assertLess(row["ci_low"], 0)
                self.assertGreater(row["ci_high"], 0)

    def test_strata_remove_frequency_confound(self):
        # Reuse depends only on frequency, and rank follows frequency. A large page pool
        # makes random refills rarely reuse a page. With a small pool, a step full of
        # frequent pages leaves fewer refill slots, so relative rank gains a real effect.
        streams = synthetic(6, lambda rank, c: min(0.9, 0.1 + 0.2 * c), rank_by=lambda c: c,
                            pages=4000)
        row = pooled(streams)
        self.assertGreater(row["raw_top"] - row["raw_bottom"], 0.15)
        self.assertLess(row["ci_low"], 0)
        self.assertGreater(row["ci_high"], 0)

    def test_groups_and_single_cluster(self):
        streams = {("qa/1", 0): [(1, 2, 3, 4)] * 10, ("sum/2", 3): [(4, 3, 2, 1)] * 10}
        effects, curve, rows = analyze(streams, (1, 8), by=("task", "layer-group"),
                                       layer_groups=2, replicates=50)
        self.assertEqual(Counter(r["group"] for r in effects),
                         Counter({"all": 2, "task qa": 2, "task sum": 2,
                                  "layers 0-1": 2, "layers 2-3": 2}))
        single = next(r for r in effects if r["group"] == "task qa")
        self.assertIsNone(single["ci_low"])
        self.assertEqual(len(rows), 2 * 2 * 4)
        self.assertTrue(all(r["reuse_rate"] == 1 for r in curve))
        with self.assertRaises(ValueError):
            analyze(streams, (0,))
        stream_level = tally(rows, 1, cluster="stream")
        self.assertEqual(sorted(stream_level), [("qa/1", 0), ("sum/2", 3)])


if __name__ == "__main__":
    unittest.main()
