import random
import unittest

from gap_analysis import FEATURES, History, annotate, gap_heuristic, replay_scored
from replay import replay_stream

try:
    import numpy as np
except ImportError:
    np = None


class GapTests(unittest.TestCase):
    def test_unscored_and_constant_score_replay_is_lru(self):
        rng = random.Random(3)
        for _ in range(100):
            steps = [tuple(rng.sample(range(10), 3)) for _ in range(rng.randint(2, 20))]
            lru = replay_stream(steps, 5, "lru")
            for score in (None, lambda c, t, h: [0] * len(c)):
                misses, first, accesses = replay_scored(steps, 5, History(0, 0), score)
                self.assertEqual((misses, first, accesses),
                                 (lru["misses"], lru["compulsory_misses"], lru["accesses"]))

    def test_features_by_hand(self):
        h = History(layer=3, prompt_tokens=200)
        for t, pages in enumerate([(5, 6), (6,), (7,)]):
            h.observe_dropouts(t, set(pages))
            h.record(t, pages)
        h.observe_dropouts(3, {5})
        row = dict(zip(FEATURES, h.features(6, 3)))
        # Page 6: selected at t=0 (rank 2 of 2) and t=1 (rank 1 of 1), dropped at t=2.
        self.assertEqual(row, dict(
            recency=2, freq8=2, freq32=2, total=2, last_rank=1, last_rank_frac=1.0,
            last_gap=1, has_gap=1.0, mean_gap=1.0, drop_interval=0, has_drop=0.0,
            rel_pos=0.5, sink=0.0, from_end=5, near_end=1.0, layer=3))
        # Page 5 dropped at t=1 and is back at t=3; page 7 dropped at t=3.
        self.assertEqual(h.drops[5], [1])
        self.assertEqual(h.drops[7], [3])
        self.assertEqual(gap_heuristic([6, 7], 3, h), [-2, float("-inf")])

    def test_bad_evictions_by_hand(self):
        # Budget 2, one page per step. At t=2 LRU evicts page 0 (next use t=3) and keeps
        # page 1 (never used again): bad. At t=3 it evicts page 1 and keeps 2: both unused.
        streams = {("doc/4-8k/a", 0): [(0,), (1,), (2,), (0,)]}
        summary, victim_bins, bad_bins, by_doc = annotate(streams, {"doc/4-8k/a": 0}, 2)
        self.assertEqual((summary["decisions"], summary["victims"], summary["bad_victims"]),
                         (2, 2, 1))
        self.assertEqual(victim_bins, {"1": 1, "never": 1})
        self.assertEqual(bad_bins, {"1": 1})

    @unittest.skipIf(np is None, "numpy is not installed")
    def test_models(self):
        from gap_analysis import Logistic, Tree, auc
        self.assertEqual(auc(np.array([1., 2, 3, 4]), np.array([0, 0, 1, 1])), 1.0)
        self.assertEqual(auc(np.array([1., 1, 1, 1]), np.array([0, 1, 0, 1])), 0.5)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4000, len(FEATURES)))
        y = (X[:, 2] > 0.3).astype(float)
        tree = Tree(max_depth=1, min_leaf=50).fit(X, y)
        self.assertEqual(tree.root["feature"], 2)
        self.assertAlmostEqual(tree.root["threshold"], 0.3, delta=0.1)
        p = Logistic(range(len(FEATURES))).fit(X, y).predict(X)
        self.assertGreater(auc(p, y), 0.95)


if __name__ == "__main__":
    unittest.main()
