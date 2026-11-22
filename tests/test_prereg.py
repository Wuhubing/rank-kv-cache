import json
import unittest
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from evaluate import success, table
from longbench_score import qa_f1_score
from preregister import PAGE_BYTES, pick

ROOT = Path(__file__).resolve().parents[1]

try:
    import numpy as np
except ImportError:
    np = None


def synthetic_counts():
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    repeat = {"lru": [100, 200], "belady": [40, 80], "rf": [95, 190], "full": [80, 170]}
    for name, values in repeat.items():
        for doc, r in zip(("a", "b"), values):
            counts[128][name][doc].update(misses=r + 10, repeat_misses=r, accesses=1000,
                                          compulsory_misses=10, steps=100)
    return counts


class PreregTests(unittest.TestCase):
    def test_page_bytes(self):
        self.assertEqual(PAGE_BYTES, 65536)

    def test_tie_break_prefers_smaller_values(self):
        misses = {(0, 0): 10, (0, 1): 9, (1, 0): 9, (2, 4): 9}
        self.assertEqual(pick(misses, list(misses)), (0, 1))
        self.assertEqual(pick(misses, [k for k in misses if k[1] == 0]), (1, 0))

    def test_longbench_f1(self):
        self.assertEqual(qa_f1_score("The cat.", "cat"), 1.0)
        self.assertAlmostEqual(qa_f1_score("a red cat sat", "the cat"), 0.5)
        self.assertEqual(qa_f1_score("", "cat"), 0)

    def test_success_and_table(self):
        counts = synthetic_counts()
        prereg = dict(primary_budget=128, budgets=[128], page_bytes=PAGE_BYTES,
                      bootstrap=dict(replicates=200, seed=0),
                      policies=[dict(name=n) for n in ("lru", "rf", "full", "belady")])
        result = success(counts, prereg, ["a", "b"])
        self.assertAlmostEqual(result["s1_full_vs_lru_repeat_reduction"], 50 / 300)
        self.assertTrue(result["s1_met"])
        self.assertEqual(result["s2_repeat_misses_saved_full_vs_rf"], 35)
        self.assertTrue(result["s2_met"])
        rows = {r["policy"]: r for r in table(counts, prereg, ["a", "b"], 0)}
        self.assertAlmostEqual(rows["full"]["G"], 50 / 180)
        self.assertEqual(rows["belady"]["G"], 1)
        self.assertAlmostEqual(rows["lru"]["h2d_mib_per_token"], 320 * 65536 / 200 / 2 ** 20)

    def test_preregistration_matches_split(self):
        path = ROOT / "splits/preregistration.json"
        if not path.exists():
            self.skipTest("no preregistration yet")
        prereg = json.loads(path.read_text())
        self.assertEqual(prereg["split_sha256"],
                         sha256((ROOT / "splits/longbench_v1.json").read_bytes()).hexdigest())
        names = [p["name"] for p in prereg["policies"]]
        self.assertEqual(names, ["fifo", "lru", "rank_only", "rf", "rrank", "full", "logistic",
                                 "belady"])
        self.assertEqual(prereg["budgets"], [128, 256])
        policies = {p["name"]: p for p in prereg["policies"]}
        self.assertEqual(policies["rf"]["delta"], 0)
        self.assertEqual(policies["rrank"]["beta"], 0)

    @unittest.skipIf(np is None, "numpy is not installed")
    def test_logistic_round_trip(self):
        from gap_analysis import FEATURES, Logistic
        rng = np.random.default_rng(1)
        X = rng.normal(size=(500, len(FEATURES)))
        y = (X[:, 0] + X[:, 3] > 0).astype(float)
        model = Logistic(range(len(FEATURES))).fit(X, y)
        again = Logistic.from_dict(json.loads(json.dumps(model.to_dict())))
        np.testing.assert_allclose(again.predict(X), model.predict(X))


if __name__ == "__main__":
    unittest.main()
