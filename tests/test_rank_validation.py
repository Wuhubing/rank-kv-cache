import unittest

from rank_validation import audit_stream, decision_audit, matched_beta


class RankValidationTests(unittest.TestCase):
    def setUp(self):
        # At t=2, matched-delta baseline evicts top-ranked page 6 on the LRU tie.
        # A rank bonus retains 6, which is selected again immediately at t=3.
        self.steps = [(6, 5), (7,), (8,), (6,)]
        self.streams = {("toy", 0): self.steps}

    def test_matched_beta_isolates_delta(self):
        rows, docs = matched_beta(self.streams, [3], [0, 1], [0], top_k=4,
                                  replicates=20, seed=0)
        by_delta = {r["delta"]: r for r in rows}
        self.assertEqual(by_delta[0]["repeat_misses"], 1)
        self.assertEqual(by_delta[1]["repeat_misses"], 0)
        self.assertEqual(by_delta[1]["repeat_saved"], 1)
        self.assertEqual(by_delta[1]["repeat_reduction"], 1)
        self.assertEqual(len(docs), 2)

    def test_one_step_audit_explains_saved_page(self):
        counts = audit_stream(self.steps, budget=3, beta=0, deltas=[0, 1], top_k=4)[1]
        self.assertEqual(counts["eviction_events"], 2)
        self.assertEqual(counts["changed_events"], 1)
        self.assertEqual(counts["changed_slots"], 1)
        self.assertEqual(counts["top_saved"], 1)
        self.assertEqual(counts["top_newly_evicted"], 0)
        self.assertEqual(counts["avoided_near_reuse_1"], 1)
        self.assertEqual(counts["avoided_near_reuse_8"], 1)

    def test_audit_aggregates_by_document(self):
        streams = {("a", 0): self.steps, ("a", 1): self.steps, ("b", 0): self.steps}
        rows, docs = decision_audit(streams, [3], [0, 1], [0], top_k=4)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["eviction_events"], 6)
        self.assertEqual(rows[0]["changed_event_rate"], 0.5)
        self.assertEqual({r["document"] for r in docs}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
