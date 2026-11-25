import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from posthoc_rank_increment import POLICIES, contrasts, per_document
from replay import read_trace, replay_stream

ROOT = Path(__file__).resolve().parents[1]


def synthetic_trace(path, docs=3, layers=2, steps=40, pages=12, k=4, seed=1):
    rng = random.Random(seed)
    with path.open("w") as f:
        for d in range(docs):
            for layer in range(layers):
                for t in range(steps):
                    # A small hot set plus noise, so policies disagree at a tight budget.
                    hot = rng.sample(range(3), 2)
                    rest = rng.sample([p for p in range(pages) if p not in hot], k - 2)
                    f.write(json.dumps(dict(request_id=f"doc{d}", layer_id=layer,
                                            decode_step=t, ranked_page_ids=hot + rest)) + "\n")


class PosthocTests(unittest.TestCase):
    def test_contrasts_decompose_and_match_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            synthetic_trace(path)
            streams = read_trace(path)
        counts = per_document(streams, [5, 6], 4)
        for name, (delta, beta) in POLICIES.items():
            expected = sum(replay_stream(s, 5, "score", delta=delta, beta=beta, top_k=4)
                           ["repeat_misses"] for (r, _), s in streams.items() if r == "doc0")
            self.assertEqual(counts[5][name]["doc0"], expected)
        docs = ["doc0", "doc1", "doc2"]
        rows = contrasts(counts, [5, 6], docs, replicates=50, seed=0)
        self.assertEqual(len(rows), 6)
        for b in (5, 6):
            r = {x["contrast"]: x for x in rows if x["budget"] == b}
            total = lambda n: sum(counts[b][n][d] for d in docs)
            self.assertEqual(r["rank_increment"]["repeat_misses_saved"],
                             total("rank_removed") - total("full"))
            self.assertEqual(r["s2_as_preregistered"]["repeat_misses_saved"],
                             r["rank_increment"]["repeat_misses_saved"]
                             + r["beta_change"]["repeat_misses_saved"])
            for x in r.values():
                self.assertLessEqual(x["ci_low"], x["ci_high"])
                self.assertIn("post-hoc", x["label"])

    def test_bootstrap_is_paired(self):
        # Identical policies give a zero contrast in every replicate.
        counts = {1: {n: {"a": 5, "b": 9} for n in POLICIES}}
        for r in contrasts(counts, [1], ["a", "b"], replicates=20):
            self.assertEqual((r["repeat_misses_saved"], r["ci_low"], r["ci_high"]), (0, 0, 0))

    def test_s2_row_matches_preregistered_success(self):
        # The S2 contrast must draw the same bootstrap documents as evaluate.success,
        # otherwise it would not reproduce the preregistered S2 interval.
        from evaluate import success
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            synthetic_trace(path, docs=5)
            streams = read_trace(path)
        counts = per_document(streams, [5], 4)
        docs = sorted(counts[5]["full"])
        nested = {5: {name: {d: {"repeat_misses": counts[5][name][d]} for d in docs}
                      for name in ("full", "rf")}}
        nested[5]["lru"] = {d: {"repeat_misses": counts[5]["rf"][d] + 100} for d in docs}
        prereg = {"primary_budget": 5, "bootstrap": {"seed": 0, "replicates": 200}}
        expected = success(nested, prereg, docs)
        s2 = next(r for r in contrasts(counts, [5], docs, replicates=200, seed=0)
                  if r["contrast"] == "s2_as_preregistered")
        self.assertEqual(s2["repeat_misses_saved"], expected["s2_repeat_misses_saved_full_vs_rf"])
        self.assertEqual([s2["ci_low"], s2["ci_high"]], expected["s2_ci"])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            synthetic_trace(tmp / "trace.jsonl", pages=200, k=64, steps=10)
            subprocess.run([sys.executable, str(ROOT / "posthoc_rank_increment.py"),
                            str(tmp / "trace.jsonl"), "--out", str(tmp / "out"),
                            "--prereg", str(ROOT / "splits/preregistration.json")],
                           check=True, capture_output=True, cwd=ROOT)
            self.assertTrue((tmp / "out/rank_increment.csv").exists())
            self.assertTrue((tmp / "out/rank_increment_by_doc.csv").exists())


if __name__ == "__main__":
    unittest.main()
