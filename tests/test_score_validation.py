import json
import tempfile
import unittest
from pathlib import Path

from score_validation import normalized_margins, read_score_trace, replay_margin


class ScoreValidationTests(unittest.TestCase):
    def test_normalized_margins(self):
        self.assertEqual(normalized_margins((9, 7, 5)), (1, 0.5, 0))
        self.assertEqual(normalized_margins((3, 3)), (0, 0))
        self.assertEqual(normalized_margins(()), ())

    def test_margin_bonus_can_save_reused_page(self):
        # Pages 6 and 5 share a step, but score magnitude says 6 is much stronger.
        steps = [((6, 5), (10.0, 0.0)), ((7,), (1.0,)), ((8,), (1.0,)), ((6,), (1.0,))]
        self.assertEqual(replay_margin(steps, 3, gamma=0)["repeat_misses"], 1)
        self.assertEqual(replay_margin(steps, 3, gamma=1)["repeat_misses"], 0)

    def test_score_trace_validation(self):
        row = dict(request_id="r", layer_id=0, decode_step=0,
                   ranked_page_ids=[2, 1], ranked_page_scores=[4.0, 3.0])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(read_score_trace(path)["r", 0][0][0], (2, 1))
            for scores in ([3.0], [3.0, 4.0], [3.0, float("inf")]):
                path.write_text(json.dumps({**row, "ranked_page_scores": scores}) + "\n")
                with self.assertRaises(ValueError):
                    read_score_trace(path)


if __name__ == "__main__":
    unittest.main()
