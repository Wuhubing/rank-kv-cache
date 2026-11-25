import unittest

from truncate_trace import cut_step

VOCAB = {1: " The", 2: " answer", 3: ".\n", 4: "Answer", 5: ":\n", 6: "\n"}


def decode(ids):
    return "".join(VOCAB[i] for i in ids)


class TruncateTests(unittest.TestCase):
    def test_cut_at_first_stop_token(self):
        # generated[2] holds the first newline, so steps 0 and 1 produce the answer.
        self.assertEqual(cut_step([1, 2, 3, 4, 5, 1], decode, "\n"), 2)
        self.assertEqual(cut_step([6, 1, 2], decode, "\n"), 0)
        self.assertIsNone(cut_step([1, 2, 4], decode, "\n"))


if __name__ == "__main__":
    unittest.main()
