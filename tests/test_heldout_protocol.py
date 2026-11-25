"""Audit item 3: the held-out driver must check stored rids, not file names."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import heldout


def write(folder, name, rid, T=3, L=28, K=64, P=80):
    R = np.tile(np.arange(K, dtype=np.int16), (T, L, 1))
    S = np.zeros((T, L, P), np.float16)
    path = Path(folder) / name
    np.savez(path, ranked=R, scores=S, rid=np.array(rid))
    return str(path)


class TraceInventory(unittest.TestCase):
    def test_renamed_dev_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = write(d, "renamed.npz", "gov_report/dev1")
            with self.assertRaisesRegex(ValueError, "development"):
                heldout.trace_inventory([p], forbid={"gov_report/dev1"})

    def test_duplicate_missing_and_unexpected(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = write(d, "a.npz", "x/1"), write(d, "b.npz", "x/1")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                heldout.trace_inventory([a, b])
            c = write(d, "c.npz", "x/2")
            with self.assertRaisesRegex(ValueError, "missing"):
                heldout.trace_inventory([a, c], expected={"x/1", "x/3"})

    def test_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "ranked shape"):
                heldout.trace_inventory([write(d, "a.npz", "x/1", L=1)])
            inv = heldout.trace_inventory([write(d, "b.npz", "x/2")], expected={"x/2"})
            self.assertEqual(inv["x/2"]["steps"], 3)
            self.assertEqual(len(inv["x/2"]["sha256"]), 64)

    def test_frozen_alpha_is_used(self):
        frozen = {"alpha": {"96": 0.3}, "linear": {}}
        self.assertEqual(heldout.alpha_of(96, frozen), 0.3)
        self.assertEqual(heldout.configs(96, frozen)["ema"][1]["alpha"], 0.3)
        self.assertEqual(heldout.alpha_of(96), heldout.ALPHA[96])

    def test_split_rids(self):
        rids = heldout.split_rids(Path(heldout.__file__).parent / "splits/longctx_v1.json", "dev")
        self.assertEqual(len(rids), 14)


if __name__ == "__main__":
    unittest.main()
