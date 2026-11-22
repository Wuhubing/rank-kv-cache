import json
import unittest
from collections import Counter
from pathlib import Path

import split

ROOT = Path(__file__).resolve().parents[1]
SPLIT = json.loads((ROOT / "splits/longbench_v1.json").read_text())


class SplitTests(unittest.TestCase):
    def test_cells_and_documents(self):
        prompts = SPLIT["prompts"]
        self.assertEqual(len(prompts), 60)
        cells = Counter((p["task"], p["band"], p["split"]) for p in prompts)
        for task in split.TEMPLATES:
            for band in split.BANDS:
                self.assertEqual(cells[task, band, "dev"], 5)
                self.assertEqual(cells[task, band, "heldout"], 10)
        self.assertEqual(len({p["context_sha256"] for p in prompts}), 60)
        for p in prompts:
            low, high = split.BANDS[p["band"]]
            self.assertTrue(low <= p["prompt_tokens"] < high)
            self.assertEqual(p["request_id"], f"{p['task']}/{p['band']}/{p['_id']}")

    def test_pilot_in_dev_and_collection_from_dev(self):
        by_id = {(p["task"], p["_id"]): p for p in SPLIT["prompts"]}
        self.assertEqual(by_id[split.PILOT]["split"], "dev")
        dev = {p["request_id"]: p for p in SPLIT["prompts"] if p["split"] == "dev"}
        collection = SPLIT["dev_collection"]
        self.assertEqual(len(collection), 8)
        self.assertTrue(set(collection) <= set(dev))
        self.assertEqual(Counter((dev[r]["task"], dev[r]["band"]) for r in collection),
                         Counter({(t, b): 2 for t in split.TEMPLATES for b in split.BANDS}))

    def test_frozen_settings_match_script(self):
        self.assertEqual(SPLIT["seed"], split.SEED)
        self.assertEqual(SPLIT["templates"], split.TEMPLATES)
        self.assertEqual({k: tuple(v) for k, v in SPLIT["bands"].items()}, split.BANDS)
        self.assertEqual((SPLIT["model_revision"], SPLIT["data_revision"]),
                         (split.MODEL_REV, split.DATA_REV))

    def test_notebook_config_matches_split(self):
        notebook = json.loads((ROOT / "notebooks/first_trace.ipynb").read_text())
        cell, = [c for c in notebook["cells"] if c["id"] == "config"]
        namespace = {}
        exec("".join(cell["source"]), namespace)
        tokens = {p["request_id"]: p["prompt_tokens"] for p in SPLIT["prompts"]}
        self.assertEqual(list(namespace["DEV_COLLECTION"]), SPLIT["dev_collection"])
        self.assertEqual(namespace["DEV_COLLECTION"],
                         {r: tokens[r] for r in SPLIT["dev_collection"]})
        self.assertEqual(namespace["TEMPLATES"], split.TEMPLATES)
        self.assertEqual((namespace["MODEL_REV"], namespace["DATA_REV"], namespace["MAX_NEW"]),
                         (split.MODEL_REV, split.DATA_REV, 128))
        heldout = {p["request_id"] for p in SPLIT["prompts"] if p["split"] == "heldout"}
        text = "".join("".join(c["source"]) for c in notebook["cells"])
        self.assertFalse([r for r in heldout if r.split("/")[-1] in text])


if __name__ == "__main__":
    unittest.main()
