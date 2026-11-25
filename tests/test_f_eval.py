"""F3 evaluation on synthetic 'held-out' traces."""
import argparse
import json

import numpy as np

from analysis_v3 import f_eval
from test_kvpolicy_v2 import synthetic


def test_replay_and_criteria(tmp_path):
    rids = [f"doc{i}" for i in range(3)]
    tr = tmp_path / "tr"
    tr.mkdir()
    for i, rid in enumerate(rids):
        layers = [synthetic(seed=10 * i + l, T=80, P0=90, K=8) for l in range(2)]
        R = np.stack([x[0] for x in layers], 1).astype(np.int16)
        S = np.stack([x[1] for x in layers], 1).astype(np.float16)
        np.savez(tr / f"{rid}.npz", ranked=R, scores=S, rid=np.array(rid))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"prompts": [{"rid": r, "split": "heldout"} for r in rids]}))
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"primary_B": 12, "models": {"m": {"budgets": {"12": {
        "ema_alpha": 0.5, "p2": {"policy": "kernel", "kw": {"kernel": "power", "gamma": 1.0, "normalize": True}},
        "offline_alloc": [10, 14], "p4": {"kind": "online", "kw": {"N": 16, "s": 2, "decay": 0.5}}}}}}}))
    out = tmp_path / "out"
    a = argparse.Namespace(traces=tr, frozen=frozen, model="m", split=split, out=out, procs=2)
    f_eval.cmd_replay(a)
    f_eval.cmd_criteria(a)
    res = json.loads((out / "criteria.json").read_text())
    b = res["budgets"]["12"]
    assert {"ema", "p2", "alloc_ema", "p4", "p2_minus_ema", "p4_minus_uniform_ema"} <= set(b)
    assert "H1_pass" in res and "H2_pass" in res and "H3_pass" in res
    # a folder that does not match the split is rejected
    (tr / "doc2.npz").unlink()
    try:
        f_eval.cmd_replay(a)
        raise AssertionError("expected SystemExit")
    except SystemExit:
        pass
