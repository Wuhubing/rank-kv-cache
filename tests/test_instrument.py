"""Instrumented replays run end to end on a synthetic trace and produce consistent counts."""
import json

import numpy as np

import kvpolicy as kp
from analysis_v3 import instrument as ins
from test_kvpolicy_v2 import synthetic


def test_job_outputs(tmp_path):
    layers = [synthetic(seed=l, T=90, P0=100, K=8) for l in range(2)]
    R = np.stack([x[0] for x in layers], 1).astype(np.int16)
    S = np.stack([x[1] for x in layers], 1).astype(np.float16)
    p = tmp_path / "t.npz"
    np.savez(p, ranked=R, scores=S, rid=np.array("t"))
    (tmp_path / "t.json").write_text(json.dumps({"generated": [1] * 40 + [ins.THINK_END] + [2] * 50}))
    res = ins.job((str(p), 12))
    assert ins.think_end_step(str(p), 90) == 39
    ph = {(r[1], r[2]): r for r in res["phase"]}
    R0, S0, _ = kp.load(str(p))
    for l in range(2):
        tot = ph[l, "q1"][4] + ph[l, "q2-3"][4] + ph[l, "q4"][4]
        assert tot == kp.simulate(R0[:, l], None, 12, "lru")["misses"]
        assert ph[l, "thinking"][4] + ph[l, "answer"][4] == tot
        assert ph[l, "thinking"][3] == 40
    allr = [r for r in res["errors"] if r[2] == "all"]
    assert all(0 <= r[5] <= r[4] for r in allr)
    assert len(res["spatial"]) == 2 and len(res["stale"]) == 4 and len(res["freq"]) == 2 and len(res["pins"]) == 2
    st = {r[2]: r for r in res["stale"] if r[1] == 0}
    assert st["full"][3] == 0 and st["sel_only"][3] > 0
