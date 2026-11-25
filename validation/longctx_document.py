"""Replay four B=96 policies on one archived development trace and compare every layer."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from efficient_dev import CONFIGS
from evaluate import write_csv
from kvpolicy import simulate

POLICIES = ("lru", "belady", "score", "ema:0.5")
FIELDS = ("accesses", "compulsory", "misses")


def replay(path):
    split = json.loads((ROOT / "splits/longctx_v1.json").read_text())
    dev = {d["rid"]: d for d in split["prompts"] if d["split"] == "dev"}
    with np.load(path, allow_pickle=False) as z:
        rid = str(z["rid"])
        if rid not in dev or int(z["prompt_tokens"]) != dev[rid]["prompt_tokens"]:
            raise ValueError("Document ID or prompt length differs from the development split")
        ranked, scores = z["ranked"], z["scores"]
    if (ranked.ndim != 3 or ranked.shape[1:] != (28, 64) or len(ranked) == 0
            or scores.ndim != 3 or scores.shape[:2] != ranked.shape[:2]
            or ranked.dtype.kind not in "iu" or ranked.min() < 0 or ranked.max() >= scores.shape[2]
            or np.any(np.diff(np.sort(ranked, axis=2), axis=2) == 0)):
        raise ValueError("Unexpected trace dimensions or invalid selected page IDs")
    with (ROOT / "docs/results/longctx_dev/misses.csv").open() as f:
        refs = [r for r in csv.DictReader(f) if r["doc"] == rid and r["B"] == "96"
                and r["config"] in POLICIES]
    reference = {(int(r["layer"]), r["config"]): r for r in refs}
    if len(refs) != len(reference) or len(refs) != 28 * len(POLICIES):
        raise ValueError("Incomplete or duplicate archive rows")

    rows = []
    for layer in range(28):
        ranks = ranked[:, layer]
        values = scores[:, layer].astype(np.float32)
        # Unfinished/future pages may have NaN padding. Every resident must still have a score.
        max_seen = np.maximum.accumulate(ranks.max(axis=1))
        for t, limit in enumerate(max_seen):
            if not np.isfinite(values[t, :int(limit)+1]).all():
                raise ValueError(f"Missing page score in layer {layer}, step {t}")
        for config in POLICIES:
            policy, params = CONFIGS[config]
            observed = simulate(ranks, values, 96, policy, **params)
            archived = {k:int(reference[layer, config][k]) for k in FIELDS}
            if any(observed[k] != archived[k] for k in FIELDS):
                raise ValueError(f"Layer {layer} {config}: replay {observed}, archive {archived}")
            rows.append(dict(layer=layer, policy=config, **observed,
                             repeat_misses=observed["misses"]-observed["compulsory"],
                             **{"archive_"+k:v for k,v in archived.items()}))
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    meta = dict(doc=rid, sha256=digest, prompt_tokens=dev[rid]["prompt_tokens"],
                steps=len(ranked), layers=28, K=64, B=96, matched_rows=len(rows))
    return meta, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    meta, rows = replay(args.trace)
    totals = []
    for policy in POLICIES:
        group = [r for r in rows if r["policy"] == policy]
        totals.append(dict(policy=policy, **{k:sum(r[k] for r in group)
                                             for k in (*FIELDS, "repeat_misses")}))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/"by_layer.csv", rows)
    write_csv(args.out/"totals.csv", totals)
    (args.out/"manifest.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(meta, *totals, sep="\n")


if __name__ == "__main__":
    main()
