"""Rerun the corrected B=96 rank baseline and AUCs on the archived long-context dev traces."""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import efficient_dev as ed
import kvpolicy as kp

BUDGET = ed.PRIMARY
BASELINES = ("lru", "belady")
RANK_CONFIGS = ed.FAMILIES["rank"]
RESULTS = ROOT / "docs/results/longctx_dev"


def dev_inputs(folder, smoke):
    split = json.loads((ROOT / "splits/longctx_v1.json").read_text())
    expected = {entry["rid"] for entry in split["prompts"] if entry["split"] == "dev"}
    inputs = {}
    for path in sorted(folder.glob("*.npz")):
        with np.load(path, allow_pickle=False) as trace:
            if not {"rid", "ranked", "scores"}.issubset(trace.files):
                raise ValueError(f"{path.name} is missing rid, ranked or scores")
            rid = str(trace["rid"])
        if rid not in expected or rid in inputs:
            raise ValueError(f"Unexpected or duplicate dev ID in {path.name}: {rid}")
        inputs[rid] = path
    if not inputs or (not smoke and inputs.keys() != expected):
        raise ValueError(f"Expected the {len(expected)} frozen dev traces; missing: {sorted(expected - inputs.keys())}")
    return dict(list(sorted(inputs.items()))[:1]) if smoke else dict(sorted(inputs.items()))


def previous_counts():
    with (RESULTS / "misses.csv").open(newline="", encoding="utf-8") as source:
        return {(r["doc"], int(r["layer"]), r["config"]):
                tuple(int(r[k]) for k in ("accesses", "misses", "compulsory"))
                for r in csv.DictReader(source)
                if int(r["B"]) == BUDGET and r["config"] in (*BASELINES, *RANK_CONFIGS)}


def replay_document(job):
    rid, path, smoke = job
    with np.load(path, allow_pickle=False) as trace:
        ranked = trace["ranked"]
    if ranked.ndim != 3 or ranked.shape[1:] != (28, 64) or not ranked.shape[0]:
        raise ValueError(f"{path.name}: expected a nonempty ranked array of shape (steps, 28, 64)")
    rows = []
    for layer in range(1 if smoke else 28):
        selection = ranked[:, layer]
        if (selection < 0).any() or (np.diff(np.sort(selection, axis=1), axis=1) == 0).any():
            raise ValueError(f"{path.name}: invalid selected page IDs in layer {layer}")
        for config in (*BASELINES, *RANK_CONFIGS):
            policy, params = ed.CONFIGS[config]
            c = kp.simulate(selection, None, BUDGET, policy, **params)
            rows.append((rid, layer, config, BUDGET, c["accesses"], c["misses"], c["compulsory"]))
    print(f"Replayed {rid}", flush=True)
    return rows


def compare_counts(results, original, output):
    paired = []
    per_doc = defaultdict(lambda: [0, 0, 0])
    for doc, layer, config, budget, acc, after, comp in results:
        old_acc, before, old_comp = original[doc, layer, config]
        if (acc, comp) != (old_acc, old_comp) or (config in BASELINES and before != after):
            raise ValueError(f"Trace replay differs from the archived controls: {doc}, layer {layer}, {config}")
        if config == "rank:0:0" and after != original[doc, layer, "lru"][1]:
            raise AssertionError(f"Zero-weight rank must equal LRU: {doc}, layer {layer}")
        paired.append((doc, layer, config, budget, acc, comp, before, after, after - before))
        totals = per_doc[doc, config]
        totals[0] += before
        totals[1] += after
        totals[2] += comp
    ed.write(output / "rank_by_layer.csv", ["doc", "layer", "config", "B", "accesses", "compulsory",
                                            "original_misses", "corrected_misses", "delta_misses"], paired)
    return per_doc


def write_lodo(docs, totals, output):
    folds = []
    stats = []
    for version, index in (("original", 0), ("corrected", 1)):
        policies = []
        for held in docs:
            train = [doc for doc in docs if doc != held]
            chosen = min(RANK_CONFIGS, key=lambda c: (sum(totals[d, c][index] for d in train), c))
            misses = totals[held, chosen][index]
            folds.append((version, held, chosen, misses))
            policies.append(misses)
        lru = [totals[d, "lru"][index] for d in docs]
        oracle = [totals[d, "belady"][index] for d in docs]
        compulsory = sum(totals[d, "lru"][2] for d in docs)
        saved = sum(lru) - sum(policies)
        low, high = kp.bootstrap(lru, policies, oracle)
        stats.append(dict(version=version, budget=BUDGET, misses=sum(policies),
                          G=saved / (sum(lru) - sum(oracle)), G_CI=[float(low), float(high)],
                          repeat_miss_reduction=saved / (sum(lru) - compulsory)))
    ed.write(output / "rank_lodo.csv", ["version", "doc", "chosen", "misses"], folds)
    return stats


def write_auc(inputs, original, output):
    features, labels, windows = [], [], []
    for rid, path in inputs.items():
        data = ed.collect(str(path), BUDGET, track_horizon=True)
        for layer, counts in data["misses"].items():
            if counts != original[rid, layer, "lru"]:
                raise ValueError(f"AUC candidate replay differs from archived LRU: {rid}, layer {layer}")
        features.append(data["X"])
        labels.append(data["soon"])
        windows.append(data["complete_horizon"])
        print(f"Collected AUC features for {rid}", flush=True)
    scores = np.concatenate(features)
    del features
    soon = np.concatenate(labels)
    complete = np.concatenate(windows)
    with (RESULTS / "gap.csv").open(newline="", encoding="utf-8") as source:
        before = {row["item"]: float(row["value"]) for row in csv.DictReader(source)
                  if row["kind"] == "auc_soon"}
    results = []
    for j, feature in enumerate(kp.FEATURES):
        column = scores[:, j]
        results.append((feature, before[feature], ed.auc(column, soon),
                        ed.auc(column[complete], soon[complete]), len(soon), int(complete.sum())))
    ed.write(output / "auc.csv", ["feature", "original_auc", "tie_corrected_auc", "complete_32_step_auc",
                                  "candidates", "complete_window_candidates"], results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, help="Directory containing the original 14 dev NPZs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="Replay layer 0 of one dev document")
    args = parser.parse_args()

    inputs = dev_inputs(args.traces, args.smoke)
    args.out.mkdir(parents=True, exist_ok=False)
    jobs = [(rid, path, args.smoke) for rid, path in inputs.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = [r for part in pool.map(replay_document, jobs) for r in part]
    original = previous_counts()
    totals = compare_counts(results, original, args.out)
    metadata = {"budget": BUDGET, "smoke": args.smoke, "traces": []}
    for rid, path in inputs.items():
        with path.open("rb") as source:
            metadata["traces"].append({"rid": rid, "sha256": hashlib.file_digest(source, "sha256").hexdigest()})
    if not args.smoke:
        metadata["lodo"] = write_lodo(inputs, totals, args.out)
        write_auc(inputs, original, args.out)
    (args.out / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Results: {args.out}")


if __name__ == "__main__":
    main()
