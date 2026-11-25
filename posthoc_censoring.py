"""POST-HOC / EXPLORATORY: how many logistic training labels are right-censored.

The frozen logistic model (preregister.py) labels an LRU candidate at step t positive
when its next selection is within 16 steps. A candidate not selected again before the
trace ends is labelled negative even when t + 16 runs past the end, so its label is
unobserved, not negative. This script rebuilds the exact training sample preregister.py
used (same dataset order, numpy seed and sample size) and counts those rows. As a check
that the sample is the same, it refits the logistic model on it and compares the result
with the frozen coefficients; the frozen model itself is never changed. Needs numpy.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gap_analysis import FEATURES, Logistic, build_dataset, censored_labels
from replay import read_trace

LABEL = "post-hoc / exploratory (not preregistered)"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dev_dir", type=Path)
    parser.add_argument("--prereg", type=Path, default=Path("splits/preregistration.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.prereg.read_text())
    spec = next(p for p in prereg["policies"] if p["policy"] == "logistic")
    horizon, rows, seed = spec["label_horizon"], spec["sample_rows"], spec["seed"]
    budget = prereg["primary_budget"]
    streams = read_trace(args.dev_dir / "trace.jsonl")
    prompts = {}
    for path in (args.dev_dir / "manifests").glob("*.json"):
        m = json.loads(path.read_text())
        prompts[m["request_id"]] = m["prompt_tokens"]
    X, distance, _, _, _, t, length = build_dataset(streams, prompts, budget, with_time=True)
    sample = np.random.default_rng(seed).choice(len(X), min(len(X), rows), replace=False)
    censored = censored_labels(distance, t, length, horizon)
    positive = distance <= horizon

    def summary(idx):
        n = len(idx)
        neg = int((~positive[idx]).sum())
        c = int(censored[idx].sum())
        return dict(rows=n, positives=n - neg, negatives=neg, censored=c,
                    censored_share_of_rows=c / n, censored_share_of_negatives=c / neg)

    result = dict(label=LABEL, trace=str(args.dev_dir / "trace.jsonl"), budget=budget,
                  label_horizon=horizon, sample_rows=rows, seed=seed,
                  all_candidates=summary(np.arange(len(X))), training_sample=summary(sample),
                  steps_per_stream=sorted({int(v) for v in length}))
    refit = Logistic(range(len(FEATURES))).fit(X[sample], positive[sample].astype(float))
    result["sample_check"] = dict(
        refit_reproduces_frozen_coefficients=refit.to_dict() == spec["model"],
        note="refit of the logistic model on this rebuilt training sample, compared with "
             "the frozen coefficients in the preregistration")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "censoring.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
