"""POST-HOC / EXPLORATORY: the clean rank increment that criterion S2 does not isolate.

Preregistered S2 compares `full` (delta=8, beta=2) with `rf` (delta=0, beta=4), so it
changes the frequency weight as well as adding the rank term. This script holds beta
fixed and compares score:8:2 (`full`) with score:0:2 (`full` with the rank term
removed). It also reports score:0:2 vs `rf` so the S2 difference splits exactly into
    rf - full = (rf - score:0:2) + (score:0:2 - full)
                beta change 4 -> 2   rank term at beta=2

Not part of the preregistration: the comparison was chosen after the held-out result
was seen. Statistics follow evaluate.py: per-document repeat misses, paired document
bootstrap (2000 replicates, seed 0, the same resample shared by every contrast).
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from evaluate import percentile
from replay import read_trace, replay_stream

LABEL = "post-hoc / exploratory (not preregistered)"
# name -> (delta, beta); full and rf are the frozen preregistered weights.
POLICIES = {"full": (8, 2), "rank_removed": (0, 2), "rf": (0, 4)}
CONTRASTS = (  # (name, baseline, treatment): saved = baseline - treatment
    ("rank_increment", "rank_removed", "full"),  # the clean rank term, beta held at 2
    ("beta_change", "rf", "rank_removed"),       # the confound: beta 4 -> 2, no rank
    ("s2_as_preregistered", "rf", "full"),       # the sum of the two
)


def per_document(streams, budgets, top_k):
    """counts[budget][policy][doc] -> repeat misses summed over layers."""
    counts = defaultdict(lambda: defaultdict(Counter))
    for budget in budgets:
        for (request, _), steps in sorted(streams.items()):
            for name, (delta, beta) in POLICIES.items():
                c = replay_stream(steps, budget, "score", delta=delta, beta=beta, top_k=top_k)
                counts[budget][name][request] += c["repeat_misses"]
    return counts


def contrasts(counts, budgets, docs, replicates=2000, seed=0):
    rows = []
    for budget in budgets:
        rep = lambda name, sample: sum(counts[budget][name][d] for d in sample)
        rng = random.Random(seed)
        samples = [[rng.choice(docs) for _ in docs] for _ in range(replicates)]
        for name, base, treat in CONTRASTS:
            saved = rep(base, docs) - rep(treat, docs)
            boot = [rep(base, s) - rep(treat, s) for s in samples]
            low, high = percentile(boot, .025), percentile(boot, .975)
            rows.append(dict(label=LABEL, budget=budget, contrast=name,
                             baseline=f"{base} {POLICIES[base]}",
                             treatment=f"{treat} {POLICIES[treat]}",
                             baseline_repeat_misses=rep(base, docs),
                             treatment_repeat_misses=rep(treat, docs),
                             repeat_misses_saved=saved, ci_low=low, ci_high=high,
                             relative_to_baseline=saved / rep(base, docs) if rep(base, docs)
                             else None,
                             ci_excludes_zero=low > 0 or high < 0,
                             documents=len(docs), replicates=replicates, seed=seed))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, default=Path("splits/preregistration.json"))
    args = parser.parse_args()
    prereg = json.loads(args.prereg.read_text())
    frozen = {p["name"]: (p["delta"], p["beta"]) for p in prereg["policies"]
              if p["name"] in ("full", "rf")}
    if frozen != {"full": POLICIES["full"], "rf": POLICIES["rf"]}:
        raise SystemExit(f"frozen weights changed: {frozen}")
    streams = read_trace(args.trace)
    docs = sorted({r for r, _ in streams})
    budgets = prereg["budgets"]
    counts = per_document(streams, budgets, prereg["top_k"])
    rows = contrasts(counts, budgets, docs, prereg["bootstrap"]["replicates"],
                     prereg["bootstrap"]["seed"])
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "rank_increment_by_doc.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["budget", "policy", "delta", "beta", "doc", "repeat_misses"])
        for b in budgets:
            for name in POLICIES:
                for d in docs:
                    w.writerow([b, name, *POLICIES[name], d, counts[b][name][d]])
    with (args.out / "rank_increment.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"{LABEL.upper()}: {len(docs)} documents from {args.trace}")
    for r in rows:
        rel = r["relative_to_baseline"]
        print(f"B={r['budget']:<4}{r['contrast']:<20} {r['baseline']:>14} -> {r['treatment']:<14}"
              f" saved {r['repeat_misses_saved']:>5} [{r['ci_low']}, {r['ci_high']}]"
              f"  {'n/a' if rel is None else f'{100 * rel:+.3f}%'} of baseline")


if __name__ == "__main__":
    main()
