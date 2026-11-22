"""Evaluate whether selector-score magnitude is more useful than ordinal rank.

Requires a Trace v2 JSONL produced with RECORD_SCORES=True in first_trace.ipynb.
For each selected page, its cutoff-normalized margin is

    (score - Kth score) / (best score - Kth score),

which is zero at the selection boundary and one for the best page.  The replacement
score is tau + gamma * last_margin + beta * capped_recent_frequency.  Comparisons use
gamma=0 at the same beta, so they isolate the new score-magnitude term.
"""
import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from rank_validation import percentile
from replay import FREQ_CAP, FREQ_WINDOW


def read_score_trace(path):
    streams = defaultdict(list)
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                key = row["request_id"], row["layer_id"]
                step = row["decode_step"]
                pages = row["ranked_page_ids"]
                scores = row["ranked_page_scores"]
                if step != len(streams[key]):
                    raise ValueError(f"expected decode_step {len(streams[key])}")
                if len(pages) != len(scores) or len(pages) != len(set(pages)):
                    raise ValueError("page IDs and scores must be aligned and unique")
                if any(type(p) is not int or p < 0 for p in pages):
                    raise ValueError("page IDs must be nonnegative integers")
                if any(type(s) not in (int, float) or not math.isfinite(s) for s in scores):
                    raise ValueError("scores must be finite numbers")
                if any(a < b for a, b in zip(scores, scores[1:])):
                    raise ValueError("scores must be ordered best first")
                streams[key].append((tuple(pages), tuple(float(s) for s in scores)))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}, line {line_number}: {error}") from error
    if not streams:
        raise ValueError("empty score trace")
    return dict(streams)


def normalized_margins(scores):
    if not scores:
        return ()
    span = scores[0] - scores[-1]
    if span <= 0:
        return (0.0,) * len(scores)
    return tuple((score - scores[-1]) / span for score in scores)


def replay_margin(steps, budget, gamma=0, beta=0):
    resident = {}
    last_margin = {}
    history = []
    seen = set()
    misses = accesses = 0
    for step, (pages, scores) in enumerate(steps):
        selected = set(pages)
        if len(selected) > budget:
            raise ValueError(f"budget {budget} is below the selection at step {step}")
        hits = selected.intersection(resident)
        missing = selected - hits
        evict = max(0, len(resident) + len(missing) - budget)
        if evict:
            candidates = resident.keys() - selected

            def frequency(page):
                return sum(page in history[s]
                           for s in range(max(0, step - FREQ_WINDOW), step))

            def key(page):
                bonus = gamma * last_margin[page]
                bonus += beta * min(frequency(page), FREQ_CAP)
                return (resident[page][0] + bonus,) + resident[page]

            for page in sorted(candidates, key=key)[:evict]:
                del resident[page], last_margin[page]
        margins = normalized_margins(scores)
        for page, margin in zip(pages, margins):
            resident[page] = (step, page in hits, -page)
            last_margin[page] = margin
        history.append(selected)
        accesses += len(pages)
        misses += len(missing)
        seen.update(selected)
    return Counter(misses=misses, repeat_misses=misses - len(seen), accesses=accesses,
                   compulsory_misses=len(seen))


def evaluate(streams, budgets, gammas, betas, replicates=2000, seed=0):
    by_doc = defaultdict(Counter)
    for budget in budgets:
        for beta in betas:
            for gamma in gammas:
                for (request, _), steps in sorted(streams.items()):
                    by_doc[budget, beta, gamma, request].update(
                        replay_margin(steps, budget, gamma, beta))
    docs = sorted({request for request, _ in streams})
    rng = random.Random(seed)
    samples = [[rng.choice(docs) for _ in docs] for _ in range(replicates)]
    rows, doc_rows = [], []
    for (budget, beta, gamma, request), counts in sorted(by_doc.items()):
        base = by_doc[budget, beta, 0, request]["repeat_misses"]
        doc_rows.append(dict(budget=budget, beta=beta, gamma=gamma, document=request,
                             **counts, repeat_saved_vs_matched_beta=base-counts["repeat_misses"]))
    for budget in budgets:
        for beta in betas:
            base = sum(by_doc[budget, beta, 0, d]["repeat_misses"] for d in docs)
            for gamma in gammas:
                counts = Counter()
                for d in docs:
                    counts.update(by_doc[budget, beta, gamma, d])
                reductions = []
                for sample in samples:
                    b = sum(by_doc[budget, beta, 0, d]["repeat_misses"] for d in sample)
                    m = sum(by_doc[budget, beta, gamma, d]["repeat_misses"] for d in sample)
                    if b:
                        reductions.append((b-m)/b)
                saved = base-counts["repeat_misses"]
                rows.append(dict(budget=budget, beta=beta, gamma=gamma, documents=len(docs),
                                 **counts, matched_beta_repeat_misses=base,
                                 repeat_saved=saved, repeat_reduction=saved/base if base else None,
                                 reduction_low=percentile(reductions, .025),
                                 reduction_high=percentile(reductions, .975)))
    return rows, doc_rows


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--budgets", type=int, nargs="+", default=[128])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--betas", type=float, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/score_validation"))
    args = parser.parse_args()
    gammas, betas = tuple(dict.fromkeys(args.gammas)), tuple(dict.fromkeys(args.betas))
    if 0 not in gammas or any(x < 0 for x in (*gammas, *betas)):
        parser.error("gammas must include 0 and all weights must be nonnegative")
    rows, docs = evaluate(read_score_trace(args.trace), args.budgets, gammas, betas,
                          args.bootstrap, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "score_margin.csv", rows)
    write_csv(args.out / "score_margin_by_doc.csv", docs)
    print(f"Wrote score-margin validation to {args.out}")


if __name__ == "__main__":
    main()
