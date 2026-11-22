"""Post-hoc validation of whether last selection rank changes replacement decisions.

This script intentionally stays separate from the preregistered evaluation.  It runs
matched-beta ablations (delta=0 versus delta>0 at the same frequency weight) and a
one-step decision audit on the delta=0 cache trajectory.  The audit asks how often a
rank bonus would change the victims when cache state, candidates and frequency are
held fixed, then compares the victims' true near-future reuse.

The trace is never modified.  All computations use the standard library.
"""
import argparse
import csv
import json
import math
import random
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from replay import FREQ_CAP, FREQ_WINDOW, infer_top_k, read_trace, replay_stream

AUDIT_HORIZONS = (1, 8, 16, 32)
AUDIT_COUNT_KEYS = (
    "eviction_events", "eviction_slots", "candidate_pages", "top_candidates",
    "top_baseline_victims", "changed_events", "changed_slots", "top_saved",
    "top_newly_evicted",
) + tuple(
    f"{prefix}_{horizon}"
    for horizon in AUDIT_HORIZONS
    for prefix in ("baseline_victims_reused", "rank_victims_reused", "avoided_near_reuse")
)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, int(q * len(values)))]


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def audit_values(counts):
    """Return every audit counter in a stable order, including explicit zeros."""
    return {key: counts[key] for key in AUDIT_COUNT_KEYS}


def matched_beta(streams, budgets, deltas, betas, top_k, replicates=2000, seed=0):
    """Return pooled and per-document score-policy results at matched beta values."""
    by_doc = defaultdict(Counter)
    for budget in budgets:
        for beta in betas:
            for delta in deltas:
                for (request, _), steps in sorted(streams.items()):
                    result = replay_stream(steps, budget, "score", delta=delta, beta=beta,
                                           top_k=top_k)
                    cell = by_doc[budget, beta, delta, request]
                    for key in ("misses", "repeat_misses", "accesses", "compulsory_misses"):
                        cell[key] += result[key]

    docs = sorted({request for request, _ in streams})
    doc_rows = []
    for (budget, beta, delta, request), counts in sorted(by_doc.items()):
        base = by_doc[budget, beta, 0, request]["repeat_misses"]
        doc_rows.append(dict(
            budget=budget, beta=beta, delta=delta, document=request,
            misses=counts["misses"], repeat_misses=counts["repeat_misses"],
            accesses=counts["accesses"], compulsory_misses=counts["compulsory_misses"],
            repeat_saved_vs_matched_beta=base - counts["repeat_misses"],
        ))

    rng = random.Random(seed)
    samples = [[rng.choice(docs) for _ in docs] for _ in range(replicates)]
    rows = []
    for budget in budgets:
        for beta in betas:
            base = sum(by_doc[budget, beta, 0, d]["repeat_misses"] for d in docs)
            for delta in deltas:
                counts = Counter()
                for d in docs:
                    counts.update(by_doc[budget, beta, delta, d])
                reductions = []
                for sample in samples:
                    b = sum(by_doc[budget, beta, 0, d]["repeat_misses"] for d in sample)
                    m = sum(by_doc[budget, beta, delta, d]["repeat_misses"] for d in sample)
                    if b:
                        reductions.append((b - m) / b)
                saved = base - counts["repeat_misses"]
                rows.append(dict(
                    budget=budget, beta=beta, delta=delta, documents=len(docs),
                    misses=counts["misses"], repeat_misses=counts["repeat_misses"],
                    matched_beta_repeat_misses=base, repeat_saved=saved,
                    repeat_reduction=saved / base if base else None,
                    reduction_low=percentile(reductions, .025),
                    reduction_high=percentile(reductions, .975),
                    accesses=counts["accesses"], compulsory_misses=counts["compulsory_misses"],
                ))
    return rows, doc_rows


def _next_use_lists(steps):
    uses = defaultdict(list)
    for t, pages in enumerate(steps):
        for page in pages:
            uses[page].append(t)
    return uses


def _next_use(uses, page, step):
    positions = uses[page]
    i = bisect_right(positions, step)
    return positions[i] if i < len(positions) else math.inf


def audit_stream(steps, budget, beta, deltas, top_k):
    """Audit rank counterfactuals on the matched delta=0 trajectory of one stream."""
    top_width = -(-top_k // 4)
    history = [set(pages) for pages in steps]
    uses = _next_use_lists(steps)
    resident = {}
    last_rank = {}
    totals = {delta: Counter() for delta in deltas if delta != 0}

    def frequency(page, step):
        return sum(page in history[s] for s in range(max(0, step - FREQ_WINDOW), step))

    for step, pages in enumerate(steps):
        selected = set(pages)
        if len(selected) > budget:
            raise ValueError(f"budget {budget} is below the selection at step {step}")
        hits = selected.intersection(resident)
        missing = selected - hits
        evict_count = max(0, len(resident) + len(missing) - budget)
        if evict_count:
            candidates = resident.keys() - selected

            def key(page, delta):
                bonus = delta * (last_rank[page] <= top_width)
                bonus += beta * min(frequency(page, step), FREQ_CAP)
                return (resident[page][0] + bonus,) + resident[page]

            baseline = set(sorted(candidates, key=lambda p: key(p, 0))[:evict_count])
            top_candidates = {p for p in candidates if last_rank[p] <= top_width}
            for delta, counts in totals.items():
                alternate = set(sorted(candidates, key=lambda p: key(p, delta))[:evict_count])
                changed = baseline - alternate
                replacement = alternate - baseline
                counts["eviction_events"] += 1
                counts["eviction_slots"] += evict_count
                counts["candidate_pages"] += len(candidates)
                counts["top_candidates"] += len(top_candidates)
                counts["top_baseline_victims"] += len(baseline & top_candidates)
                counts["changed_events"] += baseline != alternate
                counts["changed_slots"] += len(changed)
                counts["top_saved"] += len(changed & top_candidates)
                counts["top_newly_evicted"] += len(replacement & top_candidates)
                for horizon in AUDIT_HORIZONS:
                    base_near = sum(_next_use(uses, p, step) <= step + horizon
                                    for p in baseline)
                    alt_near = sum(_next_use(uses, p, step) <= step + horizon
                                   for p in alternate)
                    counts[f"baseline_victims_reused_{horizon}"] += base_near
                    counts[f"rank_victims_reused_{horizon}"] += alt_near
                    counts[f"avoided_near_reuse_{horizon}"] += base_near - alt_near

            for page in baseline:
                del resident[page]
        for rank, page in enumerate(pages, 1):
            resident[page] = (step, page in hits, -page)
            last_rank[page] = rank
    return totals


def decision_audit(streams, budgets, deltas, betas, top_k):
    """Aggregate one-step rank counterfactuals, plus per-document audit counts."""
    by_doc = defaultdict(Counter)
    for budget in budgets:
        for beta in betas:
            for (request, _), steps in sorted(streams.items()):
                result = audit_stream(steps, budget, beta, deltas, top_k)
                for delta, counts in result.items():
                    by_doc[budget, beta, delta, request].update(counts)

    doc_rows = []
    for (budget, beta, delta, request), counts in sorted(by_doc.items()):
        doc_rows.append(dict(budget=budget, beta=beta, delta=delta, document=request,
                             **audit_values(counts)))

    docs = sorted({request for request, _ in streams})
    rows = []
    for budget in budgets:
        for beta in betas:
            for delta in deltas:
                if delta == 0:
                    continue
                counts = Counter()
                for request in docs:
                    counts.update(by_doc[budget, beta, delta, request])
                events = counts["eviction_events"]
                candidates = counts["candidate_pages"]
                row = dict(
                    budget=budget, beta=beta, delta=delta, documents=len(docs),
                    **audit_values(counts),
                    changed_event_rate=counts["changed_events"] / events if events else None,
                    changed_slots_per_event=counts["changed_slots"] / events if events else None,
                    top_candidate_share=counts["top_candidates"] / candidates if candidates else None,
                )
                rows.append(row)
    return rows, doc_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--budgets", type=int, nargs="+", default=[128])
    parser.add_argument("--deltas", type=float, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--betas", type=float, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/rank_validation"))
    args = parser.parse_args()
    deltas = tuple(dict.fromkeys(args.deltas))
    betas = tuple(dict.fromkeys(args.betas))
    if 0 not in deltas:
        parser.error("--deltas must include 0 for the matched-beta baseline")
    if any(x < 0 for x in (*deltas, *betas)):
        parser.error("delta and beta values must be nonnegative")
    streams = read_trace(args.trace)
    top_k = infer_top_k(streams) if args.top_k is None else args.top_k
    args.out.mkdir(parents=True, exist_ok=True)

    matched, matched_docs = matched_beta(streams, args.budgets, deltas, betas, top_k,
                                         args.bootstrap, args.seed)
    audit, audit_docs = decision_audit(streams, args.budgets, deltas, betas, top_k)
    write_csv(args.out / "matched_beta.csv", matched)
    write_csv(args.out / "matched_beta_by_doc.csv", matched_docs)
    write_csv(args.out / "decision_audit.csv", audit)
    write_csv(args.out / "decision_audit_by_doc.csv", audit_docs)
    metadata = dict(trace=str(args.trace), budgets=args.budgets, deltas=list(deltas),
                    betas=list(betas), top_k=top_k, bootstrap=args.bootstrap, seed=args.seed,
                    note="Post-hoc validation; not part of the preregistered held-out analysis.")
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote matched-beta and decision-audit results to {args.out}")


if __name__ == "__main__":
    main()
