"""Replay batched sparse-KV page selections. Core replay uses the standard library."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

POLICIES = ("lru", "fifo", "rank", "score", "belady")
FREQ_WINDOW, FREQ_CAP = 8, 4


def read_trace(path):
    """Load JSONL, preserving time order independently for each request and layer."""
    streams = defaultdict(list)
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                request = row["request_id"]
                layer = row["layer_id"]
                step = row["decode_step"]
                pages = row["ranked_page_ids"]
                if not isinstance(request, str) or not request.strip():
                    raise ValueError("request_id must be a nonempty string")
                if type(layer) is not int or layer < 0:
                    raise ValueError("layer_id must be a nonnegative integer")
                stream = streams[request, layer]
                if type(step) is not int or step != len(stream):
                    raise ValueError(f"expected decode_step {len(stream)} for {(request, layer)}")
                if not isinstance(pages, list) or any(type(p) is not int or p < 0 for p in pages):
                    raise ValueError("ranked_page_ids must be a list of nonnegative integers")
                if len(pages) != len(set(pages)):
                    raise ValueError("duplicate page in the same step")
                stream.append(tuple(pages))
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError(f"{path}, line {line_number}: {error}") from error
    if not streams:
        raise ValueError("empty trace")
    return dict(streams)


def infer_top_k(streams):
    """K is the widest selection in the trace. Early steps can select fewer pages."""
    return max((len(pages) for steps in streams.values() for pages in steps), default=0)


def iter_replay(steps, budget, policy, *, delta=0, beta=0, top_k=None):
    """Start one empty cache and yield (hits, misses, residents) after every step.

    A step is a set of simultaneous, compulsory admissions. Every policy protects the
    whole current selection and evicts only as many unselected residents as necessary.
    Eviction keys sort ascending, so the smallest key leaves first:
      lru    (last selection step, was a hit at that step, -page ID)
      fifo   (admission step, -page ID); hits do not refresh
      rank   (-rank at last selection, then the LRU key); rank 1 is best
      score  (tau + delta*[rank <= ceil(K/4)] + beta*min(f, 4), then the LRU key),
             where f counts selections in the FREQ_WINDOW steps before the current one
      belady (-next use step, page ID); no future use counts as infinitely far
    """
    if policy not in POLICIES or type(budget) is not int or budget < 1:
        raise ValueError(f"expected policy in {POLICIES} and a positive integer budget")
    for weight in (delta, beta):
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
            raise ValueError("delta and beta must be finite nonnegative numbers")
    if top_k is None:
        top_k = max((len(pages) for pages in steps), default=0)
    if type(top_k) is not int or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    top_quartile = -(-top_k // 4)
    future = defaultdict(deque)
    if policy == "belady":
        for step, pages in enumerate(steps):
            for page in pages:
                future[page].append(step)
    history = [set(pages) for pages in steps]

    def frequency(page, step):
        return sum(page in history[s] for s in range(max(0, step - FREQ_WINDOW), step))

    # Larger (step, was_hit, -page_id) means more recent in our LRU convention.
    resident = {}
    admitted = {}
    last_rank = {}
    for step, pages in enumerate(steps):
        selected = set(pages)
        if len(selected) != len(pages) or len(selected) > budget:
            raise ValueError(f"duplicate selection or insufficient budget at step {step}")
        hits = selected.intersection(resident)
        missing = selected - hits
        if policy == "belady":
            for page in selected:
                future[page].popleft()
        evict_count = max(0, len(resident) + len(missing) - budget)
        if evict_count:
            candidates = resident.keys() - selected
            if policy == "lru":
                key = resident.__getitem__
            elif policy == "fifo":
                def key(page):
                    return admitted[page], -page
            elif policy == "rank":
                def key(page):
                    return (-last_rank[page],) + resident[page]
            elif policy == "score":
                def key(page):
                    bonus = delta * (last_rank[page] <= top_quartile)
                    bonus += beta * min(frequency(page, step), FREQ_CAP)
                    return (resident[page][0] + bonus,) + resident[page]
            else:
                def key(page):
                    next_step = future[page][0] if future[page] else len(steps)
                    return -next_step, page
            for page in sorted(candidates, key=key)[:evict_count]:
                del resident[page], admitted[page], last_rank[page]
        for rank, page in enumerate(pages, 1):
            resident[page] = (step, page in hits, -page)
            last_rank[page] = rank
            if page in missing:
                admitted[page] = step
        yield hits, missing, frozenset(resident)


def replay_stream(steps, budget, policy, *, delta=0, beta=0, top_k=None):
    """Count accesses and misses for one request/layer stream."""
    seen = set()
    accesses = misses = 0
    for hits, missing, _ in iter_replay(steps, budget, policy,
                                        delta=delta, beta=beta, top_k=top_k):
        accesses += len(hits) + len(missing)
        misses += len(missing)
        seen.update(hits, missing)
    return Counter(accesses=accesses, misses=misses, compulsory_misses=len(seen),
                   repeat_misses=misses - len(seen), repeat_accesses=accesses - len(seen))


def parse_policy(spec, delta=0, beta=0):
    """Map a CLI name to (label, policy, delta, beta).

    rf is recency plus frequency (delta=0), rrank is recency plus rank (beta=0) and
    full uses both weights. score:D:B names explicit weights, for example score:2:1.
    """
    fixed = {"lru": "lru", "fifo": "fifo", "rank": "rank", "belady": "belady"}
    presets = {"rf": (0, beta), "rrank": (delta, 0), "full": (delta, beta)}
    if spec in fixed:
        return spec, fixed[spec], None, None
    if spec in presets:
        return (spec, "score") + presets[spec]
    parts = spec.split(":")
    if len(parts) == 3 and parts[0] == "score":
        try:
            weights = [int(p) if p.isdigit() else float(p) for p in parts[1:]]
        except ValueError:
            weights = None
        if weights is not None:
            return (spec, "score") + tuple(weights)
    raise ValueError(f"unknown policy {spec!r}; use {', '.join([*fixed, *presets])} or score:D:B")


def compare(streams, budgets, policies=("lru", "belady"), *, delta=0, beta=0, top_k=None):
    """Pool counts across streams for each budget and policy, in the order given."""
    if top_k is None:
        top_k = infer_top_k(streams)
    specs = [parse_policy(spec, delta, beta) for spec in policies]
    rows = []
    for budget in budgets:
        for label, policy, d, b in specs:
            counts = Counter()
            for steps in streams.values():
                counts.update(replay_stream(steps, budget, policy,
                                            delta=d or 0, beta=b or 0, top_k=top_k))
            rows.append(dict(
                budget=budget, policy=label, delta=d, beta=b, top_k=top_k,
                accesses=counts["accesses"], misses=counts["misses"],
                compulsory_misses=counts["compulsory_misses"],
                repeat_misses=counts["repeat_misses"], repeat_accesses=counts["repeat_accesses"],
                miss_rate=counts["misses"] / counts["accesses"] if counts["accesses"] else None,
                repeat_miss_rate=(counts["repeat_misses"] / counts["repeat_accesses"]
                                  if counts["repeat_accesses"] else None),
            ))
    return rows


def save_comparison(rows, out_dir, title):
    """Write comparison.csv, then comparison.png when Matplotlib is installed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed, so comparison.png was skipped.")
        return
    names = {"lru": "LRU", "belady": "Bélády", "fifo": "FIFO", "rank": "Rank only",
             "rf": "Recency + freq.", "rrank": "Recency + rank", "full": "Full"}
    styles = {"lru": "o-", "belady": "x--"}
    labels = list(dict.fromkeys(r["policy"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for axis, metric, label in zip(axes, ("miss_rate", "repeat_miss_rate"),
                                  ("All selected pages", "Previously selected pages")):
        for policy in labels:
            subset = sorted((r for r in rows if r["policy"] == policy), key=lambda r: r["budget"])
            rates = [float("nan") if r[metric] is None else 100 * r[metric] for r in subset]
            axis.plot([r["budget"] for r in subset], rates, styles.get(policy, "s:"),
                      label=names.get(policy, policy))
        axis.set(title=label, xlabel="Cache budget (pages per layer)", ylabel="Miss rate (%)")
        axis.set_xticks(sorted({r["budget"] for r in rows}))
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=.25)
        axis.legend()
    fig.suptitle(title)
    fig.savefig(out_dir / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--policies", nargs="+", default=["lru", "belady"],
                        help="lru fifo rank rf rrank full belady, or score:D:B")
    parser.add_argument("--delta", type=float, default=0,
                        help="decode steps added for a top-quartile last rank (rrank, full)")
    parser.add_argument("--beta", type=float, default=0,
                        help="decode steps per recent selection, capped at 4 (rf, full)")
    parser.add_argument("--top-k", type=int, help="selection width K; default is the widest step")
    parser.add_argument("--out", type=Path, default=Path("results/replay"))
    parser.add_argument("--no-save", action="store_true", help="print counts only")
    args = parser.parse_args()
    weights = [int(w) if float(w).is_integer() else w for w in (args.delta, args.beta)]
    try:
        rows = compare(read_trace(args.trace), args.budgets, args.policies,
                       delta=weights[0], beta=weights[1], top_k=args.top_k)
    except ValueError as error:
        parser.error(str(error))
    if not args.no_save:
        save_comparison(rows, args.out, args.trace.stem)
    for row in rows:
        print(f"B={row['budget']:>3}  {row['policy']:<8}  misses {row['misses']}/{row['accesses']}"
              f"  first {row['compulsory_misses']}  repeat {row['repeat_misses']}/{row['repeat_accesses']}")


if __name__ == "__main__":
    main()
