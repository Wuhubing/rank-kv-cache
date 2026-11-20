"""Replay batched sparse-KV page selections. Core replay uses the standard library."""
import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path


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


def replay_stream(steps, budget, policy):
    """Start one empty cache. A step is a set of simultaneous, compulsory admissions."""
    if policy not in ("lru", "belady") or type(budget) is not int or budget < 1:
        raise ValueError("expected policy lru or belady and a positive integer budget")
    future = defaultdict(deque)
    if policy == "belady":
        for step, pages in enumerate(steps):
            for page in pages:
                future[page].append(step)

    # Larger (step, was_hit, -page_id) means more recent in our LRU convention.
    resident = {}
    seen = set()
    accesses = misses = 0
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
                victims = sorted(candidates, key=resident.__getitem__)
            else:
                def farthest_first(page):
                    next_step = future[page][0] if future[page] else len(steps)
                    return -next_step, page
                victims = sorted(candidates, key=farthest_first)
            for page in victims[:evict_count]:
                del resident[page]
        for page in selected:
            resident[page] = (step, page in hits, -page)
        accesses += len(selected)
        misses += len(missing)
        seen.update(selected)

    return Counter(accesses=accesses, misses=misses, compulsory_misses=len(seen),
                   repeat_misses=misses - len(seen), repeat_accesses=accesses - len(seen))


def compare(streams, budgets):
    rows = []
    for budget in budgets:
        for policy in ("lru", "belady"):
            counts = Counter()
            for steps in streams.values():
                counts.update(replay_stream(steps, budget, policy))
            rows.append(dict(
                budget=budget, policy=policy, **counts,
                miss_rate=counts["misses"] / counts["accesses"] if counts["accesses"] else None,
                repeat_miss_rate=(counts["repeat_misses"] / counts["repeat_accesses"]
                                  if counts["repeat_accesses"] else None),
            ))
    return rows


def save_comparison(rows, out_dir, title):
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for axis, metric, label in zip(axes, ("miss_rate", "repeat_miss_rate"),
                                  ("All selected pages", "Previously selected pages")):
        for policy, name, style in (("lru", "LRU", "o-"), ("belady", "Bélády", "x--")):
            subset = sorted((r for r in rows if r["policy"] == policy), key=lambda r: r["budget"])
            rates = [float("nan") if r[metric] is None else 100 * r[metric] for r in subset]
            axis.plot([r["budget"] for r in subset], rates, style, label=name)
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
    parser.add_argument("--out", type=Path, default=Path("results/replay"))
    args = parser.parse_args()
    rows = compare(read_trace(args.trace), args.budgets)
    save_comparison(rows, args.out, args.trace.stem)
    for row in rows:
        print(f"B={row['budget']:>3}  {row['policy']:<6}  misses {row['misses']}/{row['accesses']}"
              f"  first {row['compulsory_misses']}  repeat {row['repeat_misses']}/{row['repeat_accesses']}")


if __name__ == "__main__":
    main()
