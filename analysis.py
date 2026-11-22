"""H1: does a page's selection rank predict reuse beyond recency and frequency?

For every page selected at decode step t we record its rank at t (its most recent
rank), the steps since its previous selection (recency), its selections in the eight
steps before t (frequency), and whether it is selected again within h steps. Only
steps followed by max(horizons) observed steps are used, so every horizon is measured
on the same observations. The effect is the top-quartile minus bottom-quartile reuse
probability within frequency x recency strata, pooled with stratum-size weights. A
bootstrap that resamples whole requests (source documents) gives a 95% interval.

The core uses the standard library. Matplotlib is optional for the figure.
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from replay import read_trace

HORIZONS = (1, 2, 4, 8)
FREQ_WINDOW = 8
FEATURES = ("request_id", "task", "layer_id", "decode_step", "page_id", "rank", "n_selected",
            "quartile", "recency", "recency_bin", "frequency")


def recency_bin(recency):
    """None means the page was never selected before in its request and layer."""
    if recency is None:
        return "new"
    for limit, label in ((1, "1"), (2, "2"), (4, "3-4"), (8, "5-8")):
        if recency <= limit:
            return label
    return ">8"


def quartile(rank, n_selected):
    """Top is rank <= ceil(n/4), bottom is rank > n - ceil(n/4). One page is neither."""
    if n_selected < 2:
        return None
    width = -(-n_selected // 4)
    if rank <= width:
        return "top"
    if rank > n_selected - width:
        return "bottom"
    return "middle"


def observations(streams, horizons=HORIZONS, window=FREQ_WINDOW):
    """Yield one dict per selected page at each step with a complete future window."""
    if not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("horizons must be positive integers")
    longest = max(horizons)
    for (request, layer), steps in sorted(streams.items()):
        history = [set(pages) for pages in steps]
        next_use = [None] * len(steps)
        upcoming = {}
        for t in range(len(steps) - 1, -1, -1):
            next_use[t] = {page: upcoming.get(page) for page in steps[t]}
            upcoming.update((page, t) for page in steps[t])
        last_seen = {}
        for t, pages in enumerate(steps):
            if t + longest < len(steps):
                for rank, page in enumerate(pages, 1):
                    previous = last_seen.get(page)
                    recency = None if previous is None else t - previous
                    following = next_use[t][page]
                    row = dict(
                        request_id=request, task=request.split("/", 1)[0], layer_id=layer,
                        decode_step=t, page_id=page, rank=rank, n_selected=len(pages),
                        quartile=quartile(rank, len(pages)), recency=recency,
                        recency_bin=recency_bin(recency),
                        frequency=sum(page in history[s] for s in range(max(0, t - window), t)),
                    )
                    for h in horizons:
                        row[f"reuse_{h}"] = following is not None and following - t <= h
                    yield row
            for page in pages:
                last_seen[page] = t


def tally(rows, horizon, cluster="request"):
    """Count (top n, top reused, bottom n, bottom reused) per cluster and stratum."""
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    for row in rows:
        if row["quartile"] not in ("top", "bottom"):
            continue
        unit = row["request_id"] if cluster == "request" else (row["request_id"], row["layer_id"])
        cell = counts[unit][row["frequency"], row["recency_bin"]]
        offset = 0 if row["quartile"] == "top" else 2
        cell[offset] += 1
        cell[offset + 1] += row[f"reuse_{horizon}"]
    return counts


def stratified_effect(clusters):
    """Pool cluster tables. Return (effect, strata used, raw top rate, raw bottom rate)."""
    merged = defaultdict(lambda: [0, 0, 0, 0])
    for table in clusters:
        for stratum, cell in table.items():
            total = merged[stratum]
            for i in range(4):
                total[i] += cell[i]
    numerator = weight = used = 0
    top = [0, 0]
    bottom = [0, 0]
    for top_n, top_hit, bottom_n, bottom_hit in merged.values():
        top[0] += top_n
        top[1] += top_hit
        bottom[0] += bottom_n
        bottom[1] += bottom_hit
        if top_n and bottom_n:
            size = top_n + bottom_n
            numerator += size * (top_hit / top_n - bottom_hit / bottom_n)
            weight += size
            used += 1
    return (numerator / weight if weight else None, used,
            top[1] / top[0] if top[0] else None, bottom[1] / bottom[0] if bottom[0] else None)


def bootstrap_interval(counts, replicates=1000, seed=0, level=0.95):
    """Percentile interval from resampling clusters with replacement; None below 2."""
    units = sorted(counts, key=repr)
    if len(units) < 2 or replicates < 1:
        return None, None
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        effect = stratified_effect(counts[rng.choice(units)] for _ in units)[0]
        if effect is not None:
            estimates.append(effect)
    if not estimates:
        return None, None
    estimates.sort()
    tail = (1 - level) / 2
    pick = lambda q: estimates[min(len(estimates) - 1, int(q * len(estimates)))]
    return pick(tail), pick(1 - tail)


def layer_group(layer, num_layers, groups):
    """Split layers 0..num_layers-1 into contiguous groups of near-equal size."""
    g = layer * groups // num_layers
    first = -(-g * num_layers // groups)
    last = -(-(g + 1) * num_layers // groups) - 1
    return f"layers {first}-{last}"


def analyze(streams, horizons=HORIZONS, *, by=(), layer_groups=3, cluster="request",
            replicates=1000, seed=0):
    """Return (effect rows, rank-curve rows, observation count)."""
    if cluster not in ("request", "stream"):
        raise ValueError("cluster must be request or stream")
    rows = list(observations(streams, horizons))
    num_layers = max(layer for _, layer in streams) + 1
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        if "task" in by:
            groups[f"task {row['task']}"].append(row)
        if "layer-group" in by:
            groups[layer_group(row["layer_id"], num_layers, layer_groups)].append(row)
    effects = []
    for name, members in groups.items():
        for h in horizons:
            counts = tally(members, h, cluster)
            effect, used, top, bottom = stratified_effect(counts.values())
            low, high = bootstrap_interval(counts, replicates, seed)
            effects.append(dict(group=name, horizon=h, observations=len(members),
                                clusters=len(counts), strata_used=used, effect=effect,
                                ci_low=low, ci_high=high, raw_top=top, raw_bottom=bottom))
    curve = defaultdict(lambda: [0, 0])
    for row in rows:
        decile = min(9, (row["rank"] - 1) * 10 // row["n_selected"])
        for h in horizons:
            cell = curve[h, decile]
            cell[0] += 1
            cell[1] += row[f"reuse_{h}"]
    curve_rows = [dict(horizon=h, rank_decile=d + 1, observations=n, reuse_rate=hit / n)
                  for (h, d), (n, hit) in sorted(curve.items())]
    return effects, curve_rows, rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_figure(effects, curve, out_dir, title):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed, so h1.png was skipped.")
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for h in sorted({r["horizon"] for r in curve}):
        subset = [r for r in curve if r["horizon"] == h]
        left.plot([r["rank_decile"] for r in subset], [100 * r["reuse_rate"] for r in subset],
                  "o-", label=f"within {h}")
    left.set(title="Reuse by rank decile (1 = best)", xlabel="Rank decile",
             ylabel="Selected again (%)", xticks=range(1, 11))
    left.set_ylim(bottom=0)
    left.legend()
    pooled = [r for r in effects if r["group"] == "all" and r["effect"] is not None]
    xs = [r["horizon"] for r in pooled]
    ys = [100 * r["effect"] for r in pooled]
    right.plot(xs, ys, "o-", label="Stratified top - bottom")
    bounded = [r for r in pooled if r["ci_low"] is not None]
    if bounded:
        right.fill_between([r["horizon"] for r in bounded], [100 * r["ci_low"] for r in bounded],
                           [100 * r["ci_high"] for r in bounded], alpha=.2, label="95% CI")
    right.axhline(0, color="gray", lw=.8)
    right.set(title="Rank effect after frequency x recency", xlabel="Horizon (steps)",
              ylabel="Percentage points", xticks=xs)
    right.legend()
    fig.suptitle(title)
    fig.savefig(Path(out_dir) / "h1.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--by", nargs="*", default=[], choices=["task", "layer-group"],
                        help="also report each task (request_id prefix before /) or layer group")
    parser.add_argument("--layer-groups", type=int, default=3)
    parser.add_argument("--cluster", choices=["request", "stream"], default="request",
                        help="bootstrap unit; stream (request, layer) ignores layer correlation")
    parser.add_argument("--bootstrap", type=int, default=1000, help="replicates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/analysis"))
    parser.add_argument("--features", action="store_true", help="also write features.csv")
    args = parser.parse_args()
    try:
        effects, curve, rows = analyze(
            read_trace(args.trace), tuple(args.horizons), by=args.by,
            layer_groups=args.layer_groups, cluster=args.cluster,
            replicates=args.bootstrap, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))
    if not rows:
        print(f"No step has {max(args.horizons)} later steps in its stream, so nothing to "
              "analyze. Use a longer trace or smaller --horizons.")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "h1_effects.csv", effects)
    write_csv(args.out / "rank_curve.csv", curve)
    if args.features:
        columns = FEATURES + tuple(f"reuse_{h}" for h in args.horizons)
        write_csv(args.out / "features.csv", [{k: r[k] for k in columns} for r in rows])
    save_figure(effects, curve, args.out, args.trace.stem)
    pct = lambda x: "   n/a" if x is None else f"{100 * x:+6.2f}"
    print(f"{len(rows)} observations. Effects in percentage points, top minus bottom quartile.")
    for r in effects:
        interval = ("CI n/a (needs 2+ clusters)" if r["ci_low"] is None
                    else f"CI [{pct(r['ci_low'])}, {pct(r['ci_high'])}]")
        print(f"{r['group']:<16} h={r['horizon']}  effect {pct(r['effect'])}  {interval}"
              f"  strata {r['strata_used']}  clusters {r['clusters']}")


if __name__ == "__main__":
    main()
