"""H1: does a page's selection rank predict reuse beyond recency and frequency?

Three samples define which pages are observed at decode step t:
  selected   pages selected at t; rank is their rank at t (H1-selected, for contrast)
  candidate  pages selected before t but not at t: the pages any replacement policy
             chooses among (H1-candidate, the primary sample; policy independent)
  lru        candidates that an LRU cache of --lru-budget slots actually holds when
             step t arrives, i.e. the residents LRU must choose among

For each observation: last rank is the rank at the page's most recent selection tau
(tau = t for selected pages), recency = t - tau (None for a first selection), frequency
counts selections in steps t-8..t-1, and reuse_h records reselection in t+1..t+h. Only
steps followed by max(horizons) observed steps are used, so every horizon is measured
on the same observations. The effect is the top-quartile minus bottom-quartile reuse
probability within frequency x recency strata, pooled with stratum-size weights. A
bootstrap that resamples whole requests (source documents) gives a 95% interval.

The core uses the standard library. Matplotlib is optional for the figure.
"""
import argparse
import csv
import random
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from replay import iter_replay, read_trace

HORIZONS = (1, 2, 4, 8)
FREQ_WINDOW = 8
SAMPLES = ("selected", "candidate", "lru")
FEATURES = ("sample", "request_id", "task", "layer_id", "decode_step", "page_id", "rank",
            "n_selected", "quartile", "recency", "recency_bin", "frequency")
RECENCY_LIMITS = ((1, "1"), (2, "2"), (4, "3-4"), (8, "5-8"), (16, "9-16"), (32, "17-32"),
                  (64, "33-64"))


def recency_bin(recency):
    """None means the page was never selected before in its request and layer."""
    if recency is None:
        return "new"
    for limit, label in RECENCY_LIMITS:
        if recency <= limit:
            return label
    return ">64"


def quartile(rank, n_selected):
    """Top is rank <= ceil(n/4), bottom is rank > n - ceil(n/4). One page is neither.

    n is the width of the selection that assigned the rank (K once K pages exist).
    """
    if n_selected < 2:
        return None
    width = -(-n_selected // 4)
    if rank <= width:
        return "top"
    if rank > n_selected - width:
        return "bottom"
    return "middle"


def observations(streams, horizons=HORIZONS, window=FREQ_WINDOW, sample="selected",
                 budget=None):
    """Yield one dict per observed page at each step with a complete future window."""
    if not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("horizons must be positive integers")
    if sample not in SAMPLES:
        raise ValueError(f"sample must be one of {SAMPLES}")
    if sample == "lru" and (type(budget) is not int or budget < 1):
        raise ValueError("the lru sample needs a positive integer budget")
    longest = max(horizons)
    for (request, layer), steps in sorted(streams.items()):
        history = [set(pages) for pages in steps]
        uses = defaultdict(list)
        for t, pages in enumerate(steps):
            for page in pages:
                uses[page].append(t)
        if sample == "lru":
            residents = [r for _, _, r in iter_replay(steps, budget, "lru")]
        last = {}  # page -> (step, rank, selection width) at its latest selection
        for t, pages in enumerate(steps):
            if t + longest < len(steps):
                if sample == "selected":
                    pool = [(page, rank, len(pages), t - last[page][0] if page in last else None)
                            for rank, page in enumerate(pages, 1)]
                else:
                    held = last if sample == "candidate" else (residents[t - 1] if t else ())
                    pool = [(page, last[page][1], last[page][2], t - last[page][0])
                            for page in sorted(held) if page not in history[t]]
                for page, rank, width, recency in pool:
                    later = uses[page][bisect_right(uses[page], t):]
                    following = later[0] if later else None
                    row = dict(
                        sample=sample, request_id=request, task=request.split("/", 1)[0],
                        layer_id=layer, decode_step=t, page_id=page, rank=rank,
                        n_selected=width, quartile=quartile(rank, width), recency=recency,
                        recency_bin=recency_bin(recency),
                        frequency=sum(page in history[s] for s in range(max(0, t - window), t)),
                    )
                    for h in horizons:
                        row[f"reuse_{h}"] = following is not None and following - t <= h
                    yield row
            for rank, page in enumerate(pages, 1):
                last[page] = (t, rank, len(pages))


def new_table():
    return defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))


def add(table, row, horizon, cluster="request"):
    """Count (top n, top reused, bottom n, bottom reused) per cluster and stratum."""
    if row["quartile"] not in ("top", "bottom"):
        return
    unit = row["request_id"] if cluster == "request" else (row["request_id"], row["layer_id"])
    cell = table[unit][row["frequency"], row["recency_bin"]]
    offset = 0 if row["quartile"] == "top" else 2
    cell[offset] += 1
    cell[offset + 1] += row[f"reuse_{horizon}"]


def tally(rows, horizon, cluster="request"):
    table = new_table()
    for row in rows:
        add(table, row, horizon, cluster)
    return table


def stratified_effect(clusters, key=None):
    """Pool cluster tables. Return (effect, strata used, raw top rate, raw bottom rate).

    key maps a (frequency, recency bin) stratum to a coarser one, for example
    lambda s: s[1] to stratify by recency alone.
    """
    merged = defaultdict(lambda: [0, 0, 0, 0])
    for table in clusters:
        for stratum, cell in table.items():
            total = merged[key(stratum) if key else stratum]
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


def analyze(streams, horizons=HORIZONS, *, samples=("selected",), budget=128, by=(),
            layer_groups=3, cluster="request", replicates=1000, seed=0, on_row=None):
    """Return (effect rows, rank-curve rows, observations per sample) in one pass."""
    if cluster not in ("request", "stream"):
        raise ValueError("cluster must be request or stream")
    num_layers = max(layer for _, layer in streams) + 1
    tables = defaultdict(new_table)
    sizes = defaultdict(int)
    curve = defaultdict(lambda: [0, 0])
    for sample in samples:
        for row in observations(streams, horizons, sample=sample, budget=budget):
            if on_row:
                on_row(row)
            groups = ["all"]
            if "task" in by:
                groups.append(f"task {row['task']}")
            if "layer-group" in by:
                groups.append(layer_group(row["layer_id"], num_layers, layer_groups))
            for group in groups:
                sizes[sample, group] += 1
                for h in horizons:
                    add(tables[sample, group, h], row, h, cluster)
            decile = min(9, (row["rank"] - 1) * 10 // row["n_selected"])
            for h in horizons:
                cell = curve[sample, h, decile]
                cell[0] += 1
                cell[1] += row[f"reuse_{h}"]
    effects = []
    for (sample, group), size in sizes.items():
        for h in horizons:
            counts = tables[sample, group, h]
            effect, used, top, bottom = stratified_effect(counts.values())
            by_recency = stratified_effect(counts.values(), key=lambda s: s[1])[0]
            low, high = bootstrap_interval(counts, replicates, seed)
            effects.append(dict(
                sample=sample, group=group, horizon=h, observations=size,
                clusters=len(counts), strata_used=used, effect=effect, ci_low=low, ci_high=high,
                effect_recency_only=by_recency, raw_top=top, raw_bottom=bottom,
                raw_difference=None if top is None or bottom is None else top - bottom))
    curve_rows = [dict(sample=s, horizon=h, rank_decile=d + 1, observations=n, reuse_rate=hit / n)
                  for (s, h, d), (n, hit) in sorted(curve.items(),
                                                    key=lambda kv: (samples.index(kv[0][0]),
                                                                    kv[0][1:]))]
    return effects, curve_rows, {s: sizes.get((s, "all"), 0) for s in samples}


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
    samples = list(dict.fromkeys(r["sample"] for r in curve))
    fig, axes = plt.subplots(2, len(samples), figsize=(4.5 * len(samples), 7.2),
                             layout="constrained", squeeze=False)
    for column, sample in enumerate(samples):
        top, bottom = axes[0][column], axes[1][column]
        for h in sorted({r["horizon"] for r in curve}):
            subset = [r for r in curve if r["sample"] == sample and r["horizon"] == h]
            top.plot([r["rank_decile"] for r in subset], [100 * r["reuse_rate"] for r in subset],
                     "o-", label=f"within {h}")
        top.set(title=f"{sample}: reuse by last-rank decile", xlabel="Rank decile (1 = best)",
                ylabel="Selected again (%)", xticks=range(1, 11))
        top.set_ylim(0, 100)
        top.legend()
        pooled = [r for r in effects if r["sample"] == sample and r["group"] == "all"
                  and r["effect"] is not None]
        xs = [r["horizon"] for r in pooled]
        bottom.plot(xs, [100 * r["raw_difference"] for r in pooled], "s:", label="Raw")
        bottom.plot(xs, [100 * r["effect_recency_only"] for r in pooled], "^--",
                    label="Within recency")
        bottom.plot(xs, [100 * r["effect"] for r in pooled], "o-", label="Within freq. x recency")
        bounded = [r for r in pooled if r["ci_low"] is not None]
        if bounded:
            bottom.fill_between([r["horizon"] for r in bounded],
                                [100 * r["ci_low"] for r in bounded],
                                [100 * r["ci_high"] for r in bounded], alpha=.2, label="95% CI")
        bottom.axhline(0, color="gray", lw=.8)
        bottom.set(title=f"{sample}: top minus bottom quartile", xlabel="Horizon (steps)",
                   ylabel="Percentage points", xticks=xs)
        bottom.legend()
    fig.suptitle(title)
    fig.savefig(Path(out_dir) / "h1.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--samples", nargs="+", default=["selected", "candidate"],
                        choices=SAMPLES)
    parser.add_argument("--lru-budget", type=int, default=128,
                        help="cache slots for the lru sample")
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
    samples = tuple(dict.fromkeys(args.samples))
    args.out.mkdir(parents=True, exist_ok=True)
    columns = FEATURES + tuple(f"reuse_{h}" for h in args.horizons)
    feature_file = writer = None
    if args.features:
        feature_file = (args.out / "features.csv").open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(feature_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
    try:
        effects, curve, sizes = analyze(
            read_trace(args.trace), tuple(args.horizons), samples=samples,
            budget=args.lru_budget, by=args.by, layer_groups=args.layer_groups,
            cluster=args.cluster, replicates=args.bootstrap, seed=args.seed,
            on_row=writer.writerow if writer else None)
    except ValueError as error:
        parser.error(str(error))
    finally:
        if feature_file:
            feature_file.close()
    if not any(sizes.values()):
        print(f"No step has {max(args.horizons)} later steps in its stream, so nothing to "
              "analyze. Use a longer trace or smaller --horizons.")
        return
    write_csv(args.out / "h1_effects.csv", effects)
    write_csv(args.out / "rank_curve.csv", curve)
    save_figure(effects, curve, args.out, args.trace.stem)
    pct = lambda x: "   n/a" if x is None else f"{100 * x:+6.2f}"
    print("Observations: " + ", ".join(f"{s} {n}" for s, n in sizes.items())
          + ". Effects are top minus bottom quartile, in percentage points.")
    for r in effects:
        interval = ("CI n/a (needs 2+ clusters)" if r["ci_low"] is None
                    else f"CI [{pct(r['ci_low'])}, {pct(r['ci_high'])}]")
        print(f"{r['sample']:<9} {r['group']:<14} h={r['horizon']}  raw {pct(r['raw_difference'])}"
              f"  recency {pct(r['effect_recency_only'])}  freq x recency {pct(r['effect'])}"
              f"  {interval}  clusters {r['clusters']}")


if __name__ == "__main__":
    main()
