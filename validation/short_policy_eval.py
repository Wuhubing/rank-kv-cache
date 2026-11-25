"""Replay the personal short-trace experiment, using its original dev grid and bootstrap."""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import pooled, write_csv
from replay import read_trace, replay_stream

WEIGHTS = (0, 1, 2, 4, 8)
DIGESTS = {
    "development": "0d0656528229a10f0cb034780bd39388b97548840434b7bf3dea26bd9f89e306",
    "heldout": "24e5cfc913352ac8b85451dc77f0e4f72052b76584f449ad82b1ec8cc05c82f3",
}


def manifest(path, split):
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != DIGESTS[split]:
            raise ValueError(f"Unexpected {split} archive")
    with ZipFile(path) as z:
        return json.loads(z.read("manifest.json"))


def replay_document(job):
    path, index, doc, configs = job
    rid = doc["request_id"]
    with TemporaryDirectory() as tmp, ZipFile(path) as z:
        trace = Path(tmp) / "trace.jsonl"
        trace.write_bytes(z.read(f"{index:02d}.jsonl"))
        streams = read_trace(trace)
    if set(streams) != {(rid, layer) for layer in range(28)}:
        raise ValueError(f"Invalid layer or document identity for {rid}")
    nsteps = len(streams[rid, 0])
    totals = defaultdict(Counter)
    for steps in streams.values():
        if len(steps) != nsteps or any(len(pages) != 64 for pages in steps):
            raise ValueError(f"Invalid trace dimensions for {rid}")
        if any(b == 64 for b, *_ in configs):
            # At B=K the previous selection is the entire cache.
            expected = 64 + sum(len(set(b).difference(a)) for a, b in zip(steps, steps[1:]))
        for budget, label, policy, delta, beta in configs:
            c = replay_stream(steps, budget, policy, delta=delta, beta=beta, top_k=64)
            if budget == 64 and c["misses"] != expected:
                raise ValueError(f"B=K invariant failed for {rid}, {label}")
            totals[budget, label].update(c)
    return rid, nsteps, totals


def select_weights(grid):
    def key(pair):
        return grid[pair]["misses"], sum(pair), pair[0]
    return (min((p for p in grid if p[0] == 0), key=key),
            min((p for p in grid if p[1] == 0), key=key), min(grid, key=key))


def comparisons(counts, docs, seed):
    groups = defaultdict(list)
    for i, doc in enumerate(docs):
        groups[doc["request_id"].split("/")[0], doc["length_band_start"]].append(i)
    rng = random.Random(seed)
    # Same stratum order, random draws and percentile convention as the personal H2 run.
    sample = np.array([[i for group in groups.values() for i in rng.choices(group, k=len(group))]
                       for _ in range(10000)])
    def interval(x):
        ordered = np.sort(x)
        return ordered[int(.025*(len(x)-1))], ordered[int(.975*(len(x)-1))]
    out = []
    for budget in (128, 256):
        for base in ("lru", "rf"):
            x, y = (np.array([counts[budget][p][d["request_id"]]["repeat_misses"] for d in docs])
                    for p in (base, "full"))
            delta = x-y
            saved = delta[sample].sum(axis=1)
            low, high = interval(saved)
            lo_rel, hi_rel = interval(saved / x[sample].sum(axis=1))
            out.append(dict(budget=budget, baseline=base, policy="full", saved=delta.sum(),
                            saved_low=low, saved_high=high, relative=delta.sum()/x.sum(),
                            relative_low=lo_rel, relative_high=hi_rel,
                            wins=(delta>0).sum(), ties=(delta==0).sum(), losses=(delta<0).sum()))
    return out


def plot_primary(counts, docs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    labels = {"fifo": "FIFO", "rank_only": "Last rank only", "lru": "LRU",
              "rrank": "Recency + rank", "rf": "Recency + frequency",
              "full": "Recency + frequency + rank", "belady": "Bélády (oracle)"}
    fig, (totals, paired) = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained")
    for i, (policy, label) in enumerate(labels.items()):
        value = sum(counts[policy][d]["repeat_misses"] for d in docs)
        color = "#b25025" if policy == "full" else "#456d8e"
        totals.barh(i, value, color=color)
        totals.text(value, i, f"  {value:,}", va="center", fontsize=8)
    totals.set_yticks(range(len(labels)), labels.values())
    totals.invert_yaxis()
    totals.set_xlim(0, totals.get_xlim()[1]*1.23)
    totals.xaxis.set_major_locator(MaxNLocator(nbins=4))
    totals.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    totals.set_title("Repeat misses by policy")
    totals.set_xlabel("Repeat misses across 40 documents")
    saved = sorted(counts["rf"][d]["repeat_misses"] - counts["full"][d]["repeat_misses"]
                   for d in docs)
    paired.bar(range(1, len(docs)+1), saved, color=["#b25025" if v < 0 else "#456d8e" for v in saved])
    paired.axhline(0, lw=.7, color="#444444")
    paired.set_title("Adding rank at fixed frequency weight")
    paired.set_ylabel("Repeat misses saved")
    paired.set_xlabel("Held-out documents, sorted by savings")
    fig.suptitle("Short-trace KV cache replay · 128 pages per layer")
    fig.savefig(path, dpi=175)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("development", type=Path)
    ap.add_argument("heldout", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    dev, held = (manifest(getattr(args, s), s) for s in DIGESTS)
    for key in ("request_id", "source_sha256"):
        a, b = ({d[key] for d in m[s]} for m, s in ((dev, "development"), (held, "heldout")))
        if len(a) != 20 or len(b) != 40 or a & b:
            raise ValueError(f"Split mismatch: {key}")
    if set(held["reserved_development_ids"]) != {d["request_id"] for d in dev["development"]}:
        raise ValueError("Development identities differ from the held-out manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        configs = [(128, p, "score", *p) for p in product(WEIGHTS, repeat=2)]
        jobs = [(args.development, i, d, configs) for i, d in enumerate(dev["development"], 1)]
        grid = defaultdict(Counter)
        for _, _, totals in pool.map(replay_document, jobs):
            for (_, pair), c in totals.items():
                grid[pair].update(c)
        write_csv(args.out/"development-grid.csv", [dict(delta=d, beta=b, **c)
                                                   for (d,b), c in sorted(grid.items())])
        rf, rrank, full = select_weights(grid)
        if (rf, full) != ((0, 2), (4, 2)):
            raise ValueError("Development tuning differs from the personal archived lock")
        print(f"25 configurations replayed. Locked R+F {rf}, R+rank {rrank}, full {full}.", flush=True)
        settings = {"grid": WEIGHTS, "primary_budget": 128, "rf": rf, "rrank": rrank, "full": full,
                    "tie_break": ["misses", "delta+beta", "delta"], "sha256": DIGESTS,
                    "dev_gpu": dev["gpu"], "heldout_gpu": held["gpu"]}
        (args.out/"replay-lock.json").write_text(json.dumps(settings, indent=2)+"\n")
        policies = [("fifo","fifo",0,0), ("lru","lru",0,0), ("rank_only","rank",0,0),
                    ("rf","score",*rf), ("rrank","score",*rrank), ("full","score",*full),
                    ("belady","belady",0,0)]
        configs = [(b, *p) for b in (64, 128, 256) for p in policies]
        jobs = [(args.heldout, i, d, configs) for i, d in enumerate(held["heldout"], 1)]
        counts = defaultdict(lambda: defaultdict(dict))
        doc_rows = []
        for i, (rid, steps, totals) in enumerate(pool.map(replay_document, jobs), 1):
            for (budget, name), c in totals.items():
                if c["misses"] < totals[budget,"belady"]["misses"]:
                    raise ValueError(f"Oracle bound failed for {rid}")
                c["steps"] = steps
                counts[budget][name][rid] = c
                doc_rows.append(dict(doc=rid, budget=budget, policy=name, **c))
            if i % 10 == 0:
                print(f"Held-out replay {i}/40.", flush=True)
    write_csv(args.out/"per-document.csv", doc_rows)
    docs = [d["request_id"] for d in held["heldout"]]
    summary = []
    for budget, by_policy in counts.items():
        for name in by_policy:
            metrics = pooled(counts, budget, name, docs, 65536)
            lru = sum(counts[budget]["lru"][d]["misses"] for d in docs)
            oracle = sum(counts[budget]["belady"][d]["misses"] for d in docs)
            summary.append(dict(budget=budget, policy=name, **metrics,
                                G=(lru-metrics["misses"])/(lru-oracle) if lru>oracle else None))
    write_csv(args.out/"policies.csv", summary)
    write_csv(args.out/"paired.csv", comparisons(counts, held["heldout"], dev["seed"]))
    plot_primary(counts[128], docs, args.out/"primary.png")


if __name__ == "__main__":
    main()
