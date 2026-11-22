"""Run the preregistered evaluation on one run directory, exactly as configured.

Reads splits/preregistration.json, replays every policy at every budget on each
document, and writes per-document counts, pooled tables with document-bootstrap
intervals, the two success criteria, per-group (task, band) tables and the three H1
samples. Run it once per trace on held-out data; the same command also works on dev.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from analysis import analyze
from replay import read_trace, replay_stream, request_group


def per_document(streams, prompts, prereg):
    """counts[budget][policy][doc] -> Counter(misses, repeat_misses, accesses, steps)."""
    from gap_analysis import History, Logistic, model_score, replay_scored
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    models = {p["name"]: model_score(Logistic.from_dict(p["model"]))
              for p in prereg["policies"] if p["policy"] == "logistic"}
    for budget in prereg["budgets"]:
        for (request, layer), steps in sorted(streams.items()):
            for p in prereg["policies"]:
                if p["policy"] == "logistic":
                    misses, first, accesses = replay_scored(
                        steps, budget, History(layer, prompts[request]), score=models[p["name"]])
                    c = Counter(misses=misses, compulsory_misses=first, accesses=accesses,
                                repeat_misses=misses - first)
                else:
                    c = replay_stream(steps, budget, p["policy"], delta=p.get("delta", 0),
                                      beta=p.get("beta", 0), top_k=prereg["top_k"])
                cell = counts[budget][p["name"]][request]
                for key in ("misses", "repeat_misses", "accesses", "compulsory_misses"):
                    cell[key] += c[key]
                cell["steps"] = len(steps)
        print(f"  budget {budget} replayed", flush=True)
    return counts


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def pooled(counts, budget, policy, docs, page_bytes):
    c = Counter()
    steps = 0
    for d in docs:
        c.update(counts[budget][policy][d])
        steps += counts[budget][policy][d]["steps"]
    return dict(misses=c["misses"], repeat_misses=c["repeat_misses"], accesses=c["accesses"],
                compulsory_misses=c["compulsory_misses"],
                miss_rate=c["misses"] / c["accesses"],
                repeat_miss_rate=c["repeat_misses"] / (c["accesses"] - c["compulsory_misses"]),
                h2d_mib_per_token=c["misses"] * page_bytes / steps / 2 ** 20)


def table(counts, prereg, docs, rng_seed, label="all"):
    """Pooled metrics per budget and policy with document-bootstrap intervals."""
    rows = []
    names = [p["name"] for p in prereg["policies"]]
    reps = prereg["bootstrap"]["replicates"]
    for budget in prereg["budgets"]:
        rep = lambda name, sample: sum(counts[budget][name][d]["repeat_misses"] for d in sample)
        rng = random.Random(rng_seed)
        samples = [[rng.choice(docs) for _ in docs] for _ in range(reps)]
        for name in names:
            base = pooled(counts, budget, name, docs, prereg["page_bytes"])
            lru, bel, mine = rep("lru", docs), rep("belady", docs), rep(name, docs)
            gs, rs = [], []
            for s in samples:
                l, b, m = rep("lru", s), rep("belady", s), rep(name, s)
                if l > b:
                    gs.append((l - m) / (l - b))
                if l:
                    rs.append((l - m) / l)
            rows.append(dict(group=label, budget=budget, policy=name, **base,
                             G=(lru - mine) / (lru - bel) if lru > bel else None,
                             G_low=percentile(gs, .025), G_high=percentile(gs, .975),
                             repeat_reduction_vs_lru=(lru - mine) / lru if lru else None,
                             reduction_low=percentile(rs, .025), reduction_high=percentile(rs, .975),
                             documents=len(docs)))
    return rows


def success(counts, prereg, docs):
    budget = prereg["primary_budget"]
    rep = lambda name, sample: sum(counts[budget][name][d]["repeat_misses"] for d in sample)
    rng = random.Random(prereg["bootstrap"]["seed"])
    reductions, saved = [], []
    for _ in range(prereg["bootstrap"]["replicates"]):
        s = [rng.choice(docs) for _ in docs]
        reductions.append((rep("lru", s) - rep("full", s)) / rep("lru", s))
        saved.append(rep("rf", s) - rep("full", s))
    s1 = (rep("lru", docs) - rep("full", docs)) / rep("lru", docs)
    s2 = rep("rf", docs) - rep("full", docs)
    return dict(
        budget=budget,
        s1_full_vs_lru_repeat_reduction=s1, s1_ci=[percentile(reductions, .025),
                                                   percentile(reductions, .975)],
        s1_met=s1 >= 0.10,
        s2_repeat_misses_saved_full_vs_rf=s2,
        s2_ci=[percentile(saved, .025), percentile(saved, .975)],
        s2_relative_to_rf=s2 / rep("rf", docs),
        s2_met=percentile(saved, .025) > 0)


def write_csv(path, rows):
    rows = list(rows)
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--prereg", type=Path, default=Path("splits/preregistration.json"))
    parser.add_argument("--trace", default="trace.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.prereg.read_text())
    streams = read_trace(args.run_dir / args.trace)
    prompts = {}
    for path in (args.run_dir / "manifests").glob("*.json"):
        m = json.loads(path.read_text())
        prompts[m["request_id"]] = m["prompt_tokens"]
    docs = sorted({r for r, _ in streams})
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {len(docs)} documents from {args.run_dir / args.trace}")

    counts = per_document(streams, prompts, prereg)
    write_csv(args.out / "policies_by_doc.csv", [
        dict(budget=b, policy=p, doc=d, **counts[b][p][d])
        for b in counts for p in counts[b] for d in sorted(counts[b][p])])
    seed = prereg["bootstrap"]["seed"]
    rows = table(counts, prereg, docs, seed)
    write_csv(args.out / "policies.csv", rows)
    group_rows = []
    for kind in prereg["groups"]:
        for label in sorted({request_group(d, kind) for d in docs}):
            members = [d for d in docs if request_group(d, kind) == label]
            group_rows += table(counts, prereg, members, seed, f"{kind} {label}")
    write_csv(args.out / "policies_by_group.csv", group_rows)
    result = success(counts, prereg, docs)
    (args.out / "success.json").write_text(json.dumps(result, indent=2) + "\n")

    h1 = prereg["h1"]
    effects, curve, sizes = analyze(streams, tuple(h1["horizons"]), samples=tuple(h1["samples"]),
                                    budget=h1["lru_budget"], by=tuple(h1["by"]),
                                    cluster=h1["cluster"], replicates=h1["replicates"],
                                    seed=h1["seed"])
    write_csv(args.out / "h1_effects.csv", effects)
    write_csv(args.out / "rank_curve.csv", curve)

    pct = lambda x: "n/a" if x is None else f"{100 * x:.2f}%"
    for r in rows:
        print(f"B={r['budget']:<4}{r['policy']:<10} miss {pct(r['miss_rate'])}  repeat "
              f"{r['repeat_misses']:>7}  G {pct(r['G'])} [{pct(r['G_low'])}, {pct(r['G_high'])}]"
              f"  H2D {r['h2d_mib_per_token']:.2f} MiB/token")
    print(json.dumps(result, indent=2))
    print("H1 observations:", sizes)


if __name__ == "__main__":
    main()
