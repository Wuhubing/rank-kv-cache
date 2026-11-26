"""Measure rank/reuse and one-decision counterfactuals on the personal 20-document dev ZIP."""
import argparse
import csv
import hashlib
import json
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import layer_group, quartile
from gap_analysis import History, next_use, next_uses, replay_scored
from replay import FREQ_CAP, FREQ_WINDOW, read_trace, replay_stream

BUDGET, K, HORIZON, DELTA = 128, 64, 8, 4
DEV_SHA = "0d0656528229a10f0cb034780bd39388b97548840434b7bf3dea26bd9f89e306"
TALLY = ("top", "top_reused", "bottom", "bottom_reused")
EVENTS = ("steps", "evictions", "victims", "changed", "swaps", "complete_evictions",
          "complete_candidates", "complete_changed", "rank_first", "base_first", "same_step", "neither",
          "base_repeats", "rank_repeats")
SAMPLES = ("selected", "rf_candidates")


def count_quartile(cells, freq, rank, reused):
    group = quartile(rank, K)
    if group != "middle":
        offset = 0 if group == "top" else 2
        cells[freq, offset] += 1
        cells[freq, offset + 1] += int(reused)


def selected_reuse(steps, cells):
    counts = Counter()
    for t in range(len(steps) - HORIZON):
        future = set().union(*steps[t+1:t+HORIZON+1])
        for rank, page in enumerate(steps[t], 1):
            count_quartile(cells, counts[page], rank, page in future)
        counts.update(steps[t])
        if t >= FREQ_WINDOW:
            counts.subtract(steps[t-FREQ_WINDOW])


def decision_replay(steps, beta, cells=None, *, budget=BUDGET, width=K):
    """Both alternative victim sets start on the baseline path, with its exact LRU tiebreak."""
    uses = next_uses(steps)
    events = Counter(steps=len(steps))

    def score(candidates, t, history):
        # replay_scored supplies candidates already in LRU tie order.
        freqs = [len(history.steps[p]) - bisect_left(history.steps[p], t-FREQ_WINDOW)
                 for p in candidates]
        base = [history.steps[p][-1] + beta * min(f, FREQ_CAP) for p, f in zip(candidates, freqs)]
        bonus = [DELTA * (history.rank[p] <= (width+3)//4) for p in candidates]
        alternative = sorted(range(len(base)), key=lambda i: base[i] + bonus[i])
        alternative_order[:] = [candidates[i] for i in alternative]
        if cells is not None and t + HORIZON < len(steps):
            for p, f in zip(candidates, freqs):
                count_quartile(cells, f, history.rank[p], next_use(uses, p, t) <= t+HORIZON)
        return base

    def compare(t, candidates, victims):
        rank_victims = set(alternative_order[:len(victims)])
        kept_by_rank = set(victims) - rank_victims
        kept_by_base = rank_victims - set(victims)
        events.update(evictions=1, victims=len(victims), changed=bool(kept_by_rank),
                      swaps=len(kept_by_rank))
        if t + HORIZON >= len(steps):
            return
        events["complete_evictions"] += 1
        events["complete_candidates"] += len(candidates)
        if not kept_by_rank:
            return
        events["complete_changed"] += 1
        first_rank = min(next_use(uses, p, t) for p in kept_by_rank)
        first_base = min(next_use(uses, p, t) for p in kept_by_base)
        if min(first_rank, first_base) > t + HORIZON:
            outcome = "neither"
        elif first_rank == first_base:
            outcome = "same_step"
        else:
            outcome = "rank_first" if first_rank < first_base else "base_first"
        events[outcome] += 1

    alternative_order = []
    misses, compulsory, _ = replay_scored(steps, budget, History(0, 0), score, compare)
    events["base_repeats"] = misses - compulsory
    events["rank_repeats"] = replay_stream(steps, budget, "score", delta=DELTA, beta=beta,
                                           top_k=width)["repeat_misses"]
    return events


def write_csv(path, rows):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def collect(bundle):
    strata, events = [], []
    with ZipFile(bundle) as z, TemporaryDirectory() as tmp:
        manifest = json.loads(z.read("manifest.json"))
        docs = manifest["development"]
        if (len(docs) != 20 or len({d["source_sha256"] for d in docs}) != 20
                or manifest["top_k"] != K or manifest["page_size"] != 16):
            raise ValueError("Unexpected development manifest")
        for i, doc in enumerate(docs, 1):
            rid = doc["request_id"]
            path = Path(tmp) / "trace.jsonl"
            path.write_bytes(z.read(f"{i:02d}.jsonl"))
            streams = read_trace(path)
            if set(streams) != {(rid, l) for l in range(28)}:
                raise ValueError(f"Missing layers or incorrect document ID for {rid}")
            for (_, layer), steps in sorted(streams.items()):
                if any(len(pages) != K for pages in steps):
                    raise ValueError(f"Unexpected selection width in {rid}, layer {layer}")
                meta = dict(doc=rid, task=rid.split("/")[0], band=doc["length_band_start"],
                            layers=layer_group(layer, 28, 3))
                cells = {s: np.zeros((FREQ_WINDOW+1, 4), dtype=int) for s in SAMPLES}
                selected_reuse(steps, cells["selected"])
                for beta in (0, 2):
                    ev = decision_replay(steps, beta, cells["rf_candidates"] if beta == 2 else None)
                    events.append(dict(**meta, layer=layer, beta=beta, **{k:ev[k] for k in EVENTS}))
                for sample, counts in cells.items():
                    for f, values in enumerate(counts):
                        strata.append(dict(**meta, layer=layer, sample=sample, freq=f,
                                           **dict(zip(TALLY, values))))
            print(f"{i}/{len(docs)} {rid[:20]}", flush=True)
    return strata, events


def bootstrap_indices(docs, metadata):
    groups = defaultdict(list)
    for i, doc in enumerate(docs):
        groups[metadata[doc]].append(i)
    rng = np.random.default_rng(0)
    return np.concatenate([rng.choice(g, size=(2000, len(g))) for g in groups.values()], axis=1)


def effect(table):
    top, top_hits, bottom, bottom_hits = np.moveaxis(table, -1, 0)
    weights = np.where((top > 0) & (bottom > 0), top + bottom, 0)
    denom = weights.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = top_hits / np.maximum(top, 1) - bottom_hits / np.maximum(bottom, 1)
        return (weights * diff).sum(axis=-1) / denom, denom


def reuse_summary(rows):
    groups = defaultdict(dict)
    metadata = {}
    for r in rows:
        doc = r["doc"]
        metadata[doc] = r["task"], r["band"]
        for task, layers in (("all", "all"), (r["task"], r["layers"])):
            group = groups[r["sample"], task, layers]
            if doc not in group:
                group[doc] = np.zeros((FREQ_WINDOW+1, 4), dtype=int)
            group[doc][r["freq"]] += [r[k] for k in TALLY]
    summary = []
    for (sample, task, layers), by_doc in groups.items():
        docs = sorted(by_doc)
        tables = np.stack([by_doc[d] for d in docs])
        total = tables.sum(axis=0)
        point, n = effect(total)
        ci = np.nanpercentile(effect(tables[bootstrap_indices(docs, metadata)].sum(axis=1))[0], [2.5, 97.5])
        top, top_hit, bottom, bottom_hit = total.sum(axis=0)
        summary.append(dict(sample=sample, task=task, layers=layers, docs=len(docs),
                            top=top, bottom=bottom, comparable=n, raw_top=top_hit/top,
                            raw_bottom=bottom_hit/bottom, adjusted_gap=point, ci_low=ci[0], ci_high=ci[1]))
    return summary


def decisions_summary(rows):
    out = []
    for beta in (0, 2):
        by_doc = defaultdict(Counter)
        metadata = {}
        for r in rows:
            if r["beta"] == beta:
                by_doc[r["doc"]].update({k:r[k] for k in EVENTS})
                metadata[r["doc"]] = r["task"], r["band"]
        docs = sorted(by_doc)
        totals = sum(by_doc.values(), Counter())
        result = dict(beta=beta, **{k:totals[k] for k in EVENTS})
        boot = bootstrap_indices(docs, metadata)
        for name, num, den in (("change_rate", "changed", "evictions"),
                                ("rank_first_rate", "rank_first", "complete_changed"),
                                ("base_first_rate", "base_first", "complete_changed")):
            a, b = (np.array([by_doc[d][k] for d in docs]) for k in (num, den))
            result[name] = a.sum()/b.sum()
            lo, hi = np.percentile(a[boot].sum(axis=1)/b[boot].sum(axis=1), [2.5, 97.5])
            result[name+"_low"], result[name+"_high"] = lo, hi
        out.append(result)
    return out


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"selected":"Selected now", "rf_candidates":"Eligible for eviction (R+F)"}
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    for sample, color, dy in zip(SAMPLES, ("#256094", "#ad5521"), (-0.10, 0.10)):
        sub = [r for r in rows if r["sample"] == sample]
        y = np.arange(len(sub)) + dy
        x, lo, hi = [100*np.array([r[k] for r in sub]) for k in ("adjusted_gap","ci_low","ci_high")]
        ax.errorbar(x, y, xerr=[x-lo,hi-x], fmt="o", capsize=3, color=color, label=labels[sample])
    ax.set_yticks(range(len(sub)), ["All documents" if r["task"]=="all" else
                  f"{r['task']} · {r['layers']}" for r in sub])
    ax.invert_yaxis()
    ax.axvline(0, color="#777", lw=0.8)
    ax.set_xlabel("8-step reuse probability gap, top minus bottom rank (percentage points)")
    ax.set_title("Reuse at matched frequency · 20 development documents")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dev_zip", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    with args.dev_zip.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != DEV_SHA:
            parser.error("Expected the personal 20-document development archive")
    strata, events = collect(args.dev_zip)
    summary = reuse_summary(strata)
    args.out.mkdir(exist_ok=True, parents=True)
    for name, rows in (("rank_strata", strata), ("event_counts", events), ("reuse_summary", summary),
                       ("decision_summary", decisions_summary(events))):
        write_csv(args.out / (name + ".csv"), rows)
    plot(summary, args.out / "rank_reuse.png")


if __name__ == "__main__":
    main()
