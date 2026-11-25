"""What makes up the LRU-Belady gap?

1. Decision annotation. Replay LRU at one budget. At every eviction, record each
   candidate's true next-use distance (infinite when it is never selected again in the
   trace) and flag bad evictions: a victim whose next use is sooner than that of some
   retained candidate. Repeat misses avoided by Belady are broken down by layer, task,
   band and document.
2. Feature profile. Every candidate at an LRU eviction gets trace-derived features
   (FEATURES below). Reuse within h steps is profiled per feature, with univariate AUCs.
3. Achievable bound. Leave one document out: train logistic regression, a depth <= 3
   decision tree, and single-feature rules on seven documents, then replay the held-out
   document evicting the candidate with the lowest predicted reuse (ties follow LRU).
   A heuristic predicts the next use as tau + the last selection gap. G is the share of
   the LRU-Belady repeat-miss gap a policy closes, with a bootstrap over documents.

Replay, features and annotation use the standard library; models and statistics use
numpy. Only dev traces belong here.
"""
import argparse
import csv
import json
import math
import random
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from replay import read_trace, replay_stream, request_group

INF = math.inf
FEATURES = ("recency", "freq8", "freq32", "total", "last_rank", "last_rank_frac",
            "last_gap", "has_gap", "mean_gap", "drop_interval", "has_drop",
            "rel_pos", "sink", "from_end", "near_end", "layer")
RANK_FEATURES = ("last_rank", "last_rank_frac")
SINK_PAGES, NEAR_END_PAGES = 4, 8
HORIZONS = (1, 2, 4, 8, 16)
DISTANCE_BINS = ((1, "1"), (2, "2"), (4, "3-4"), (8, "5-8"), (16, "9-16"), (32, "17-32"),
                 (64, "33-64"), (128, "65-127"))


class History:
    """Selection history of one request/layer stream, built step by step.

    At a decision at step t, observe_dropouts(t, ...) has run but record(t, ...) has not,
    so selection counts only cover steps before t. Candidates are unselected at t anyway.
    """

    def __init__(self, layer, prompt_tokens, page_size=16):
        self.layer = layer
        self.prompt_tokens = prompt_tokens
        self.page_size = page_size
        self.steps = defaultdict(list)
        self.rank = {}
        self.width = {}
        self.drops = defaultdict(list)
        self.previous = set()

    def observe_dropouts(self, t, selected):
        for page in self.previous - selected:
            self.drops[page].append(t)

    def record(self, t, pages):
        for rank, page in enumerate(pages, 1):
            self.steps[page].append(t)
            self.rank[page] = rank
            self.width[page] = len(pages)
        self.previous = set(pages)

    def features(self, page, t):
        sel = self.steps[page]
        tau = sel[-1]
        drops = self.drops[page]
        n_pages = max(1, (self.prompt_tokens + t) // self.page_size)
        from_end = n_pages - 1 - page
        gap = sel[-1] - sel[-2] if len(sel) > 1 else 0
        return [
            t - tau,
            len(sel) - bisect_left(sel, t - 8),
            len(sel) - bisect_left(sel, t - 32),
            len(sel),
            self.rank[page],
            self.rank[page] / self.width[page],
            gap,
            float(len(sel) > 1),
            (sel[-1] - sel[0]) / (len(sel) - 1) if len(sel) > 1 else 0,
            drops[-1] - drops[-2] if len(drops) > 1 else 0,
            float(len(drops) > 1),
            page / n_pages,
            float(page < SINK_PAGES),
            from_end,
            float(from_end < NEAR_END_PAGES),
            self.layer,
        ]


def next_uses(steps):
    uses = defaultdict(list)
    for t, pages in enumerate(steps):
        for page in pages:
            uses[page].append(t)
    return uses


def next_use(uses, page, t):
    later = uses[page]
    i = bisect_right(later, t)
    return later[i] if i < len(later) else INF


def replay_scored(steps, budget, history, score=None, on_decision=None):
    """Replay one stream. Evict the lowest score first, ties in LRU order.

    score(candidates, t, history) returns one keep-score per candidate; None is LRU.
    on_decision(t, candidates, victims) sees candidates in eviction order.
    Returns (misses, compulsory misses, accesses).
    """
    resident = {}
    seen = set()
    misses = accesses = 0
    for t, pages in enumerate(steps):
        selected = set(pages)
        if len(selected) > budget:
            raise ValueError(f"budget {budget} is below the selection at step {t}")
        history.observe_dropouts(t, selected)
        hits = selected.intersection(resident)
        missing = selected - hits
        evict = len(resident) + len(missing) - budget
        if evict > 0:
            candidates = sorted(resident.keys() - selected, key=resident.__getitem__)
            if score is not None:
                values = dict(zip(candidates, score(candidates, t, history)))
                candidates.sort(key=lambda p: (values[p], resident[p]))
            victims = candidates[:evict]
            if on_decision:
                on_decision(t, candidates, victims)
            for page in victims:
                del resident[page]
        for page in pages:
            resident[page] = (t, page in hits, -page)
        history.record(t, pages)
        accesses += len(pages)
        misses += len(missing)
        seen.update(selected)
    return misses, len(seen), accesses


def distance_bin(distance):
    if distance == INF:
        return "never"
    for limit, label in DISTANCE_BINS:
        if distance <= limit:
            return label
    return ">127"


def annotate(streams, prompts, budget):
    """LRU decisions: bad-eviction counts and victim next-use distances."""
    summary = Counter()
    victim_bins = Counter()
    bad_bins = Counter()
    by_doc = defaultdict(Counter)
    for (request, layer), steps in sorted(streams.items()):
        uses = next_uses(steps)

        def on_decision(t, candidates, victims, request=request):
            chosen = set(victims)
            retained = [next_use(uses, p, t) for p in candidates if p not in chosen]
            worst_kept = max(retained) if retained else -INF
            summary["decisions"] += 1
            summary["candidates"] += len(candidates)
            any_bad = False
            for page in victims:
                distance = next_use(uses, page, t) - t
                bad = distance + t < worst_kept
                victim_bins[distance_bin(distance)] += 1
                summary["victims"] += 1
                by_doc[request]["victims"] += 1
                if bad:
                    any_bad = True
                    bad_bins[distance_bin(distance)] += 1
                    summary["bad_victims"] += 1
                    by_doc[request]["bad_victims"] += 1
            summary["decisions_with_bad"] += any_bad

        replay_scored(steps, budget, History(layer, prompts[request]), on_decision=on_decision)
    return summary, victim_bins, bad_bins, by_doc


def gap_by_group(streams, budget):
    """Repeat misses of LRU and Belady per layer, task, band, cell and document."""
    totals = defaultdict(Counter)
    for (request, layer), steps in streams.items():
        lru = replay_stream(steps, budget, "lru")
        belady = replay_stream(steps, budget, "belady")
        labels = ["all", f"layer {layer:02d}", f"doc {request}"]
        labels += [f"{kind} {request_group(request, kind)}" for kind in ("task", "band", "cell")]
        for label in labels:
            totals[label]["lru_repeat"] += lru["repeat_misses"]
            totals[label]["belady_repeat"] += belady["repeat_misses"]
            totals[label]["repeat_accesses"] += lru["repeat_accesses"]
    return totals


def build_dataset(streams, prompts, budget, with_time=False):
    """Features, next-use distance and document index for every LRU candidate.

    with_time=True also returns, per row, the decision step t and the stream length,
    so callers can find right-censored labels (see censored_labels).
    """
    import numpy as np
    docs = sorted({request for request, _ in streams})
    blocks, distances, doc_ids, victim_flags, times, lengths = [], [], [], [], [], []
    for (request, layer), steps in sorted(streams.items()):
        uses = next_uses(steps)
        history = History(layer, prompts[request])
        rows, dist, victim, when = [], [], [], []

        def on_decision(t, candidates, victims):
            chosen = set(victims)
            for page in candidates:
                rows.append(history.features(page, t))
                dist.append(next_use(uses, page, t) - t)
                victim.append(page in chosen)
                when.append(t)

        replay_scored(steps, budget, history, on_decision=on_decision)
        if rows:
            blocks.append(np.asarray(rows, dtype=np.float64))
            distances.append(np.asarray(dist, dtype=np.float64))
            victim_flags.append(np.asarray(victim, dtype=bool))
            doc_ids.append(np.full(len(rows), docs.index(request)))
            times.append(np.asarray(when))
            lengths.append(np.full(len(rows), len(steps)))
    result = (np.concatenate(blocks), np.concatenate(distances), np.concatenate(doc_ids),
              np.concatenate(victim_flags), docs)
    if with_time:
        result += (np.concatenate(times), np.concatenate(lengths))
    return result


def censored_labels(distance, t, length, horizon):
    """Rows labelled negative only because the trace ends before step t + horizon.

    The label is distance <= horizon. A row with no reuse up to the last step
    (length - 1) but t + horizon >= length was never observed for the full window;
    its true label is unknown, yet training counts it as negative.
    """
    return (distance > horizon) & (t + horizon >= length)


def auc(score, label):
    """Probability that a random positive outscores a random negative (ties count half)."""
    import numpy as np
    label = label.astype(bool)
    n_pos, n_neg = label.sum(), (~label).sum()
    if not n_pos or not n_neg:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    _, first, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    average = first + (counts + 1) / 2
    ranks[order] = np.repeat(average, counts)
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def profile(X, distance, docs_idx):
    """Binned reuse rates, AUCs and group medians per feature and horizon."""
    import numpy as np
    curve, aucs, medians = [], [], []
    for j, name in enumerate(FEATURES):
        values = X[:, j]
        distinct = np.unique(values)
        if len(distinct) <= 12:  # binary or small integer features: one bin per value
            bins = np.searchsorted(distinct, values)
            bounds = [(v, v) for v in distinct]
        else:
            edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
            bins = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
            bounds = list(zip(edges[:-1], edges[1:]))
        for h in HORIZONS:
            label = distance <= h
            for b in np.unique(bins):
                mask = bins == b
                curve.append(dict(feature=name, horizon=h, bin=int(b), low=float(bounds[b][0]),
                                  high=float(bounds[b][1]), candidates=int(mask.sum()),
                                  reuse_rate=float(label[mask].mean())))
            per_doc = [auc(values[docs_idx == d], label[docs_idx == d])
                       for d in np.unique(docs_idx)]
            per_doc = [a for a in per_doc if a is not None]
            aucs.append(dict(feature=name, horizon=h, auc=auc(values, label),
                             doc_auc_min=min(per_doc) if per_doc else None,
                             doc_auc_max=max(per_doc) if per_doc else None))
        label = distance <= 8
        medians.append(dict(feature=name, median_reused_within_8=float(np.median(values[label])),
                            median_not_reused_within_8=float(np.median(values[~label])),
                            mean_reused_within_8=float(values[label].mean()),
                            mean_not_reused_within_8=float(values[~label].mean())))
    return curve, aucs, medians


class Logistic:
    def __init__(self, columns, l2=1e-3, iterations=25):
        self.columns, self.l2, self.iterations = list(columns), l2, iterations

    def fit(self, X, y):
        import numpy as np
        Z = X[:, self.columns]
        self.mean, self.std = Z.mean(0), Z.std(0) + 1e-9
        A = np.hstack([np.ones((len(Z), 1)), (Z - self.mean) / self.std])
        w = np.zeros(A.shape[1])
        for _ in range(self.iterations):
            p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
            gradient = A.T @ (p - y) + self.l2 * np.r_[0, w[1:]]
            hessian = (A * (p * (1 - p))[:, None]).T @ A + self.l2 * np.eye(len(w))
            w -= np.linalg.solve(hessian, gradient)
        self.w = w
        return self

    def predict(self, X):
        import numpy as np
        Z = (X[:, self.columns] - self.mean) / self.std
        return 1 / (1 + np.exp(-np.clip(self.w[0] + Z @ self.w[1:], -30, 30)))

    def to_dict(self):
        return dict(features=[FEATURES[j] for j in self.columns], mean=self.mean.tolist(),
                    std=self.std.tolist(), intercept=float(self.w[0]),
                    weights=self.w[1:].tolist(), l2=self.l2)

    @classmethod
    def from_dict(cls, params):
        import numpy as np
        model = cls([FEATURES.index(f) for f in params["features"]], l2=params.get("l2", 1e-3))
        model.mean, model.std = np.asarray(params["mean"]), np.asarray(params["std"])
        model.w = np.r_[params["intercept"], params["weights"]]
        return model


class Tree:
    """CART on Gini impurity with quantile thresholds, depth <= max_depth."""

    def __init__(self, max_depth=3, min_leaf=200, thresholds=32):
        self.max_depth, self.min_leaf, self.thresholds = max_depth, min_leaf, thresholds

    def fit(self, X, y):
        self.root = self._grow(X, y.astype(float), 0)
        return self

    def _grow(self, X, y, depth):
        import numpy as np
        node = dict(rate=float(y.mean()), n=len(y))
        if depth == self.max_depth or len(y) < 2 * self.min_leaf or node["rate"] in (0, 1):
            return node
        best = None
        parent = len(y) * node["rate"] * (1 - node["rate"])
        for j in range(X.shape[1]):
            candidates = np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.thresholds + 1)[1:-1]))
            for threshold in candidates:
                left = X[:, j] <= threshold
                n_left = left.sum()
                if n_left < self.min_leaf or len(y) - n_left < self.min_leaf:
                    continue
                p_left, p_right = y[left].mean(), y[~left].mean()
                impurity = (n_left * p_left * (1 - p_left)
                            + (len(y) - n_left) * p_right * (1 - p_right))
                if best is None or impurity < best[0]:
                    best = (impurity, j, float(threshold), left)
        if best is None or best[0] >= parent:
            return node
        _, j, threshold, left = best
        node.update(feature=j, threshold=threshold,
                    left=self._grow(X[left], y[left], depth + 1),
                    right=self._grow(X[~left], y[~left], depth + 1))
        return node

    def predict(self, X):
        import numpy as np
        out = np.empty(len(X))
        for i, row in enumerate(X):
            node = self.root
            while "feature" in node:
                node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
            out[i] = node["rate"]
        return out

    def rules(self, node=None, indent=""):
        node = node or self.root
        if "feature" not in node:
            return [f"{indent}reuse rate {node['rate']:.3f} (n={node['n']})"]
        name = FEATURES[node["feature"]]
        return ([f"{indent}if {name} <= {node['threshold']:g}:"]
                + self.rules(node["left"], indent + "    ")
                + [f"{indent}else:  # {name} > {node['threshold']:g}"]
                + self.rules(node["right"], indent + "    "))


def model_score(model):
    import numpy as np

    def score(candidates, t, history):
        X = np.asarray([history.features(p, t) for p in candidates], dtype=np.float64)
        return model.predict(X).tolist()
    return score


def feature_score(j, sign):
    def score(candidates, t, history):
        return [sign * history.features(p, t)[j] for p in candidates]
    return score


def gap_heuristic(candidates, t, history):
    """Keep the soonest predicted next use, tau + last gap; no gap means never."""
    out = []
    for page in candidates:
        sel = history.steps[page]
        out.append(-(sel[-1] + sel[-1] - sel[-2]) if len(sel) > 1 else -INF)
    return out


def bootstrap_g(per_doc, policy, replicates, seed):
    """Pooled G and repeat-miss reduction with percentile intervals over documents."""
    docs = sorted(per_doc)
    rng = random.Random(seed)

    def stats(sample):
        lru = sum(per_doc[d]["lru"] for d in sample)
        belady = sum(per_doc[d]["belady"] for d in sample)
        mine = sum(per_doc[d][policy] for d in sample)
        g = (lru - mine) / (lru - belady) if lru > belady else None
        return g, (lru - mine) / lru if lru else None

    g, reduction = stats(docs)
    gs, rs = [], []
    for _ in range(replicates):
        sample = [rng.choice(docs) for _ in docs]
        a, b = stats(sample)
        if a is not None:
            gs.append(a)
            rs.append(b)
    pick = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))] if v else None
    return dict(G=g, G_low=pick(gs, .025), G_high=pick(gs, .975), reduction=reduction,
                reduction_low=pick(rs, .025), reduction_high=pick(rs, .975))


def evaluate(streams, prompts, X, distance, doc_idx, docs, budget, horizon, sample_rows, seed):
    """Leave-one-document-out replay of learned and single-feature policies."""
    import numpy as np
    rng = np.random.default_rng(seed)
    label = (distance <= horizon).astype(float)
    no_rank = [j for j, f in enumerate(FEATURES) if f not in RANK_FEATURES]
    single = [j for j, f in enumerate(FEATURES) if f != "layer"]
    per_doc = defaultdict(Counter)
    signs = defaultdict(list)
    for d, doc in enumerate(docs):
        train = np.flatnonzero(doc_idx != d)
        if len(train) > sample_rows:
            train = rng.choice(train, sample_rows, replace=False)
        Xt, yt = X[train], label[train]
        policies = {
            "logistic": model_score(Logistic(range(len(FEATURES))).fit(Xt, yt)),
            "logistic_no_rank": model_score(Logistic(no_rank).fit(Xt, yt)),
            "tree": model_score(Tree().fit(Xt, yt)),
            "gap_heuristic": gap_heuristic,
        }
        for j in single:
            a = auc(Xt[:, j], yt)
            sign = 1 if a is None or a >= 0.5 else -1
            signs[FEATURES[j]].append(sign)
            policies[f"feature {FEATURES[j]}"] = feature_score(j, sign)
        for (request, layer), steps in streams.items():
            if request != doc:
                continue
            lru = replay_stream(steps, budget, "lru")
            belady = replay_stream(steps, budget, "belady")
            compulsory = lru["compulsory_misses"]
            per_doc[doc]["lru"] += lru["repeat_misses"]
            per_doc[doc]["belady"] += belady["repeat_misses"]
            for name, score in policies.items():
                misses, first, _ = replay_scored(steps, budget, History(layer, prompts[request]),
                                                 score=score)
                assert first == compulsory
                per_doc[doc][name] += misses - first
        print(f"  fold {d + 1}/{len(docs)} done: {doc}", flush=True)
    return per_doc, signs


def write_csv(path, rows):
    rows = list(rows)
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def save_profile_figure(curve, out_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed, so profile.png was skipped.")
        return
    names = [f for f in FEATURES if f != "layer"] + ["layer"]
    cols = 4
    fig, axes = plt.subplots(-(-len(names) // cols), cols, figsize=(16, 3.2 * -(-len(names) // cols)),
                             layout="constrained")
    for axis, name in zip(axes.flat, names):
        for h in HORIZONS:
            rows = [r for r in curve if r["feature"] == name and r["horizon"] == h]
            axis.plot([(r["low"] + r["high"]) / 2 for r in rows],
                      [100 * r["reuse_rate"] for r in rows], "o-", ms=3, label=f"h={h}")
        axis.set(title=name, ylabel="reused (%)")
        axis.set_ylim(0, 100)
    for axis in list(axes.flat)[len(names):]:
        axis.axis("off")
    axes.flat[0].legend(fontsize="small")
    fig.suptitle("LRU eviction candidates: reuse within h steps by feature (decile bins)")
    fig.savefig(Path(out_dir) / "profile.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="directory with a trace and manifests/")
    parser.add_argument("--trace", default="trace.jsonl", help="trace file inside run_dir")
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--label-horizon", type=int, default=16,
                        help="models predict reuse within this many steps")
    parser.add_argument("--sample-rows", type=int, default=300000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip-policies", action="store_true")
    args = parser.parse_args()
    out = args.out or args.run_dir / "gap"
    out.mkdir(parents=True, exist_ok=True)
    streams = read_trace(args.run_dir / args.trace)
    prompts = {}
    for path in (args.run_dir / "manifests").glob("*.json"):
        manifest = json.loads(path.read_text())
        prompts[manifest["request_id"]] = manifest["prompt_tokens"]
    missing = {r for r, _ in streams} - set(prompts)
    if missing:
        parser.error(f"no manifest for {sorted(missing)}")

    print("1. Annotating LRU decisions")
    summary, victim_bins, bad_bins, by_doc = annotate(streams, prompts, args.budget)
    order = [label for _, label in DISTANCE_BINS] + [">127", "never"]
    write_csv(out / "victim_next_use.csv", [
        dict(next_use=b, victims=victim_bins[b], bad_victims=bad_bins[b],
             victim_share=victim_bins[b] / summary["victims"],
             bad_share=bad_bins[b] / max(1, summary["bad_victims"])) for b in order])
    rng = random.Random(args.seed)
    docs = sorted(by_doc)
    shares = []
    for _ in range(args.bootstrap):
        sample = [rng.choice(docs) for _ in docs]
        shares.append(sum(by_doc[d]["bad_victims"] for d in sample)
                      / sum(by_doc[d]["victims"] for d in sample))
    shares.sort()
    decision = dict(budget=args.budget, **summary,
                    bad_share=summary["bad_victims"] / summary["victims"],
                    bad_share_low=shares[int(.025 * len(shares))],
                    bad_share_high=shares[int(.975 * len(shares)) - 1],
                    bad_share_by_doc={d: by_doc[d]["bad_victims"] / by_doc[d]["victims"] for d in docs})
    (out / "decisions.json").write_text(json.dumps(decision, indent=2) + "\n")
    groups = gap_by_group(streams, args.budget)
    total_gap = groups["all"]["lru_repeat"] - groups["all"]["belady_repeat"]
    write_csv(out / "gap_by_group.csv", [
        dict(group=g, lru_repeat=c["lru_repeat"], belady_repeat=c["belady_repeat"],
             avoided=c["lru_repeat"] - c["belady_repeat"],
             share_of_gap=(c["lru_repeat"] - c["belady_repeat"]) / total_gap,
             avoided_per_1k_repeat_accesses=1000 * (c["lru_repeat"] - c["belady_repeat"])
             / c["repeat_accesses"]) for g, c in sorted(groups.items())])

    print("2. Building candidate features")
    X, distance, doc_idx, victim, docs = build_dataset(streams, prompts, args.budget)
    curve, aucs, medians = profile(X, distance, doc_idx)
    write_csv(out / "feature_profile.csv", curve)
    write_csv(out / "feature_auc.csv", aucs)
    write_csv(out / "feature_medians.csv", medians)
    save_profile_figure(curve, out)
    import numpy as np
    label = (distance <= args.label_horizon).astype(float)
    sample = np.random.default_rng(args.seed).choice(len(X), min(len(X), args.sample_rows),
                                                     replace=False)
    tree = Tree().fit(X[sample], label[sample])
    (out / "tree_rules.txt").write_text(
        f"Depth-3 tree on all dev candidates (descriptive), label: reused within "
        f"{args.label_horizon} steps\n" + "\n".join(tree.rules()) + "\n")
    print(f"   {len(X)} candidates; tree rules in {out / 'tree_rules.txt'}")
    if args.skip_policies:
        return

    print("3. Leave-one-document-out policies")
    per_doc, signs = evaluate(streams, prompts, X, distance, doc_idx, docs, args.budget,
                              args.label_horizon, args.sample_rows, args.seed)
    write_csv(out / "policies_by_doc.csv", [dict(doc=d, **c) for d, c in sorted(per_doc.items())])
    names = [k for k in next(iter(per_doc.values())) if k not in ("lru", "belady")]
    rows = []
    for name in names:
        stat = bootstrap_g(per_doc, name, args.bootstrap, args.seed)
        rows.append(dict(policy=name, repeat_misses=sum(c[name] for c in per_doc.values()),
                         lru_repeat=sum(c["lru"] for c in per_doc.values()),
                         belady_repeat=sum(c["belady"] for c in per_doc.values()),
                         sign=(signs[name.split(" ", 1)[1]] if name.startswith("feature ") else ""),
                         **stat))
    rows.sort(key=lambda r: -(r["G"] or -9))
    write_csv(out / "policies.csv", rows)
    for r in rows:
        print(f"   {r['policy']:<26} G {100 * r['G']:+6.2f}% [{100 * r['G_low']:+.2f}, "
              f"{100 * r['G_high']:+.2f}]  repeat {r['repeat_misses']}  vs LRU "
              f"{100 * r['reduction']:+.2f}%")


if __name__ == "__main__":
    main()
