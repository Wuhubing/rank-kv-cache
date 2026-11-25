"""Replay the complete frozen development split at B=96 and compare policies by document."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from longctx_document import POLICIES, ROOT, replay, write_csv

COMPARISONS = (("lru", "score"), ("score", "ema:0.5"), ("lru", "ema:0.5"))


def summarize(docs):
    # Draw whole documents; every policy sees the same bootstrap samples.
    idx = np.random.default_rng(0).integers(len(docs), size=(10000, len(docs)))
    rows = []
    for base, policy in COMPARISONS:
        a, b = (np.array([d[k] for d in docs]) for k in (base, policy))
        saved = a - b
        ci = np.percentile(saved[idx].sum(axis=1) / a[idx].sum(axis=1), [2.5, 97.5])
        rows.append(dict(baseline=base, policy=policy, repeat_misses_saved=int(saved.sum()),
                         relative_drop=saved.sum()/a.sum(), ci_low=ci[0], ci_high=ci[1],
                         wins=int((saved>0).sum()), ties=int((saved==0).sum()),
                         losses=int((saved<0).sum()), smallest_drop=min(saved/a),
                         largest_drop=max(saved/a)))
    return rows


def plot_savings(docs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    labels = [r["task"] + "/" + r["doc"].split("/")[1][:6] for r in docs]
    score = np.array([(r["lru"]-r["score"])/r["lru"] for r in docs])
    extra = np.array([(r["score"]-r["ema:0.5"])/r["lru"] for r in docs])
    fig, ax = plt.subplots(figsize=(8.1, 5.3), layout="constrained")
    ax.barh(labels, score, color="#3b7392", label="Current score")
    ax.barh(labels, extra, left=score, color="#d6863c", label="Additional savings with EMA")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Repeat misses saved as a share of LRU repeat misses")
    ax.set_title("Savings by development document · B=96, K=64")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.07), ncols=2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", type=Path, help="Directory containing the 14 named dev NPZ files")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    split = json.loads((ROOT/"splits/longctx_v1.json").read_text())
    docs = [d for d in split["prompts"] if d["split"] == "dev"]
    paths = [args.traces / (d["rid"].replace("/", "__") + ".npz") for d in docs]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        ap.error(f"Missing {len(missing)} dev traces: {', '.join(missing)}")
    per_doc, by_layer = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for d, (meta, rows) in zip(docs, pool.map(replay, paths), strict=True):
            if meta["doc"] != d["rid"]:
                raise ValueError(f"Wrong trace content for {d['rid']}")
            metrics = {p:sum(r["repeat_misses"] for r in rows if r["policy"] == p) for p in POLICIES}
            common = {k:sum(r[k] for r in rows if r["policy"] == "lru") for k in ("accesses","compulsory")}
            per_doc.append(dict(**meta, task=d["task"], band=d["band"], **common, **metrics))
            by_layer.extend(dict(doc=d["rid"], **r) for r in rows)
            print(f"{len(per_doc)}/{len(docs)} matched: {d['rid']}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/"by_document.csv", per_doc)
    write_csv(args.out/"by_layer.csv", by_layer)
    write_csv(args.out/"paired.csv", summarize(per_doc))
    plot_savings(per_doc, args.out/"savings.png")


if __name__ == "__main__":
    main()
