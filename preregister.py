"""Tune on the 20 dev documents and write the machine-readable preregistration.

Tuning follows the proposal: at the primary budget, pick delta, beta in GRID by dev miss
rate, breaking ties toward smaller values (smaller delta, then smaller beta). R+F tunes
beta with delta = 0 and R+rank tunes delta with beta = 0 the same way. The logistic
regression of gap_analysis.py is fit once on all dev candidates (LRU eviction decisions
at the primary budget, label: reselected within 16 steps) and its coefficients are
frozen. Run on dev only; the output is committed before any held-out data exists.
"""
import argparse
import hashlib
import json
from pathlib import Path

from replay import compare, read_trace

GRID = (0, 1, 2, 4, 8)
PRIMARY_BUDGET = 128
BUDGETS = (128, 256)
TOP_K = 64
LABEL_HORIZON, SAMPLE_ROWS, SEED = 16, 300000, 0
BOOTSTRAP, H1_BOOTSTRAP = 2000, 1000
# 16 tokens x 8 KV heads x 128 dims x (K, V) x 2 bytes (bf16) per page per layer.
PAGE_BYTES = 16 * 8 * 128 * 2 * 2


def pick(results, keys):
    """Lowest misses, then smaller delta, then smaller beta."""
    return min(keys, key=lambda k: (results[k], k[0], k[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dev_dir", type=Path, help="run directory with the 20 dev documents")
    parser.add_argument("--out", type=Path, default=Path("splits/preregistration.json"))
    parser.add_argument("--tuning-csv", type=Path, help="also write the grid results here")
    args = parser.parse_args()
    trace_path = args.dev_dir / "trace.jsonl"
    streams = read_trace(trace_path)
    split = json.loads(Path("splits/longbench_v1.json").read_text())
    dev = {p["request_id"] for p in split["prompts"] if p["split"] == "dev"}
    docs = {request for request, _ in streams}
    if docs != dev:
        raise SystemExit(f"expected exactly the {len(dev)} dev documents, got {len(docs)}")

    specs = [f"score:{d}:{b}" for d in GRID for b in GRID]
    rows = compare(streams, [PRIMARY_BUDGET], specs, top_k=TOP_K)
    misses = {(r["delta"], r["beta"]): r["misses"] for r in rows}
    full = pick(misses, list(misses))
    rf = pick(misses, [k for k in misses if k[0] == 0])
    rrank = pick(misses, [k for k in misses if k[1] == 0])
    print("Grid misses at B=128 (rows delta, columns beta):")
    for d in GRID:
        print(f"  delta={d}: " + "  ".join(f"{misses[d, b]}" for b in GRID))
    print(f"full {full}, R+F beta={rf[1]}, R+rank delta={rrank[0]}")
    if args.tuning_csv:
        args.tuning_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.tuning_csv.open("w", encoding="utf-8") as output:
            output.write("delta,beta,misses,repeat_misses,miss_rate\n")
            for r in rows:
                output.write(f"{r['delta']},{r['beta']},{r['misses']},{r['repeat_misses']},"
                             f"{r['miss_rate']}\n")

    import numpy as np
    from gap_analysis import FEATURES, Logistic, build_dataset
    prompts = {}
    for path in (args.dev_dir / "manifests").glob("*.json"):
        manifest = json.loads(path.read_text())
        prompts[manifest["request_id"]] = manifest["prompt_tokens"]
    X, distance, _, _, _ = build_dataset(streams, prompts, PRIMARY_BUDGET)
    sample = np.random.default_rng(SEED).choice(len(X), min(len(X), SAMPLE_ROWS), replace=False)
    model = Logistic(range(len(FEATURES))).fit(X[sample], (distance[sample] <= LABEL_HORIZON).astype(float))

    manifest = json.loads((args.dev_dir / "manifest.json").read_text())
    prereg = dict(
        description="Preregistered held-out evaluation; see docs/preregistration.md",
        split_file="splits/longbench_v1.json",
        split_sha256=hashlib.sha256(Path("splits/longbench_v1.json").read_bytes()).hexdigest(),
        dev=dict(trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                 documents=len(docs), gpu=manifest["gpu"], dtype=manifest["dtype"],
                 torch=manifest["torch"], transformers=manifest["transformers"]),
        collection=dict(max_new_tokens=128, stop="EOS or 128 generated tokens", page_size=16,
                        top_k=TOP_K, gpu_required=manifest["gpu"],
                        note="Quest selections differ across GPU models, so held-out must be "
                             "collected on the same GPU model as dev."),
        tuning=dict(grid=list(GRID), budget=PRIMARY_BUDGET, criterion="dev miss rate",
                    tie_break="smaller delta, then smaller beta",
                    misses={f"{d},{b}": misses[d, b] for d in GRID for b in GRID}),
        budgets=list(BUDGETS), primary_budget=PRIMARY_BUDGET, top_k=TOP_K, page_bytes=PAGE_BYTES,
        policies=[
            dict(name="fifo", policy="fifo"),
            dict(name="lru", policy="lru"),
            dict(name="rank_only", policy="rank"),
            dict(name="rf", policy="score", delta=0, beta=rf[1]),
            dict(name="rrank", policy="score", delta=rrank[0], beta=0),
            dict(name="full", policy="score", delta=full[0], beta=full[1]),
            dict(name="logistic", policy="logistic", label_horizon=LABEL_HORIZON,
                 trained_on="20 dev documents, LRU eviction candidates at B=128",
                 sample_rows=SAMPLE_ROWS, seed=SEED, model=model.to_dict()),
            dict(name="belady", policy="belady"),
        ],
        metrics=["repeat_misses", "misses", "G", "h2d_bytes_per_token"],
        success=dict(
            s1="full has at least 10% fewer repeat misses than LRU at B=128 (pooled point "
               "estimate); a document-bootstrap 95% CI is reported alongside",
            s2="paired document-bootstrap 95% CI of repeat misses saved by full relative to "
               "rf at B=128 lies entirely above zero"),
        bootstrap=dict(replicates=BOOTSTRAP, seed=SEED, unit="document (request)"),
        h1=dict(samples=["selected", "candidate", "lru"], lru_budget=PRIMARY_BUDGET,
                horizons=[1, 2, 4, 8], by=["task", "band"], cluster="request",
                replicates=H1_BOOTSTRAP, seed=SEED),
        groups=["task", "band"],
        mfqa=dict(primary="as collected: 128 generated tokens or EOS",
                  secondary="truncate_trace.py: MultiFieldQA cut at the first generated "
                            "newline, same policies and fixed coefficients"),
        sanity=dict(metric={"multifieldqa_en": "qa_f1_score", "gov_report": "rouge_score (ROUGE-L F)"},
                    compare="sparse reference generation vs dense-attention greedy generation, "
                            "128 tokens, LongBench official metrics"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(prereg, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
