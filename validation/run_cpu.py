"""Recompute committed rank results and measure explicit synthetic score regimes.

python -m validation.run_cpu --out validation/results/cpu_experiments.json
Synthetic results are mechanism checks, not a substitute for LongBench traces.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import evaluate
import kvpolicy as kp

ROOT = Path(__file__).resolve().parents[1]


def audit_rank_tables():
    reports = []
    prereg = json.loads((ROOT / "splits/preregistration.json").read_text())
    for name in ("eval", "eval_truncated"):
        folder = ROOT / "docs/results/heldout" / name
        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
        with (folder / "policies_by_doc.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                counts[int(row["budget"])][row["policy"]][row["doc"]].update(
                    {k: int(v) for k, v in row.items() if k not in ("budget", "policy", "doc")})
        docs = sorted(counts[128]["lru"])
        table = evaluate.table(counts, prereg, docs, prereg["bootstrap"]["seed"])
        with (folder / "policies.csv").open(newline="") as f:
            reported = {(int(r["budget"]), r["policy"]): r for r in csv.DictReader(f)}
        errors = []
        for row in table:
            original = reported[row["budget"], row["policy"]]
            for key, val in row.items():
                if key in ("group", "policy"):
                    assert val == original[key]
                elif val is not None:
                    errors.append(abs(float(original[key]) - val))
        success = evaluate.success(counts, prereg, docs)
        original_success = json.loads((folder / "success.json").read_text())
        assert success == original_success
        assert max(errors) < 1e-12
        reports.append({"table": name, "documents": len(docs), "max_absolute_error": max(errors),
                        "success_matches": True, "success": success,
                        "full_policy_B128": next(r for r in table if r["budget"] == 128 and r["policy"] == "full")})
    return reports


def synthetic_regimes():
    out = []
    steps, pages, width, budget = 400, 32, 8, 12
    for rho in (0.98, 0.0, -0.9):
        per = {p: [] for p in ("lru", "score", "ema", "belady")}
        quant_deltas = []
        for seed in range(12):
            rng = np.random.default_rng(seed)
            scores = np.empty((steps, pages), dtype=np.float32)
            scores[0] = rng.normal(size=pages)
            for t in range(1, steps):
                scores[t] = rho * scores[t - 1] + np.sqrt(1 - rho*rho) * rng.normal(size=pages)
            ranked = np.argsort(-scores, axis=1, kind="stable")[:, :width]
            for name in per:
                per[name].append(kp.simulate(ranked, scores, budget, name)["misses"])
            quant = kp.simulate(ranked, scores.astype(np.float16).astype(np.float32), budget, "ema")["misses"]
            quant_deltas.append(quant - per["ema"][-1])
        totals = {p: sum(v) for p, v in per.items()}
        out.append({"rho": rho, "seeds": 12, "steps_per_seed": steps, "pages": pages,
                    "K": width, "B": budget, "misses": totals,
                    "G_ema": (totals["lru"] - totals["ema"]) / (totals["lru"] - totals["belady"]),
                    "ema_documents_better": sum(a < b for a, b in zip(per["ema"], per["lru"])),
                    "fp16_minus_fp32_misses_by_seed": quant_deltas})
    return out


def counterexample():
    """Deterministic search for a valid trace on which EMA loses to LRU."""
    rng = np.random.default_rng(17)
    for trial in range(1000):
        scores = np.array([rng.permutation(4) for _ in range(6)], dtype=np.float32)
        ranked = np.argsort(-scores, axis=1, kind="stable")[:, :2]
        results = {p: kp.simulate(ranked, scores, 3, p)["misses"] for p in ("lru", "ema", "belady")}
        if results["ema"] > results["lru"]:
            return dict(seed=17, search_trial=trial, B=3, alpha=0.5,
                        scores=scores.tolist(), ranked=ranked.tolist(), misses=results)
    raise AssertionError("No counterexample in fixed search")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_files = ["kvpolicy.py", "heldout.py", "gpu_bench.py", "efficient_dev.py", "checks.py"]
    result = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "origin/efficient-policy"], cwd=ROOT, text=True).strip(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in source_files},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": importlib.metadata.version("numpy")},
        "rank_artifact_recalculation": audit_rank_tables(),
        "synthetic_regimes": synthetic_regimes(),
        "ema_counterexample": counterexample(),
        "external_long_context_traces": "Not supplied; excluded at user request. Reported long-context results are not independently reproduced.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"rank_tables_match": True, "synthetic_regimes": result["synthetic_regimes"],
                      "counterexample": result["ema_counterexample"]}, indent=2))


if __name__ == "__main__":
    main()
