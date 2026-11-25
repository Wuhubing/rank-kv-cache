"""Summaries of S7 runs (analysis_v2/s7_bench.py outputs).

  python -m analysis_v2.s7_summarize single results_v2/gpu/single_06b
  python -m analysis_v2.s7_summarize batch  results_v2/gpu/batch_06b
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def single(d):
    run = json.loads((d / "runs.json").read_text())
    g = defaultdict(list)
    for r in run["rows"]:
        g[r["B"], r["policy"]].append(r)
    out = []
    for (B, pol), rs in sorted(g.items(), key=lambda x: (x[0][0] or 0, x[0][1])):
        med = np.array([r["median_ms"] for r in rs])
        docs = sorted({r["doc"] for r in rs})
        # repeat-to-repeat spread: std of the per-run medians within a document, averaged over documents
        within = np.mean([np.std([r["median_ms"] for r in rs if r["doc"] == dd]) for dd in docs])
        row = dict(B=B, policy=pol, runs=len(rs), docs=len(docs), tpot_median_ms=float(np.mean(med)),
                   tpot_p90_ms=float(np.mean([r["p90_ms"] for r in rs])),
                   tpot_p99_ms=float(np.mean([r.get("p99_ms", np.nan) for r in rs])),
                   tpot_step_std_ms=float(np.mean([r["std_ms"] for r in rs])), tpot_repeat_std_ms=float(within),
                   misses_per_step=float(np.mean([r.get("misses_per_step_steady") or 0 for r in rs])),
                   pcie_MiB_per_token=float(np.mean([r.get("pcie_bytes_per_token") or 0 for r in rs])) / 2**20,
                   peak_prefill_GiB=float(np.mean([r.get("peak_prefill") or 0 for r in rs])) / 2**30,
                   peak_migration_GiB=float(np.mean([r.get("peak_migration") or 0 for r in rs])) / 2**30,
                   peak_decode_warmup_GiB=float(np.mean([r.get("peak_decode_warmup") or 0 for r in rs])) / 2**30,
                   peak_decode_GiB=float(np.mean([r.get("peak_decode") or 0 for r in rs])) / 2**30,
                   mem_start_GiB=float(np.mean([r.get("mem_start") or 0 for r in rs])) / 2**30,
                   tokens_equal_resident=all(r.get("tokens_equal_resident", True) is not False for r in rs),
                   ranks_equal_resident=all(r.get("ranks_equal_resident", True) is not False for r in rs))
        for k in ("bd_append_ms", "bd_select_ms", "bd_policy_ms", "bd_fetch_ms", "bd_attention_ms"):
            if k in rs[0]:
                row[k] = float(np.mean([r[k] for r in rs]))
        out.append(row)
    keys = sorted({k for r in out for k in r}, key=lambda k: list(out[0]).index(k) if k in out[0] else 99)
    write(d / "summary.csv", keys, [[r.get(k, "") for k in keys] for r in out])
    write(d / "checks.csv", list(run["checks"][0]), [list(c.values()) for c in run["checks"]])
    for r in out:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("checks:", sum(1 for c in run["checks"] if c["equal"] is True), "equal,",
          sum(1 for c in run["checks"] if c["equal"] is False), "unequal,",
          sum(1 for c in run["checks"] if c["equal"] is None), "not simulated (EMA needs scores)")


def batch(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = json.loads((d / "runs.json").read_text())
    g = defaultdict(list)
    skipped = []
    for r in run["rows"]:
        if "skipped" in r:
            skipped.append(r)
            continue
        g[r["R"], r["policy"]].append(r)
    out = []
    for (R, pol), rs in sorted(g.items()):
        tps = np.array([r["tokens_per_s"] for r in rs])
        out.append(dict(R=R, policy=pol, reps=len(rs), tokens_per_s=float(tps.mean()), tokens_per_s_std=float(tps.std()),
                        step_median_ms=float(np.mean([r["median_ms"] for r in rs])),
                        step_p90_ms=float(np.mean([r["p90_ms"] for r in rs])),
                        misses=int(rs[0]["misses"]), pcie_MiB_per_step=float(np.mean([r["pcie_bytes_per_step"] for r in rs])) / 2**20,
                        peak_decode_GiB=float(np.mean([r["peak_decode"] for r in rs])) / 2**30,
                        host_tier_GiB=rs[0]["host_tier_bytes"] / 2**30,
                        ranks_equal_lru=all(r["ranks_equal_lru"] for r in rs),
                        tokens_equal_lru=all(r["tokens_equal_lru"] for r in rs),
                        **{k: float(np.mean([r[k] for r in rs])) for k in rs[0] if k.startswith("bd_")}))
    lru = {r["R"]: r for r in out if r["policy"] == "lru"}
    for r in out:
        l = lru[r["R"]]
        r["throughput_vs_lru"] = r["tokens_per_s"] / l["tokens_per_s"]
        r["bytes_vs_lru"] = r["pcie_MiB_per_step"] / l["pcie_MiB_per_step"] if l["pcie_MiB_per_step"] else float("nan")
    keys = list(out[0])
    write(d / "summary.csv", keys, [[r[k] for k in keys] for r in out])
    if run["checks"]:
        ks = sorted({k for c in run["checks"] for k in c})
        write(d / "checks.csv", ks, [[c.get(k, "") for k in ks] for c in run["checks"]])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for pol in sorted({r["policy"] for r in out}):
        rs = sorted([r for r in out if r["policy"] == pol], key=lambda r: r["R"])
        ax[0].plot([r["R"] for r in rs], [r["tokens_per_s"] for r in rs], marker="o", label=pol)
        ax[1].plot([r["R"] for r in rs], [r["throughput_vs_lru"] for r in rs], marker="o", label=pol)
        ax[2].plot([r["R"] for r in rs], [r["pcie_MiB_per_step"] for r in rs], marker="o", label=pol)
    for a, t in zip(ax, ("tokens/s", "throughput / LRU", "PCIe MiB per step")):
        a.set_xscale("log", base=2)
        a.set_xlabel("batch size")
        a.set_ylabel(t)
    ax[1].axhline(1, color="k", lw=0.5)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "batch.png", dpi=130)
    for r in out:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("skipped:", skipped)
    print("checks:", run["checks"])


if __name__ == "__main__":
    {"single": single, "batch": batch}[sys.argv[1]](Path(sys.argv[2]))
