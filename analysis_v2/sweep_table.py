"""Aggregate explore.py summaries of the page-size/K sweep and the sampling seeds (S6, dev, exploratory).

  python -m analysis_v2.sweep_table sweep results_v2/ext/sweep --base results_v2/ext/06b_dev_trueattn
  python -m analysis_v2.sweep_table sampling results_v2/ext/sampling
"""
import csv
import sys
from pathlib import Path

POLS = ("lru", "belady", "score", "ema(lodo)", "ema_allrank(lodo)", "infllm_eq1:0.1")


def load(p):
    return {(r["policy"], round(float(r["B_over_K"]), 3)): r for r in csv.DictReader(open(p))}


def sweep(root, base):
    rows = []
    dirs = sorted(Path(root).glob("ps*_k*")) + ([Path(base)] if base else [])
    for d in dirs:
        if d.name.startswith("ps"):
            ps, k = d.name[2:].split("_k")
        else:
            ps, k = "16", "64"
        S = load(d / "summary.csv")
        for ratio in (1.25, 1.5, 2.0, 4.0):
            if ("lru", ratio) not in S:
                continue
            lru, opt = S["lru", ratio], S["belady", ratio]
            steps = None
            row = dict(page_size=int(ps), K=int(k), B_over_K=ratio, B=int(lru["B"]), tokens_selected=int(ps) * int(k),
                       lru_miss_pct=100 * float(lru["miss_rate"]), belady_miss_pct=100 * float(opt["miss_rate"]))
            for pol in ("score", "ema(lodo)", "ema_allrank(lodo)", "infllm_eq1:0.1"):
                r = S.get((pol, ratio))
                row[f"G_{pol}"] = float(r["G"]) if r else ""
                row[f"G_{pol}_lo"] = float(r["G_lo"]) if r else ""
                row[f"G_{pol}_hi"] = float(r["G_hi"]) if r else ""
            row["ema_wins"] = S[("ema(lodo)", ratio)]["docs_better_than_lru"]
            rows.append(row)
    rows.sort(key=lambda r: (r["page_size"], r["K"], r["B_over_K"]))
    out = Path(root) / "sweep_table.csv"
    keys = list(rows[0])
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerows([[r[k] for k in keys] for r in rows])
    for r in rows:
        if r["B_over_K"] == 1.5:
            print(f"ps={r['page_size']:>2} K={r['K']:>3}: LRU {r['lru_miss_pct']:.2f}% Belady {r['belady_miss_pct']:.2f}% "
                  f"G score {r['G_score']:.3f} ema {r['G_ema(lodo)']:.3f} allrank {r['G_ema_allrank(lodo)']:.3f} "
                  f"wins {r['ema_wins']}")


def sampling(root):
    rows = []
    for d in sorted(Path(root).glob("seed*")):
        S = load(d / "summary.csv")
        for ratio in (1.25, 1.5, 2.0, 4.0):
            for pol in POLS:
                if (pol, ratio) in S:
                    r = S[pol, ratio]
                    rows.append((d.name, ratio, pol, float(r["miss_rate"]), float(r["G"]), float(r["G_lo"]),
                                 float(r["G_hi"]), r["docs_better_than_lru"]))
    with open(Path(root) / "sampling_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "B_over_K", "policy", "miss_rate", "G", "G_lo", "G_hi", "docs_better_than_lru"])
        w.writerows(rows)
    for r in rows:
        if r[1] == 1.5 and r[2] in ("lru", "score", "ema(lodo)", "ema_allrank(lodo)"):
            print(r)


if __name__ == "__main__":
    if sys.argv[1] == "sweep":
        base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else None
        sweep(sys.argv[2], base)
    else:
        sampling(sys.argv[2])
