"""Poster figures from committed result CSVs only (no traces needed).

  python poster/make_figs.py            # writes poster/figs/*.svg and *.png

F1 information ablation (B=96, dev, M1 Max bars, L4 markers)
F2 budget curve (M1 Max held-out, 21 documents)
F3 per-document score-EMA G across hardware, splits, fresh documents and a new task
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "docs/results"
OUT = ROOT / "poster/figs"
OUT.mkdir(parents=True, exist_ok=True)

# reference palette (light), validated: blue, orange, aqua (+ yellow for a 4th line)
BLUE, ORANGE, AQUA, YELLOW = "#257de0", "#f47230", "#00b894", "#f2a600"
GRAY, INK, INK2, GRID = "#9c9c9c", "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "svg.fonttype": "none",
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def stats(path, B=96):
    return {r["policy"]: r for r in csv.DictReader(open(path)) if int(r["B"]) == B}


def summary(path, B=96):
    return {r["policy"]: r for r in csv.DictReader(open(path)) if int(r["B"]) == B}


def alloc_ema_G(report, B=96):
    m = re.search(rf"B=\s*{B} G: ema \S+\s+alloc\+lru \S+\s+alloc\+ema ([+-][\d.]+)", Path(report).read_text())
    return float(m.group(1))


# ---------------------------------------------------------------- F1 information ablation
def fig1():
    mps, l4 = stats(R / "mps_dev/checks_stats.csv"), stats(R / "longctx_dev/checks_stats.csv")
    ms, ls = summary(R / "mps_dev/summary.csv"), summary(R / "longctx_dev/summary.csv")
    rows = [  # label, source key, kind
        ("Score EMA + per-layer allocation", "ALLOC", "score"),
        ("Learned next-use regression (6 features)", ("S", "linear-dagger"), "score"),
        ("Score EMA (1 float per page)", "ema(z,all) [lodo]", "score"),
        ("InfLLM-style decayed score sum", "InfLLM decayed sum (softmax, all residents) [lodo]", "score"),
        ("Current score only", "score", "score"),
        ("Decayed selection count", "decayed count (no score) [lodo]", "hist"),
        ("Rank bonus (proposal policy)", "rank(stage1-2) [lodo]", "hist"),
        ("EMA of rank", "ema(rank,all) [lodo]", "hist"),
        ("Score EMA, updated only when selected", "ema(z,selected) [lodo]", "hist"),
        ("LRU", "lru", "base"),
        ("ARC", "arc", "base"),
        ("LRU-2", "lru2", "base"),
        ("FIFO", "fifo", "base"),
    ]

    def val(src, s, summ, report):
        if src == "ALLOC":
            return alloc_ema_G(report), None, None
        if isinstance(src, tuple):
            r = summ[src[1]]
        else:
            r = s[src]
        return float(r["G"]), float(r["G_lo"]), float(r["G_hi"])

    col = {"score": BLUE, "hist": ORANGE, "base": GRAY}
    lo_lim, hi_lim = -0.22, 0.5
    fig, ax = plt.subplots(figsize=(10, 6.4))
    y = np.arange(len(rows))[::-1]
    for yi, (label, src, kind) in zip(y, rows):
        g, glo, ghi = val(src, mps, ms, R / "mps_dev/checks_report.txt")
        g4 = val(src, l4, ls, R / "longctx_dev/checks_report.txt")[0]
        shown = max(g, lo_lim)
        ax.barh(yi, shown, height=0.62, color=col[kind], edgecolor="white", linewidth=2)
        if g < lo_lim:  # off-scale
            ax.text(lo_lim + 0.005, yi, f"{g:+.2f} ◂", va="center", ha="left", color="white", fontsize=11, weight="bold")
        else:
            if glo is not None and ghi - glo > 0.004:
                ax.plot([glo, ghi], [yi, yi], color=INK, lw=1.2)
            right = max(g, 0, g4 if g4 is not None else 0, ghi if ghi is not None else 0)
            ax.text(right + 0.016, yi, f"{g:+.3f}", va="center", ha="left", color=INK, fontsize=11)
        if lo_lim <= g4:
            ax.plot(g4, yi, marker="D", ms=6, color=INK, mec="white", mew=1, zorder=5)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=12)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_xlabel("Share of the LRU–Bélády miss gap closed (G), B = 96 pages = 1.5K")
    ax.xaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color=BLUE, label="current score of every resident, every step"),
                       Patch(color=ORANGE, label="only what is seen when a page is selected"),
                       Patch(color=GRAY, label="classic replacement"),
                       Line2D([], [], marker="D", ls="", color=INK, mec="white", label="same policy, L4 traces")],
              loc="lower right", fontsize=10.5, frameon=False)
    ax.set_title("Eviction needs the current scores of pages it did not select", loc="left", fontsize=15, weight="bold", pad=24)
    ax.text(0, 1.01, "14 dev documents, M1 Max traces (bars, 95% doc-bootstrap CI); off-scale values printed in the bar",
            transform=ax.transAxes, fontsize=10.5, color=INK2, va="bottom")
    save(fig, "f1_information_ablation")


# ---------------------------------------------------------------- held-out helpers
def per_doc(path):
    M = defaultdict(lambda: defaultdict(int))  # M[(config, B)][doc]
    A = defaultdict(lambda: defaultdict(int))
    for r in csv.DictReader(open(path)):
        k = (r["config"], int(r["B"]))
        M[k][r["doc"]] += int(r["misses"]); A[k][r["doc"]] += int(r["accesses"])
    return M, A


def G(M, c, B, docs):
    l = sum(M["lru", B][d] for d in docs); o = sum(M["belady", B][d] for d in docs)
    return (l - sum(M[c, B][d] for d in docs)) / (l - o)


def boot(M, c, B, docs, reps=2000):
    rng = np.random.default_rng(0)
    gs = [G(M, c, B, [docs[i] for i in rng.integers(0, len(docs), len(docs))]) for _ in range(reps)]
    return np.percentile(gs, [2.5, 97.5])


# ---------------------------------------------------------------- F2 budget curve
def fig2():
    M, A = per_doc(R / "mps_heldout/misses.csv")
    docs = sorted(M["lru", 96])
    Bs = [72, 80, 96, 128, 256]
    series = [("Score EMA + per-layer allocation", "alloc_ema", AQUA), ("Score EMA", "ema", BLUE),
              ("Current score only", "score", YELLOW), ("Decayed selection count", "dcount", ORANGE)]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    x = np.arange(len(Bs))
    for label, c, colr in series:
        # the frozen allocation is uniform at B=72 and 256 (grid edge), where it equals EMA: draw only 80-128
        idx = [1, 2, 3] if c == "alloc_ema" else list(range(len(Bs)))
        g = [G(M, c, Bs[i], docs) for i in idx]
        ci = [boot(M, c, Bs[i], docs, 500) for i in idx]
        xi = x[idx]
        ax.fill_between(xi, [a for a, _ in ci], [b for _, b in ci], color=colr, alpha=0.15, lw=0)
        ax.plot(xi, g, color=colr, lw=2.5, marker="o", ms=8, mec="white", mew=2)
        if c == "alloc_ema":
            ax.text(xi[-1] + 0.12, g[-1], label, color=INK, va="center", fontsize=11.5)
        else:
            ax.text(xi[-1] + 0.12, g[-1], label, color=INK, va="center", fontsize=11.5)
        if c in ("alloc_ema", "ema"):
            i = idx.index(2)
            ax.text(xi[i], g[i] + 0.025, f"{g[i]:.2f}", ha="center", fontsize=11, color=INK)
    lru = [100 * sum(M["lru", B].values()) / sum(A["lru", B].values()) for B in Bs]
    bel = [100 * sum(M["belady", B].values()) / sum(A["belady", B].values()) for B in Bs]
    ax.set_xticks(x, [f"{B}\n{B / 64:g}K\n{l:.1f}% / {b:.1f}%" for B, l, b in zip(Bs, lru, bel)], fontsize=11)
    ax.set_xlabel("budget B in pages (top), as a multiple of K (middle), LRU / Bélády miss rate (bottom)", fontsize=11)
    ax.text(0.01, 0.98, "allocation is uniform at B=72 and 256, i.e. equal to EMA", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, color=INK2)
    ax.set_ylabel("Share of LRU–Bélády gap closed (G)")
    ax.set_ylim(0, 0.6); ax.set_xlim(-0.3, len(Bs) - 0.3)
    ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    ax.axvspan(1.6, 2.4, color="#f4f4f4", zorder=-1)
    ax.text(2, 0.585, "primary budget", ha="center", fontsize=10, color=INK2)
    ax.set_title("Held-out (21 documents, M1 Max): gains hold at every budget", loc="left", fontsize=15, weight="bold", pad=24)
    save(fig, "f2_budget_curve")


# ---------------------------------------------------------------- F3 per-document robustness
def dev_doc_G(path, cfg="ema:0.5", B=96):
    M, _ = per_doc_dev(path)
    return {d: (M["lru"][d] - M[cfg][d]) / (M["lru"][d] - M["belady"][d]) for d in M["lru"]}


def per_doc_dev(path, B=96):
    M = defaultdict(lambda: defaultdict(int))
    for r in csv.DictReader(open(path)):
        if int(r["B"]) == B:
            M[r["config"]][r["doc"]] += int(r["misses"])
    return M, None


def fig3():
    groups = []
    for label, path in (("L4\ndev (14)", R / "longctx_dev/misses.csv"), ("M1 Max\ndev (14)", R / "mps_dev/misses.csv")):
        d = dev_doc_G(path)
        M, _ = per_doc_dev(path)
        docs = sorted(d)
        pooled = (sum(M["lru"][x] for x in docs) - sum(M["ema:0.5"][x] for x in docs)) / \
                 (sum(M["lru"][x] for x in docs) - sum(M["belady"][x] for x in docs))
        groups.append((label, list(d.values()), pooled, "dev"))
    for label, path, kind in (("L4\nheld-out (21)", R / "longctx_heldout/misses.csv", "held"),
                              ("M1 Max\nheld-out (21)", R / "mps_heldout/misses.csv", "held"),
                              ("M1 Max fresh\nGovReport (3)", R / "v2_fresh/misses.csv", "fresh"),
                              ("M1 Max new task\nNarrativeQA (8)", R / "v2_ood/misses.csv", "fresh")):
        M, _ = per_doc(path)
        docs = sorted(M["lru", 96])
        vals = [G(M, "ema", 96, [x]) for x in docs]
        groups.append((label, vals, G(M, "ema", 96, docs), kind))
    fig, ax = plt.subplots(figsize=(11, 5.0))
    rng = np.random.default_rng(1)
    for i, (label, vals, pooled, kind) in enumerate(groups):
        xs = i + rng.uniform(-0.17, 0.17, len(vals))
        ax.scatter(xs, vals, s=46, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [pooled, pooled], color=INK, lw=2.2, zorder=4)
        ax.text(i + 0.32, pooled, f"{pooled:.3f}", va="center", fontsize=10.5)
    ax.axvspan(1.5, 3.5, color="#f4f4f4", zorder=-1)
    ax.text(2.5, 0.338, "preregistered held-out", ha="center", fontsize=10, color=INK2)
    ax.axvspan(3.5, 5.5, color="#eef6ee", zorder=-1)
    ax.text(4.5, 0.338, "never used before (reported apart)", ha="center", fontsize=10, color=INK2)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=10.5)
    ax.set_ylim(0.2, 0.35); ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("Score-EMA G per document (B = 96)")
    ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    ax.set_title("Every document improves, on both GPUs and on new data", loc="left", fontsize=15, weight="bold", pad=24)
    ax.text(0, 1.01, "dots: documents; bar: pooled G (preregistered floor 0.15 is off-axis). Each GPU captured its own traces and froze the policy on its own dev set.",
            transform=ax.transAxes, fontsize=10, color=INK2, va="bottom")
    save(fig, "f3_per_document")


# ---------------------------------------------------------------- F4 batch scaling (L4)
def fig4():
    rows = list(csv.DictReader(open(R / "batch_bench_l4/batch_bench.csv")))
    get = lambda B, p, k: [float(r[k]) for r in rows if int(r["B"]) == B and r["policy"] == p]
    bs = [int(r["batch"]) for r in rows if int(r["B"]) == 96 and r["policy"] == "lru"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for B, colr in ((96, INK), (128, GRAY)):
        y = [100 * v for v in get(B, "lru", "transfer_share")]
        ax[0].plot(bs, y, color=colr, lw=2.5, marker="o", ms=8, mec="white", mew=2)
        ax[0].text(bs[-1] * 1.15, y[-1], f"B = {B}", va="center", fontsize=11.5)
    ax[0].set_ylabel("LRU page transfer, % of GPU step time"); ax[0].set_ylim(0, 16)
    ax[0].set_title("Transfer share plateaus with batch", loc="left", fontsize=13.5, weight="bold", pad=10)
    for label, p, colr, ls in (("Bélády (oracle)", "belady", GRAY, "--"), ("Score EMA + allocation", "alloc_ema", AQUA, "-"),
                               ("Score EMA", "ema", BLUE, "-")):
        y = [100 * v for v in get(96, p, "speedup_vs_lru")]
        ax[1].plot(bs, y, color=colr, lw=2.5, ls=ls, marker="o", ms=8, mec="white", mew=2)
        ax[1].text(bs[-1] * 1.15, y[-1], f"{label}  {y[-1]:+.1f}%", va="center", fontsize=11.5)
    ax[1].set_ylabel("Step-time speedup vs LRU (%)"); ax[1].set_ylim(0, 6.5)
    ax[1].set_title("B = 96: saved misses become time, capped by that share", loc="left", fontsize=13.5, weight="bold", pad=10)
    for a in ax:
        a.set_xscale("log", base=2); a.set_xticks(bs, [str(b) for b in bs]); a.set_xlim(0.8, 64 * 3.2)
        a.set_xlabel("batch size"); a.yaxis.grid(True, color=GRID, lw=1); a.set_axisbelow(True)
    fig.text(0.0, -0.04, "L4, zero-copy Triton gather at 5.3 µs per 64 KiB page (12.3 GB/s); step = Qwen3-0.6B decode over "
             "1,088 tokens + Quest scoring + transfer, GPU kernel time, no overlap.", fontsize=9.5, color=INK2)
    fig.tight_layout()
    save(fig, "f4_batch_scaling")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
