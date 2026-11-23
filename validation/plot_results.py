"""Render the measured synthetic results; no inferred or published data plotted."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "results"
cpu = json.loads((ROOT / "cpu_experiments.json").read_text())
gpu = json.loads((ROOT / "layer_benchmark.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
values = [r["G_ema"] for r in cpu["synthetic_regimes"]]
ax = axes[0]
bars = ax.bar(["Persistent\nscores", "Independent\nscores", "Alternating\nscores"], values,
              color=["#167d8d", "#8496a8", "#b95147"], width=0.56)
ax.axhline(0, color="#465568", linewidth=0.8)
ax.set_ylim(-1.37, 0.55)
ax.set_ylabel("Fraction of LRU to oracle miss gap closed")
ax.set_title("EMA needs predictive scores", loc="left", weight="bold", pad=14)
for bar, val in zip(bars, values):
    ax.annotate(f"{val:+.3f}", (bar.get_x()+bar.get_width()/2, val),
                ha="center", va="bottom" if val >= 0 else "top",
                xytext=(0, 4 if val >= 0 else -4), textcoords="offset points")
ax.text(0.02, 0.03, "12 seeds per regime · K=8 · B=12", transform=ax.transAxes,
        fontsize=9, color="#465568")
ax = axes[1]
rows = {r["policy"]: r for r in gpu["benchmark"]["summary"] if r["B"] == 96}
names = ["lru", "ema", "belady"]
miss = [rows[n]["median_steady_misses"] / rows["lru"]["median_steady_misses"] for n in names]
lat = [rows[n]["median_of_repeat_medians_ms"] / rows["lru"]["median_of_repeat_medians_ms"] for n in names]
x = np.arange(3)
ax.bar(x-0.18, miss, width=0.34, color="#167d8d", label="Misses")
ax.bar(x+0.18, lat, width=0.34, color="#d98a32", label="Wall time")
ax.axhline(1, linestyle="--", color="#8496a8", linewidth=1)
ax.set_xticks(x, ["LRU", "EMA", "Oracle"])
ax.set_ylim(0, 1.35)
ax.set_ylabel("Relative to LRU · lower is better")
ax.set_title("Fewer misses did not reduce latency", loc="left", weight="bold", pad=14)
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
for offset, series in [(-0.18, miss), (0.18, lat)]:
    for i, val in enumerate(series):
        ax.text(i+offset, val+0.025, f"{val:.2f}", ha="center", fontsize=9)
fig.suptitle("Independent synthetic validation", fontsize=15, weight="bold", x=0.02, ha="left")
fig.supxlabel("Right: RTX 5080 Laptop · one layer · 64 KiB pages · B=96 · 7 repeats · 16 warmup steps excluded", fontsize=9)
fig.savefig(ROOT / "systems_evaluation.png", dpi=180)
plt.close(fig)
