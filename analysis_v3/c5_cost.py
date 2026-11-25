"""C5 (followup-v3): G against the replacement policy's per-token work (0.6B dev).

  python -m analysis_v3.c5_cost --out results_v3/c5/cost.csv

Work per token and layer, counted as in results_v2/gpu/flops.csv (3 operations per updated resident value: the z of the
page, one multiply-add for the EMA); the victim selection (a scan or sort of the B residents) is the same for every policy
here and is listed separately. Policies: current score (no state), EMA with every resident refreshed every step, lazy
refresh every r steps (replication-v2 S4, LODO alpha), selected pages + the M lowest-scoring unselected residents (B7; the
partial selection of the M lowest costs B comparisons), selected only, the all-page percentile EMA (sorting all P
eligible pages: P log2 P comparisons), and the kernels of B1 (normalised power law: 8 mixture terms x 2 values per page).
"""
import argparse
import csv
import math
from pathlib import Path

K, P = 64, 1400          # K selected pages; P eligible pages at 16-32K tokens (about 1 000-2 000)


def rd(p):
    return {(r["policy"], int(r["B"])): float(r["G"]) for r in csv.DictReader(open(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s4", default="results_v2/dev/s4/baselines/summary.csv")
    ap.add_argument("--b7", default="results_v3/b7/b7_dev/summary.csv")
    ap.add_argument("--b1", default="results_v3/b1/b1_dev/summary.csv")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    s4, b7, b1 = rd(a.s4), rd(a.b7), rd(a.b1)
    rows = []
    for B in (80, 96, 128, 256):
        full = b7["ema_z(lodo)", B]
        items = [("current score (no state)", s4["score", B], 0, B),
                 ("EMA, every resident every step", full, 3 * B, B)]
        for r in (2, 4, 8):
            items.append((f"EMA, lazy refresh every {r} steps", s4[f"lazy_r{r}(lodo)", B], 3 * B / r, B))
        items.append(("EMA, selected pages only", b7["scope_z_sel_M0(lodo)", B], 3 * K, 0))
        for M in (4, 8, 16, 32):
            items.append((f"EMA, selected + {M} lowest-scoring residents", b7[f"scope_z_bottom_M{M}(lodo)", B],
                          3 * (K + M), B))
        items.append(("EMA of all-page percentile", b7["ema_allrank(lodo)", B], 3 * B + P * math.log2(P), B))
        items.append(("normalised power-law kernel (8 terms)", b1["pow_norm_kth(lodo)", B], 2 * 8 * 2 * B + 3 * B, B))
        items.append(("two-scale normalised EMA", b1["twoscale(lodo)", B], 2 * 3 * B, B))
        for name, g, ops, extra in items:
            rows.append(dict(B=B, policy=name, G=g, share_of_full_ema_G=g / full, update_ops_per_token_layer=ops,
                             extra_compares=extra, meets_90pct=g >= 0.9 * full))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"B={r['B']:>3} {r['policy']:<46} G {r['G']:+.3f} ({100*r['share_of_full_ema_G']:5.1f} %) ops {r['update_ops_per_token_layer']:7.0f}")


if __name__ == "__main__":
    main()
