"""Side-by-side table of two efficient_dev/checks summary CSVs (committed L4 vs replication-v2).

  python -m analysis_v2.compare_l4 OLD_summary.csv NEW_summary.csv [--budgets 72 80 96 128 256] [--md]
"""
import argparse
import csv


def load(p):
    return {(r["policy"], int(r["B"])): r for r in csv.DictReader(open(p))}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--budgets", type=int, nargs="+", default=[72, 80, 96, 128, 256])
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    old, new = load(a.old), load(a.new)
    rows = []
    for (p, B), r in sorted(new.items(), key=lambda x: (x[0][1], x[0][0])):
        if B not in a.budgets:
            continue
        o = old.get((p, B), {})
        inside = (f(o.get("G_lo")) <= f(r["G"]) <= f(o.get("G_hi"))) if o else None
        rows.append((p, B, 100 * f(o.get("miss_rate")), 100 * f(r["miss_rate"]), f(o.get("G")), f(o.get("G_lo")),
                     f(o.get("G_hi")), f(r["G"]), f(r["G_lo"]), f(r["G_hi"]), inside))
    if a.md:
        print("| policy | B | L4 miss % | new miss % | L4 G [CI] | new G [CI] | new G inside L4 CI |")
        print("|---|---|---|---|---|---|---|")
        for p, B, om, nm, og, ol, oh, ng, nl, nh, ins in rows:
            print(f"| {p} | {B} | {om:.2f} | {nm:.2f} | {og:.3f} [{ol:.3f}, {oh:.3f}] | {ng:.3f} [{nl:.3f}, {nh:.3f}] | "
                  f"{'–' if ins is None else ('yes' if ins else 'no')} |")
    else:
        for r in rows:
            print("{:<22}{:>4} {:7.2f} {:7.2f}  {:7.3f} [{:.3f},{:.3f}]  {:7.3f} [{:.3f},{:.3f}]  {}".format(*r))


if __name__ == "__main__":
    main()
