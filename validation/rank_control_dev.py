"""All-page rank control for the "magnitude versus rank" ablation (dev traces only).

The archived ablation ScoreEMA(signal="rank") gives every unselected page the value 0, so it removes
both score magnitude and the ordering among unselected pages. This script adds the decisive control,
ScoreEMA(signal="allrank"): each page's percentile rank among all eligible pages at the step
(kvpolicy._allrank; mid-rank ties), which keeps the full ordering and discards magnitude.

Three signals are replayed with the same update scope (mode="all"), the same alpha grid (the
ema(z,all) grid of checks.py) and the same leave-one-document-out alpha selection
(min training misses, ties broken by config name):

  z        ema:{a}           score relative to the K-th selected score (main policy)
  allrank  ema_allrank:{a}   percentile rank among all eligible pages (new control)
  rank     ema_rank:{a}      rank among the selected pages, 0 for every unselected page (archived)

  python validation/rank_control_dev.py TRACES --out results/rank_control [--workers 8]
  python validation/rank_control_dev.py TRACES --out results/rank_control_smoke --smoke

TRACES must hold the 14 longctx_v1 dev NPZs (schema v2: ranked, scores, rid). The replay is checked
against every matching row of the archived docs/results/longctx_dev/misses*.csv (lru, belady,
ema:*, ema_rank:*), so a different trace file fails loudly. Outputs under --out:

  misses.csv    doc, layer, config, B, accesses, misses, compulsory
  lodo.csv      signal, B, doc, chosen, misses
  summary.csv   signal x B: misses, miss rate, G and repeat-miss drop vs LRU with document bootstrap
  contrast.csv  paired differences in G (allrank - z, allrank - rank) with document bootstrap CIs
  summary.json  trace hashes, settings and the B=96 headline

Reading the result: allrank close to z means the ordering among unselected pages carries the signal
and magnitude adds little; allrank close to rank means magnitude is what matters. The archived
G ~ 0.016 for rank at B=96 used only alpha in {0.3, 0.5, 0.7, 0.9}; this runner gives every signal
the same 10-value grid, so its rank row can differ slightly. Compare signals within contrast.csv.
Dev only: this is not part of the frozen held-out evaluation (splits/heldout_frozen_v1.json).
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import efficient_dev as ed  # noqa: E402
import kvpolicy as kp  # noqa: E402

ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)  # checks.py FAMILIES["ema(z,all)"]
BUDGETS = (72, 80, 96, 128, 256)  # budgets of the archived ablation (misses_ext.csv)
PRIMARY = 96
SIGNALS = {"z": "ema", "allrank": "ema_allrank", "rank": "ema_rank"}
ARCHIVE = ROOT / "docs/results/longctx_dev"
REPS, SEED = 2000, 0


def configs(signals, alphas):
    out = {"lru": ("lru", {}), "belady": ("belady", {})}
    for sig in signals:
        for a in alphas:
            out[f"{SIGNALS[sig]}:{a}"] = ("ema", dict(alpha=a, mode="all", signal=sig))
    return out


def dev_inputs(folder, smoke, split_check=True):
    expected = None
    if split_check:
        split = json.loads((ROOT / "splits/longctx_v1.json").read_text())
        expected = {e["rid"] for e in split["prompts"] if e["split"] == "dev"}
    inputs = {}
    for path in sorted(folder.glob("*.npz")):
        with np.load(path, allow_pickle=False) as trace:
            if not {"rid", "ranked", "scores"}.issubset(trace.files):
                raise ValueError(f"{path.name} is missing rid, ranked or scores (schema v2 needed)")
            rid = str(trace["rid"])
        if rid in inputs or (expected is not None and rid not in expected):
            raise ValueError(f"Unexpected or duplicate dev ID in {path.name}: {rid}")
        inputs[rid] = path
    if not inputs or (expected is not None and not smoke and inputs.keys() != expected):
        missing = sorted((expected or set()) - inputs.keys())
        raise ValueError(f"Expected the frozen dev traces; missing: {missing}")
    items = sorted(inputs.items())
    return dict(items[:1] if smoke else items)


def replay_job(job):
    rid, path, config, spec, budgets, layers = job
    name, params = spec
    R, S, _ = kp.load(path)
    rows = []
    for l in (range(R.shape[1]) if layers is None else range(layers)):
        for B in budgets:
            c = kp.simulate(R[:, l], S[:, l], B, name, **params)
            rows.append((rid, l, config, B, c["accesses"], c["misses"], c["compulsory"]))
    return rows


def archived_counts(archive):
    out = {}
    for fn in ("misses.csv", "misses_ext.csv"):
        path = archive / fn
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                out[r["doc"], int(r["layer"]), r["config"], int(r["B"])] = (
                    int(r["accesses"]), int(r["misses"]), int(r["compulsory"]))
    return out


def check_archive(rows, archive):
    ref = archived_counts(archive)
    n = 0
    for doc, l, config, B, acc, miss, comp in rows:
        old = ref.get((doc, l, config, B))
        if old is None:
            continue
        if old != (acc, miss, comp):
            raise ValueError(f"Replay differs from archived {archive.name}: {doc} layer {l} {config} B={B}: "
                             f"{(acc, miss, comp)} vs {old}")
        n += 1
    return n


def lodo(per, docs, cfgs):
    """per[config][doc] = misses. Same rule as checks.py / efficient_dev.py."""
    out, chosen = [], []
    for held in docs:
        train = [d for d in docs if d != held]
        best = min(cfgs, key=lambda c: (sum(per[c][d] for d in train), c))
        chosen.append(best)
        out.append(per[best][held])
    return np.array(out, float), chosen


def analyse(rows, signals, alphas, budgets):
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # per[B][config][doc]
    acc = defaultdict(lambda: defaultdict(int))
    comp = defaultdict(lambda: defaultdict(int))
    for doc, _, config, B, a, m, c in rows:
        per[B][config][doc] += m
        if config == "lru":
            acc[B][doc] += a
            comp[B][doc] += c
    docs = sorted(per[budgets[0]]["lru"])
    idx = np.random.default_rng(SEED).integers(0, len(docs), (REPS, len(docs)))
    lodo_rows, summary, contrast, headline = [], [], [], {}
    for B in budgets:
        lru = np.array([per[B]["lru"][d] for d in docs], float)
        opt = np.array([per[B]["belady"][d] for d in docs], float)
        C = np.array([comp[B][d] for d in docs], float)
        A = sum(acc[B].values())

        def G(pol, i):
            return (lru[i].sum() - pol[i].sum()) / (lru[i].sum() - opt[i].sum())

        def RD(pol, i):
            return 1 - (pol[i].sum() - C[i].sum()) / (lru[i].sum() - C[i].sum())

        allidx = np.arange(len(docs))
        pols = {}
        for sig in signals:
            cfgs = [f"{SIGNALS[sig]}:{a}" for a in alphas]
            pol, chosen = lodo(per[B], docs, cfgs)
            pols[sig] = pol
            lodo_rows += [(sig, B, d, c, int(m)) for d, c, m in zip(docs, chosen, pol)]
            Gs = [G(pol, i) for i in idx]
            Rs = [RD(pol, i) for i in idx]
            picks = " ".join(f"{c}x{chosen.count(c)}" for c in sorted(set(chosen)))
            row = dict(signal=sig, B=B, misses=int(pol.sum()), accesses=A, miss_rate=pol.sum() / A,
                       G_uniform_anchors=G(pol, allidx), G_lo=np.percentile(Gs, 2.5),
                       G_hi=np.percentile(Gs, 97.5), repeat_miss_drop_vs_lru=RD(pol, allidx),
                       rd_lo=np.percentile(Rs, 2.5), rd_hi=np.percentile(Rs, 97.5),
                       docs_better_than_lru=int((pol < lru).sum()), n_docs=len(docs), lodo_picks=picks)
            summary.append(row)
            if B == PRIMARY:
                headline[sig] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                 for k, v in row.items()}
        for a, b in (("allrank", "z"), ("allrank", "rank"), ("z", "rank")):
            if a in pols and b in pols:
                ds = [G(pols[a], i) - G(pols[b], i) for i in idx]
                contrast.append((B, f"{a}-{b}", G(pols[a], allidx) - G(pols[b], allidx),
                                 np.percentile(ds, 2.5), np.percentile(ds, 97.5)))
    return lodo_rows, summary, contrast, headline


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", type=Path, help="Directory with the 14 longctx_v1 dev NPZs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    ap.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    ap.add_argument("--signals", nargs="+", default=list(SIGNALS), choices=list(SIGNALS))
    ap.add_argument("--smoke", action="store_true", help="Layer 0 of one dev document (pipeline check only)")
    ap.add_argument("--archive", type=Path, default=ARCHIVE, help="Archived dev CSVs to cross-check against")
    ap.add_argument("--no-split-check", action="store_true", help="Accept any rid (synthetic tests only)")
    args = ap.parse_args(argv)
    budgets = sorted(set(args.budgets))

    inputs = dev_inputs(args.traces, args.smoke, split_check=not args.no_split_check)
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = configs(args.signals, args.alphas)
    layers = 1 if args.smoke else None
    jobs = [(rid, str(p), c, spec, budgets, layers) for rid, p in inputs.items() for c, spec in cfg.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for part in pool.map(replay_job, jobs) for r in part]
    rows.sort(key=lambda r: (r[0], r[2], r[3], r[1]))
    ed.write(args.out / "misses.csv", ["doc", "layer", "config", "B", "accesses", "misses", "compulsory"], rows)
    checked = check_archive(rows, args.archive) if args.archive and args.archive.exists() else 0

    meta = dict(status="dev control (not part of the frozen held-out evaluation)", budgets=budgets,
                primary=PRIMARY, alphas=args.alphas, signals=args.signals, mode="all",
                selection="leave-one-document-out, min training misses, ties by config name",
                ties="allrank uses mid-ranks over finite scores (kvpolicy._allrank)",
                smoke=args.smoke, archive_rows_matched=checked, traces=[])
    for rid, path in inputs.items():
        with path.open("rb") as f:
            meta["traces"].append({"rid": rid, "sha256": hashlib.sha256(f.read()).hexdigest()})
    if len(inputs) >= 2:
        lodo_rows, summary, contrast, headline = analyse(rows, args.signals, args.alphas, budgets)
        ed.write(args.out / "lodo.csv", ["signal", "B", "doc", "chosen", "misses"], lodo_rows)
        keys = list(summary[0])
        ed.write(args.out / "summary.csv", keys, [[r[k] for k in keys] for r in summary])
        ed.write(args.out / "contrast.csv", ["B", "contrast", "dG", "dG_lo", "dG_hi"], contrast)
        meta["headline_B96"] = headline
        for r in summary:
            print(f"B={r['B']:>3} {r['signal']:<8} miss {100 * r['miss_rate']:.2f}%  "
                  f"G {r['G_uniform_anchors']:+.3f} [{r['G_lo']:+.3f},{r['G_hi']:+.3f}]  "
                  f"repeat-drop {100 * r['repeat_miss_drop_vs_lru']:+.1f}%  picks {r['lodo_picks']}")
        for B, name, d, lo, hi in contrast:
            print(f"B={B:>3} dG {name:<13} {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
    (args.out / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Archived rows matched: {checked}.  Results: {args.out}")


if __name__ == "__main__":
    main()
