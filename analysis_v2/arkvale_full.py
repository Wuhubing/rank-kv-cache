"""ArkVale full version vs the score EMA (dev only; docs/rerun/s4/09_arkvale_full.md).

  python -m analysis_v2.arkvale_full compare  OLD NEW [--e2e] --out FILE
  python -m analysis_v2.arkvale_full replay   TRACES --out DIR [--e2e] [--procs 32]
  python -m analysis_v2.arkvale_full audit    TRACES --out DIR [--e2e] [--procs 14]
  python -m analysis_v2.arkvale_full summarize --out DIR [--e2e]
  python -m analysis_v2.arkvale_full agree    TRACES --out FILE

compare: NEW traces (with --record-arkvale-digest) against OLD (same documents, no read-out): ranked,
  scores (bitwise, NaN pattern included) and the generated tokens must be identical. With --e2e, NEW was
  collected with --selector arkvale-mean: report the token agreement with OLD instead.
replay: misses.csv with one row per (doc, layer, stream, config, B). Streams (forced pages merged into
  every step's access set, for every policy): plain (none), page (1 sink + 1 recent page: the code's
  defaults n_sink_pages=1, n_win_pages=2 at 32-token pages, same number of pages), token (2 + 2: same
  number of tokens), page_is / token_is (InferState's default n_sink_pages=2: 2 + 1, 4 + 2). Budgets
  BUDGETS on all layers, and "eq" rows: the equal-total-slots budget of layers 2.. when layers 0 and 1
  keep every page (kvpolicy_v2.unlimited_budget), where it is feasible.
  Configs: lru, belady, score (current Quest score), ema_z:a, ema_allrank:a, powerlaw:g (the S4 grids),
  arkvale_simple (kvpolicy_v2.ArkVale 2+2, Quest scores, the S4 version), and ark:<digest>:<exit>
  (kvpolicy_v2.simulate_arkvale with the stream's sink/recent, eviction by that digest, window exit on/off).
  With --e2e (traces from --selector arkvale-mean) only the page stream exists, and the digest is the
  trace's own score.
audit: LRU and Belady of every stream and budget recomputed with replay.py's independent simulator from
  forced rows built here independently; exit status 1 on any mismatch.
summarize: G (LRU-Belady anchors of the same stream and layers), document bootstrap (2000, seed 0),
  LODO for the EMA / allrank / power-law families, paired contrasts.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import kvpolicy as kp
import kvpolicy_v2 as k2

K = 64
L_UNLIMITED = 2
BUDGETS = [72, 80, 96, 128, 256]
EQ_NOMINAL = [96, 128, 192, 256, 384, 512]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
GAMMAS = [0.5, 1.0, 2.0]
DIGESTS = ["quest", "cuboid_mean", "cuboid_center", "sphere_max", "sphere_center", "sphere_mean"]
STREAMS = {"plain": (0, 0), "page": (1, 1), "token": (2, 2), "page_is": (2, 1), "token_is": (4, 2)}
REPS, SEED = 2000, 0
CODE_DEFAULT = "ark:cuboid_mean:1"     # ArkVale as released: cuboid-mean digest, window exit


def configs(e2e=False, digests=None):
    C = {"lru": ("lru", {}), "belady": ("belady", {}), "score": ("score", {})}
    for a in ALPHAS:
        C[f"ema_z:{a}"] = ("ema", dict(alpha=a, signal="z"))
        C[f"ema_allrank:{a}"] = ("ema", dict(alpha=a, signal="allrank"))
    for g in GAMMAS:
        C[f"powerlaw:{g}"] = ("powerlaw", dict(gamma=g))
    if not e2e:
        C["arkvale_simple"] = ("arkvale", dict(sink=2, win=2))
    for d in (["own"] if e2e else (digests or DIGESTS)):
        for ex in (1, 0):
            C[f"ark:{d}:{ex}"] = ("ark", dict(digest=d, window_exit=bool(ex)))
    return C


FAMILIES = {"ema_z": [f"ema_z:{a}" for a in ALPHAS], "ema_allrank": [f"ema_allrank:{a}" for a in ALPHAS],
            "powerlaw": [f"powerlaw:{g}" for g in GAMMAS]}

_CACHE = {}


def load(path):
    """ranked, scores, rid (kvpolicy.load, cached for the last path)."""
    if path not in _CACHE:
        _CACHE.clear()
        _CACHE[path] = kp.load(path)
    return _CACHE[path]


def load_digest(path, name):
    """One digest read-out [T, L, P] (float16 as stored; only the array a job needs is read)."""
    with np.load(path, allow_pickle=False) as z:
        return z[f"digest_{name}"]


def run_job(job):
    path, stream, cfg, e2e, std_budgets, do_eq = job
    R, S, rid = load(path)
    sink, recent = STREAMS[stream]
    pol, kw = configs(e2e)[cfg]   # every digest config is in the full set
    D = load_digest(path, kw["digest"]) if pol == "ark" and kw["digest"] not in ("quest", "own") else None
    T, L, P = S.shape
    budgets = [("std", B, B) for B in std_budgets]
    for nb in (EQ_NOMINAL if do_eq else []):
        b = k2.unlimited_budget(nb, L, L_UNLIMITED, P)
        if b >= K + sink + recent:
            budgets.append(("eq", nb, b))
    out = []
    for l in range(L):
        r, s = R[:, l], S[:, l]
        if pol == "ark":
            d = kw["digest"]
            est = s if D is None else D[:, l].astype(np.float32)
            stream_rows = None
        else:
            stream_rows = r if (sink, recent) == (0, 0) else k2.with_forced(r, s, sink, recent)
        for kind, nb, B in budgets:
            if kind == "eq" and l < L_UNLIMITED:
                continue
            if pol == "ark":
                c = k2.simulate_arkvale(r, s, B, est=est, sink=sink, recent=recent, window_exit=kw["window_exit"])
            else:
                c = kp.simulate(stream_rows, s, B, pol, **kw)
                c["window_exits"] = 0
            out.append((rid, l, stream, cfg, kind, nb, B, c["accesses"], c["misses"], c["compulsory"],
                        c["window_exits"]))
    return out


HEADER = ["doc", "layer", "stream", "config", "kind", "nominal_B", "B", "accesses", "misses", "compulsory",
          "window_exits"]


def cmd_replay(args):
    paths = sorted(str(p) for p in args.traces.glob("*.npz"))
    streams = ["page"] if args.e2e else (args.streams or list(STREAMS))
    C = configs(args.e2e, args.digests)
    jobs = [(p, st, c, args.e2e, args.budgets or BUDGETS, not args.no_eq) for p in paths for st in streams for c in C]
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    with Pool(args.procs) as pool, open(args.out / "misses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for rows in pool.imap_unordered(run_job, jobs, chunksize=1):
            w.writerows(rows)
            n += 1
            if n % 100 == 0:
                print(f"{n}/{len(jobs)} jobs", flush=True)
                f.flush()
    print("replay done", flush=True)


# ------------------------------------------------------------------------------------------ audit

def independent_rows(R, S, sink, recent):
    """Forced access rows built without kvpolicy_v2: completed pages = finite scores of the row."""
    rows = []
    for t in range(R.shape[0]):
        hp = int(np.count_nonzero(~np.isnan(S[t])))
        acc = {int(p) for p in R[t] if p >= 0}
        acc.update(range(min(sink, hp)))
        acc.update(range(max(0, hp - recent), hp))
        rows.append(sorted(acc))
    return rows


def audit_doc(job):
    from replay import replay_stream
    path, budgets_by_stream = job
    R, S, rid = kp.load(path)
    out = {}
    for stream, budgets in budgets_by_stream.items():
        sink, recent = STREAMS[stream]
        for l in range(R.shape[1]):
            rows = independent_rows(R[:, l], S[:, l], sink, recent)
            for B in budgets.get(l, ()):
                for pol in ("lru", "belady"):
                    c = replay_stream(rows, B, pol)
                    out[rid, l, stream, pol, B] = (c["accesses"], c["misses"], c["compulsory_misses"])
    return out


def cmd_audit(args):
    ref = {}
    need = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for path in misses_files(args.out):
      with open(path) as f:
        for r in csv.DictReader(f):
            if r["config"] in ("lru", "belady"):
                key = (r["doc"], int(r["layer"]), r["stream"], r["config"], int(r["B"]))
                ref[key] = (int(r["accesses"]), int(r["misses"]), int(r["compulsory"]))
                need[r["doc"]][r["stream"]][int(r["layer"])].add(int(r["B"]))
    paths = sorted(str(p) for p in args.traces.glob("*.npz"))
    rid_of = {}
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            rid_of[p] = str(z["rid"])
    jobs = [(p, {s: dict(v) for s, v in need[rid_of[p]].items()}) for p in paths]
    ind = {}
    with Pool(args.procs) as pool:
        for part in pool.imap_unordered(audit_doc, jobs):
            ind.update(part)
    bad = [(k, ref[k], v) for k, v in ind.items() if ref.get(k) != v]
    missing = [k for k in ref if k not in ind]
    lines = [f"traces: {len(paths)}; LRU/Belady streams in misses.csv: {len(ref)}; recomputed with replay.py: {len(ind)}",
             f"mismatches: {len(bad)}; not recomputed: {len(missing)}"]
    lines += [f"MISMATCH {k}: kvpolicy {a} replay.py {b}" for k, a, b in bad[:50]]
    (args.out / "audit.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(1 if bad or missing or not ind else 0)


# ------------------------------------------------------------------------------------------ compare

def cmd_compare(args):
    lines = []
    ok = True
    agree = []
    for new in sorted(args.new.glob("*.npz")):
        old = args.old / new.name
        mo = json.loads(old.with_suffix(".json").read_text())
        mn = json.loads(new.with_suffix(".json").read_text())
        go, gn = mo["generated"], mn["generated"]
        if args.e2e:
            n = min(len(go), len(gn))
            prefix = next((i for i in range(n) if go[i] != gn[i]), n)
            same_pos = sum(a == b for a, b in zip(go, gn)) / max(len(go), len(gn))
            agree.append((mn["rid"], len(go), len(gn), prefix, same_pos, mo["stop"], mn["stop"]))
            continue
        with np.load(old, allow_pickle=False) as a, np.load(new, allow_pickle=False) as b:
            same_r = a["ranked"].shape == b["ranked"].shape and np.array_equal(a["ranked"], b["ranked"])
            sa, sb = a["scores"], b["scores"]
            same_s = sa.shape == sb.shape and np.array_equal(sa.view(np.uint16), sb.view(np.uint16))
            fields = sorted(f for f in b.files if f.startswith("digest_"))
            shapes_ok = all(b[f].shape == sb.shape and np.array_equal(np.isnan(b[f]), np.isnan(sb)) for f in fields)
        same_g = go == gn and mo["stop"] == mn["stop"]
        ok &= same_r and same_s and same_g and shapes_ok and len(fields) == 5
        lines.append(f"{mn['rid']}: steps {mo['steps']}/{mn['steps']} ranked {'=' if same_r else 'DIFF'} "
                     f"scores {'=' if same_s else 'DIFF'} tokens {'=' if same_g else 'DIFF'} "
                     f"digests {len(fields)} {'layout ok' if shapes_ok else 'LAYOUT DIFF'}")
    if args.e2e:
        hdr = "rid quest_tokens arkvale_tokens identical_prefix same_position_share quest_stop arkvale_stop"
        lines = [hdr] + [" ".join(str(round(x, 4)) if isinstance(x, float) else str(x) for x in r) for r in agree]
        tq, ta = sum(r[1] for r in agree), sum(r[2] for r in agree)
        lines.append(f"documents {len(agree)}; tokens Quest {tq}, ArkVale {ta}; identical tokens before the first "
                     f"difference: median {int(np.median([r[3] for r in agree]))}, mean {np.mean([r[3] for r in agree]):.1f}; "
                     f"same-position agreement (over the longer output): mean {np.mean([r[4] for r in agree]):.3f}")
    else:
        lines.append(f"ALL IDENTICAL: {ok}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not args.e2e and not ok:
        sys.exit(1)


# ------------------------------------------------------------------------------------------ summarize

def read(paths):
    """M[(stream, kind, nominal, config, scope)][doc] = [accesses, misses, compulsory]; scope 'all' = every
    layer (std only), 'a' = layers >= 2 (std), 'b' = layers >= 2 at the equal-slots budget (eq rows)."""
    M = defaultdict(lambda: defaultdict(lambda: np.zeros(3, np.int64)))
    for path in paths:
      with open(path) as f:
        for r in csv.DictReader(f):
            v = np.array([int(r["accesses"]), int(r["misses"]), int(r["compulsory"])])
            key = (r["stream"], int(r["nominal_B"]), r["config"])
            if r["kind"] == "std":
                M[key + ("all",)][r["doc"]] += v
                if int(r["layer"]) >= L_UNLIMITED:
                    M[key + ("a",)][r["doc"]] += v
            else:
                M[key + ("b",)][r["doc"]] += v
    return M


def misses_files(out):
    """misses.csv plus any misses_*.csv (extra replays with other budgets) in the folder."""
    return [out / "misses.csv"] + sorted(out.glob("misses_*.csv"))


def g_of(lru, pol, opt):
    d = lru - opt
    return (lru - pol) / d if d else float("nan")


def cmd_summarize(args):
    M = read(misses_files(args.out))
    present = {k[0] for k in M}
    streams = [s for s in (["page"] if args.e2e else list(STREAMS)) if s in present]
    have = {k[2] for k in M}
    C = [c for c in configs(args.e2e) if c in have]
    idx_cache = {}
    rows, contrasts, choices = [], [], []
    for stream in streams:
        for scope in ("all", "a", "b"):
            for nb in sorted({k[1] for k in M if k[0] == stream and k[3] == scope}):
                base = (stream, nb)
                if not M.get(base + ("lru", scope)):
                    continue
                Cb = [c for c in C if M.get(base + (c, scope))]   # configs replayed at this budget
                docs = sorted(M[base + ("lru", scope)])
                if scope == "b":   # documents where the equal-slots budget is feasible for every config
                    docs = [d for d in docs if all(d in M[base + (c, scope)] for c in Cb)]
                if not docs:
                    continue
                n = len(docs)
                if n not in idx_cache:
                    idx_cache[n] = np.random.default_rng(SEED).integers(0, n, size=(REPS, n))
                idx = idx_cache[n]
                per = {c: np.array([M[base + (c, scope)][d][1] for d in docs]) for c in Cb}
                acc = np.array([M[base + ("lru", scope)][d][0] for d in docs])
                comp = np.array([M[base + ("lru", scope)][d][2] for d in docs])
                for fam, cfgs in FAMILIES.items():
                    cfgs = [c for c in cfgs if c in per]
                    if not cfgs:
                        continue
                    out = []
                    for i, d in enumerate(docs):
                        train = [j for j in range(n) if j != i]
                        best = min(cfgs, key=lambda c: (per[c][train].sum(), c))
                        out.append(per[best][i])
                        choices.append((stream, scope, nb, fam, d, best))
                    per[f"{fam}(lodo)"] = np.array(out)
                dg = [c for c in Cb if c.startswith("ark:") and c.endswith(":1")]
                if len(dg) > 1:   # tuned digest (LODO over the digests, window exit on)
                    out = []
                    for i, d in enumerate(docs):
                        train = [j for j in range(n) if j != i]
                        best = min(dg, key=lambda c: (per[c][train].sum(), c))
                        out.append(per[best][i])
                        choices.append((stream, scope, nb, "ark_digest", d, best))
                    per["ark:digest(lodo):1"] = np.array(out)
                lru, opt = per["lru"], per["belady"]
                for c, pol in per.items():
                    g = g_of(lru.sum(), pol.sum(), opt.sum())
                    gs = [g_of(lru[i].sum(), pol[i].sum(), opt[i].sum()) for i in idx]
                    lo, hi = np.nanpercentile(gs, [2.5, 97.5])
                    rows.append(dict(stream=stream, scope=scope, nominal_B=nb, policy=c, docs=n,
                                     misses=int(pol.sum()), accesses=int(acc.sum()), miss_rate=pol.sum() / acc.sum(),
                                     compulsory=int(comp.sum()), G=g, G_lo=lo, G_hi=hi,
                                     docs_better_than_lru=int((lru > pol).sum())))
                ark = "ark:own:1" if args.e2e else CODE_DEFAULT
                pairs = [(ark, "ema_z(lodo)"), (ark, "ema_allrank(lodo)"), (ark, "powerlaw(lodo)"),
                         (ark, "score"), ("ema_z(lodo)", "score"), ("ema_z(lodo)", "lru")]
                if not args.e2e:
                    pairs += [(ark, "arkvale_simple"), ("ark:digest(lodo):1", "ema_z(lodo)"),
                              ("ark:quest:1", "ema_z(lodo)"), (ark, "ark:quest:1"), (ark, "ark:cuboid_mean:0")]
                for a, b in pairs:
                    if a not in per or b not in per:
                        continue
                    pa, pb = per[a], per[b]
                    d0 = g_of(lru.sum(), pa.sum(), opt.sum()) - g_of(lru.sum(), pb.sum(), opt.sum())
                    ds = [g_of(lru[i].sum(), pa[i].sum(), opt[i].sum()) - g_of(lru[i].sum(), pb[i].sum(), opt[i].sum())
                          for i in idx]
                    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
                    contrasts.append(dict(stream=stream, scope=scope, nominal_B=nb, a=a, b=b, dG=d0, lo=lo, hi=hi,
                                          docs=n, docs_a_better=int((pa < pb).sum()), docs_b_better=int((pb < pa).sum())))
    for name, data in (("summary.csv", rows), ("contrasts.csv", contrasts)):
        with open(args.out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    # (b) against the plain all-layer budget: total misses over every layer (layers 0-1 have none under (b))
    eqs = []
    for stream in streams:
        for nb in sorted({k[1] for k in M if k[0] == stream and k[3] == "b"}):
            if not M.get((stream, nb, "lru", "all")):
                continue
            docs = sorted(d for d in M[stream, nb, "lru", "b"] if all(d in M[stream, nb, c, "b"] for c in C))

            def lodo_total(fam, scope):
                cfgs = [c for c in FAMILIES[fam] if c in C]
                tot = 0
                for d in docs:
                    best = min(cfgs, key=lambda c: (sum(M[stream, nb, c, scope][x][1] for x in docs if x != d), c))
                    tot += M[stream, nb, best, scope][d][1]
                return tot
            for c in list(C) + [f"{f}(lodo)" for f in FAMILIES]:
                if c.endswith("(lodo)"):
                    fam = c[:-6]
                    if not M.get((stream, nb, FAMILIES[fam][0], "all")):
                        continue
                    mb, ms = lodo_total(fam, "b"), lodo_total(fam, "all")
                    acc = sum(M[stream, nb, "lru", "all"][d][0] for d in docs)
                    eqs.append(dict(stream=stream, nominal_B=nb, policy=c, docs=len(docs), misses_unlimited_eq=mb,
                                    misses_all_layers_B=ms, accesses_all_layers=acc, miss_rate_unlimited_eq=mb / acc,
                                    miss_rate_all_layers_B=ms / acc, ratio=mb / ms if ms else float("nan")))
                    continue
                if not M.get((stream, nb, c, "all")):
                    continue
                mb = sum(M[stream, nb, c, "b"][d][1] for d in docs)
                ms = sum(M[stream, nb, c, "all"][d][1] for d in docs)
                acc = sum(M[stream, nb, "lru", "all"][d][0] for d in docs)
                eqs.append(dict(stream=stream, nominal_B=nb, policy=c, docs=len(docs), misses_unlimited_eq=mb,
                                misses_all_layers_B=ms, accesses_all_layers=acc, miss_rate_unlimited_eq=mb / acc,
                                miss_rate_all_layers_B=ms / acc, ratio=mb / ms if ms else float("nan")))
    if eqs:
        with open(args.out / "eq_vs_std.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(eqs[0]))
            w.writeheader()
            w.writerows(eqs)
    with open(args.out / "lodo_choices.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream", "scope", "nominal_B", "family", "doc", "chosen"])
        w.writerows(choices)
    # feasibility of the equal-slots budget
    feas = []
    with open(args.out / "misses.csv") as f:
        seen = set()
        for r in csv.DictReader(f):
            if r["kind"] == "eq" and r["config"] == "lru" and r["layer"] == str(L_UNLIMITED):
                k = (r["stream"], r["nominal_B"], r["doc"])
                if k not in seen:
                    seen.add(k)
                    feas.append((r["stream"], int(r["nominal_B"]), r["doc"], int(r["B"])))
    with open(args.out / "eq_budgets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream", "nominal_B", "doc", "limited_layer_B"])
        w.writerows(sorted(feas))
    print(f"summary rows {len(rows)}, contrasts {len(contrasts)}")


# ------------------------------------------------------------------------------------------ agree

def agree_doc(path):
    """Per digest: mean overlap of its top-K with Quest's selected top-K and mean Spearman rho with the
    Quest scores, over every 8th step of every layer."""
    R, S, rid = kp.load(path)
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for name in DIGESTS[1:]:
            D = z[f"digest_{name}"]
            ov, rho = [], []
            for l in range(R.shape[1]):
                for t in range(0, R.shape[0], 8):
                    hp = int(np.isfinite(S[t, l]).sum())
                    d = D[t, l, :hp].astype(np.float64)
                    q = S[t, l, :hp].astype(np.float64)
                    top = np.argsort(-d, kind="stable")[:K]
                    ov.append(len(set(top.tolist()) & set(int(x) for x in R[t, l] if x >= 0)) / K)
                    rd = np.argsort(np.argsort(d)); rq = np.argsort(np.argsort(q))
                    rho.append(np.corrcoef(rd, rq)[0, 1])
            out[name] = (float(np.mean(ov)), float(np.mean(rho)))
    return rid, out


def cmd_agree(args):
    paths = sorted(str(p) for p in args.traces.glob("*.npz"))
    with Pool(min(args.procs, len(paths))) as pool:
        res = pool.map(agree_doc, paths)
    lines = ["digest top64_overlap_with_quest spearman_with_quest (mean over documents; every 8th step, all layers)"]
    for name in DIGESTS[1:]:
        lines.append(f"{name} {np.mean([r[1][name][0] for r in res]):.3f} {np.mean([r[1][name][1] for r in res]):.3f}")
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ------------------------------------------------------------------------------------------ tables

def cmd_tables(args):
    """Markdown tables for docs/rerun/s4/09_arkvale_full.md from summary.csv / contrasts.csv."""
    S = {}
    with open(args.out / "summary.csv") as f:
        for r in csv.DictReader(f):
            S[r["stream"], r["scope"], int(r["nominal_B"]), r["policy"]] = r
    Cn = []
    with open(args.out / "contrasts.csv") as f:
        Cn = list(csv.DictReader(f))

    def g(st, sc, B, pol, ci=False):
        r = S.get((st, sc, B, pol))
        if r is None:
            return "–"
        x = f"{float(r['G']):.3f}"
        return x + (f" [{float(r['G_lo']):.3f}, {float(r['G_hi']):.3f}]" if ci else "")

    def mr(st, sc, B, pol):
        r = S.get((st, sc, B, pol))
        return "–" if r is None else f"{100 * float(r['miss_rate']):.2f}"

    out = []
    streams = sorted({k[0] for k in S}, key=lambda x: list(STREAMS).index(x))
    if args.e2e:
        pols = ["lru", "score", "ema_z(lodo)", "ema_allrank(lodo)", "powerlaw(lodo)", "ark:own:1", "ark:own:0", "belady"]
    else:
        pols = ["lru", "score", "ema_z(lodo)", "ema_allrank(lodo)", "powerlaw(lodo)", "arkvale_simple", "ark:quest:1",
                "ark:quest:0", "ark:cuboid_mean:1", "ark:cuboid_mean:0", "ark:digest(lodo):1", "belady"]
    for sc in ("all", "a", "b"):
        budgets = sorted({k[2] for k in S if k[1] == sc})
        for st in streams:
            if not any((st, sc, B, "lru") in S for B in budgets):
                continue
            Bs = [B for B in budgets if (st, sc, B, "lru") in S]
            out.append(f"\n### stream `{st}`, scope `{sc}` — G (miss rate %)\n")
            if sc == "b":
                out.append("documents with a feasible equal-slots budget: " +
                           ", ".join(f"B={B}: {S[st, sc, B, 'lru']['docs']}" for B in Bs) + "\n")
            out.append("| policy | " + " | ".join(f"B={B}" for B in Bs) + " |")
            out.append("|---|" + "---|" * len(Bs))
            for pol in pols:
                if not any((st, sc, B, pol) in S for B in Bs):
                    continue
                out.append(f"| {pol} | " + " | ".join(f"{g(st, sc, B, pol, ci=B in (96, 128) and pol not in ('lru', 'belady'))} ({mr(st, sc, B, pol)})"
                                                        for B in Bs) + " |")
    out.append("\n### Paired contrasts (dG = G(a) - G(b), 95 % document bootstrap; docs a better / b better)\n")
    out.append("| stream | scope | B | a | b | dG [CI] | docs |")
    out.append("|---|---|---|---|---|---|---|")
    for r in Cn:
        out.append(f"| {r['stream']} | {r['scope']} | {r['nominal_B']} | {r['a']} | {r['b']} | "
                   f"{float(r['dG']):+.3f} [{float(r['lo']):+.3f}, {float(r['hi']):+.3f}] | "
                   f"{r['docs_a_better']}/{r['docs_b_better']} of {r['docs']} |")
    (args.out / "tables.md").write_text("\n".join(out) + "\n")
    print(f"wrote {args.out / 'tables.md'} ({len(out)} lines)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("old", type=Path)
    c.add_argument("new", type=Path)
    c.add_argument("--e2e", action="store_true")
    c.add_argument("--out", type=Path, required=True)
    for name in ("replay", "audit"):
        p = sub.add_parser(name)
        p.add_argument("traces", type=Path)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--e2e", action="store_true")
        p.add_argument("--procs", type=int, default=32)
        p.add_argument("--streams", nargs="+", choices=list(STREAMS), help="replay only these streams")
        p.add_argument("--digests", nargs="+", choices=DIGESTS, help="replay only these digests")
        p.add_argument("--budgets", nargs="+", type=int, help="all-layer budgets (default BUDGETS)")
        p.add_argument("--no-eq", action="store_true", help="skip the equal-slots (unlimited layers) rows")
    t = sub.add_parser("tables")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--e2e", action="store_true")
    g = sub.add_parser("agree")
    g.add_argument("traces", type=Path)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--procs", type=int, default=14)
    s = sub.add_parser("summarize")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--e2e", action="store_true")
    args = ap.parse_args()
    dict(compare=cmd_compare, replay=cmd_replay, audit=cmd_audit, summarize=cmd_summarize, agree=cmd_agree, tables=cmd_tables)[args.cmd](args)


if __name__ == "__main__":
    main()
