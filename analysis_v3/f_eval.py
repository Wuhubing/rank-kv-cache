"""F3 (followup-v3): evaluate the frozen configurations once on the third held-out set.

  python -m analysis_v3.f_eval replay HELDOUT_TRACES --frozen splits/heldout_frozen_v3.json --model 06b \
      --split splits/longctx_v3.json --out results_v3/f3/06b [--procs 16]
  python -m analysis_v3.f_eval criteria --frozen splits/heldout_frozen_v3.json --model 06b --out results_v3/f3/06b

replay checks that the folder holds exactly the split's held-out documents (by rid), then replays, per budget of the
frozen file and per document: LRU, Belady, score EMA (frozen alpha), the P2 candidate (if frozen), uniform EMA with the
frozen offline allocation (per-layer budgets fitted on dev) and the P4 candidate (if frozen; online reallocation or
global pool from analysis_v3.c4_alloc), plus LRU and Belady at the offline allocation (same-allocation anchors).
criteria computes H1-H3 of docs/preregistration_v3.md with a document bootstrap (2000 resamples, seed 0) and writes
criteria.json / criteria.txt.
"""
import argparse
import csv
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import kvpolicy as kp
import kvpolicy_v3  # noqa: F401
from analysis_v3 import c4_alloc as c4


def frozen_for(path, model):
    f = json.loads(Path(path).read_text())
    return f, f["models"][model]


def job(args):
    path, fz = args
    R, S, rid = kp.load(path)
    L = R.shape[1]
    out = []
    for Bs, cfg in fz["budgets"].items():
        B = int(Bs)
        alloc = cfg.get("offline_alloc")
        for l in range(L):
            Rl, Sl = R[:, l], S[:, l]
            items = [("lru", "lru", {}, B), ("belady", "belady", {}, B), ("ema", "ema", dict(alpha=cfg["ema_alpha"]), B)]
            if cfg.get("p2"):
                items.append(("p2", cfg["p2"]["policy"], cfg["p2"]["kw"], B))
            if alloc:
                items += [("alloc_ema", "ema", dict(alpha=cfg["ema_alpha"]), alloc[l]),
                          ("lru_same_alloc", "lru", {}, alloc[l]), ("belady_same_alloc", "belady", {}, alloc[l])]
            for name, pol, kw, b in items:
                c = kp.simulate(Rl, Sl, b, pol, **kw)
                out.append((rid, l, name, B, c["accesses"], c["misses"], c["compulsory"]))
        if cfg.get("p4"):
            p4 = cfg["p4"]
            if p4["kind"] == "online":
                m, _ = c4.online_run(R, S, B, cfg["ema_alpha"], **p4["kw"])
            else:
                m, _ = c4.pool_run(R, S, B * L, "ema", **p4["kw"])
            out.append((rid, -1, "p4", B, 0, m, 0))
    return out


def cmd_replay(a):
    f, fz = frozen_for(a.frozen, a.model)
    split = json.loads(Path(a.split).read_text())
    want = sorted(p["rid"] for p in split["prompts"] if p["split"] == "heldout")
    paths = sorted(str(p) for p in a.traces.glob("*.npz"))
    have = sorted(str(np.load(p)["rid"]) for p in paths)
    if have != want:
        raise SystemExit(f"held-out folder does not match the split: {len(have)} traces, missing "
                         f"{sorted(set(want) - set(have))[:3]}, extra {sorted(set(have) - set(want))[:3]}")
    a.out.mkdir(parents=True, exist_ok=True)
    with Pool(a.procs) as pool, open(a.out / "misses.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["doc", "layer", "config", "B", "accesses", "misses", "compulsory"])
        for rows in pool.imap_unordered(job, [(p, fz) for p in paths]):
            w.writerows(rows)
    print(f"replayed {len(paths)} held-out traces")


def g_of(l, p, o):
    return (l - p) / (l - o) if l != o else float("nan")


def cmd_criteria(a):
    f, fz = frozen_for(a.frozen, a.model)
    M = defaultdict(lambda: defaultdict(int))
    with open(a.out / "misses.csv") as fh:
        for r in csv.DictReader(fh):
            M[r["config"], int(r["B"])][r["doc"]] += int(r["misses"])
    docs = sorted(M["lru", int(f["primary_B"])])
    idx = np.random.default_rng(0).integers(0, len(docs), size=(2000, len(docs)))
    res = {"model": a.model, "n_docs": len(docs), "budgets": {}}
    lines = [f"{a.model}: {len(docs)} held-out documents"]
    for Bs in fz["budgets"]:
        B = int(Bs)
        per = {k: np.array([M[k, B][d] for d in docs], float) for (k, b) in list(M) if b == B}
        lu, bu = per["lru"], per["belady"]

        def G(v, i=slice(None)):
            return g_of(lu[i].sum(), v[i].sum(), bu[i].sum())

        def ci(fn):
            xs = [fn(i) for i in idx]
            return float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))
        rb = {}
        for k in [k for k in ("ema", "p2", "alloc_ema", "p4") if k in per]:
            rb[k] = dict(G=G(per[k]), ci=ci(lambda i, v=per[k]: G(v, i)), wins=int((per[k] < lu).sum()))
        if "p2" in per:
            d = per["p2"]
            rb["p2_minus_ema"] = dict(dG=G(d) - G(per["ema"]), ci=ci(lambda i: G(d, i) - G(per["ema"], i)),
                                      docs_p2_fewer=int((d < per["ema"]).sum()))
        if "p4" in per:
            d = per["p4"]
            rb["p4_minus_uniform_ema"] = dict(dG=G(d) - G(per["ema"]), ci=ci(lambda i: G(d, i) - G(per["ema"], i)),
                                              docs_p4_fewer=int((d < per["ema"]).sum()))
            if "alloc_ema" in per and per["ema"].sum() != per["alloc_ema"].sum():
                rb["p4_share_of_offline_gain"] = float((per["ema"].sum() - d.sum()) /
                                                       (per["ema"].sum() - per["alloc_ema"].sum()))
        if "alloc_ema" in per:
            la, ba = per["lru_same_alloc"], per["belady_same_alloc"]
            rb["alloc_ema_within_alloc_G"] = g_of(la.sum(), per["alloc_ema"].sum(), ba.sum())
        res["budgets"][B] = rb
        lines.append(f"B={B}: " + json.dumps(rb))
    P = int(f["primary_B"])
    rp = res["budgets"][P]
    h1 = rp["ema"]["ci"][0] > 0.20 and rp["ema"]["wins"] >= 18 * len(docs) / 21
    res["H1_pass"] = bool(h1)
    if "p2_minus_ema" in rp:
        res["H2_pass"] = bool(rp["p2_minus_ema"]["ci"][0] > 0)
    if "p4_minus_uniform_ema" in rp:
        res["H3_pass"] = bool(rp["p4_minus_uniform_ema"]["ci"][0] > 0)
    lines.append(f"H1 (EMA CI lower > 0.20 and >= 18/21 wins at B={P}): {'PASS' if h1 else 'FAIL'}")
    for h in ("H2", "H3"):
        if f"{h}_pass" in res:
            lines.append(f"{h}: {'PASS' if res[h + '_pass'] else 'FAIL'}")
    (a.out / "criteria.json").write_text(json.dumps(res, indent=1))
    (a.out / "criteria.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["replay", "criteria"])
    ap.add_argument("traces", type=Path, nargs="?")
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    dict(replay=cmd_replay, criteria=cmd_criteria)[a.cmd](a)


if __name__ == "__main__":
    main()
