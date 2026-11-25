"""Instrumented replays for B3 (error classes), B5 (time structure), B6 (spatial), B7 (staleness), D4 (frequency) and
D5 (how often pinned pages are selected). Dev only.

  python -m analysis_v3.instrument TRACES --out results_v3/inst/<model> [--B 96] [--procs 16]

Per document and layer (all at budget B, score EMA z with alpha 0.5 unless stated):
  phase.csv    per-step misses of LRU, Belady, current score, EMA, summed per phase: quarter of the decode (q1, q2-3, q4) and
               thinking / answer (split at the step that emits </think>, token 151668; from the trace manifest)
  errors.csv   every EMA eviction compared with Belady's choice at the same step: a victim is an error if its next use is
               sooner than the n-th farthest next use among the candidates (n = victims at that step). Counts by
               residency, all-page score percentile, return interval, phase (thinking / answer) for errors and all victims
  spatial.csv  LRU repeat misses: share whose page id is within +-1 / +-2 of a page selected at the previous step, against the
               same share for a random unselected eligible page; and AUC of the neighbours' max score for "re-selected
               within 32 steps" among LRU eviction candidates
  stale.csv    selected-only EMA (scoped_ema M=0) vs full EMA: at every eviction, steps since each candidate's value was last
               updated, and the Spearman correlation between the candidates' stored values and their current z
  freq.csv     LRU candidates: variance of log(1 + in-cache selection count) per decision, share of decisions where all
               candidates have the same count, and the counts' range
  pins.csv     share of steps at which page i (i = 0..3) and the j-th most recent completed page (j = 1..4) is selected
"""
import argparse
import bisect
import csv
import json
import math
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import kvpolicy as kp
import kvpolicy_v3 as k3

THINK_END = 151668
H = 32
RES_BINS = (0, 1, 4, 16, 64, 256, 10 ** 9)
PCT_BINS = (0, 0.5, 0.8, 0.9, 0.95, 1.01)
RET_BINS = (1, 2, 4, 8, 16, 32, 64, 256, 10 ** 9)


def think_end_step(path, T):
    man = Path(path).with_suffix(".json")
    if not man.exists():
        return T
    gen = json.loads(man.read_text()).get("generated", [])
    if THINK_END in gen:
        return max(0, gen.index(THINK_END) - 1)       # decode step t emits generated[t + 1]
    return T


def bin_of(x, bins):
    return bisect.bisect_right(bins, x) - 1


def uses_table(R):
    u = defaultdict(list)
    for t, row in enumerate(R):
        for p in row:
            if p >= 0:
                u[int(p)].append(t)
    return u


def next_use(u, p, t):
    lst = u.get(p, [])
    i = bisect.bisect_right(lst, t)
    return lst[i] if i < len(lst) else math.inf


def job(args):
    path, B = args
    R, S, rid = kp.load(path)
    T, L, K = R.shape
    te = think_end_step(path, T)
    q1, q3 = T // 4, (3 * T) // 4
    phase_rows, err_rows, sp_rows, st_rows, fq_rows = [], [], [], [], []
    pins = np.zeros((L, 8))
    for l in range(L):
        Rl, Sl = R[:, l], S[:, l]
        u = uses_table(Rl)
        # ---- B5: per-step misses
        per = {}
        for name, pol, kw in (("lru", "lru", {}), ("belady", "belady", {}), ("score", "score", {}),
                              ("ema", "ema", dict(alpha=0.5))):
            per[name] = k3.simulate_steps(Rl, Sl, B, pol, **kw)[1]
        steps = np.arange(T)
        masks = {"q1": steps < q1, "q2-3": (steps >= q1) & (steps < q3), "q4": steps >= q3,
                 "thinking": steps <= te, "answer": steps > te}
        for ph, m in masks.items():
            phase_rows.append([rid, l, ph, int(m.sum())] + [int(per[n][m].sum()) for n in ("lru", "belady", "score", "ema")])
        # ---- B3: EMA errors against Belady, with candidate features
        cnt = defaultdict(lambda: np.zeros(2, np.int64))

        def cb(t, victims, pol):
            cands = list(pol.ema)  # residents incl. victims (still in ema at callback time)
            sel = set(pol.ranked)
            cands = [p for p in cands if p not in sel]
            nu = {p: next_use(u, p, t) for p in cands}
            far = sorted((nu[p] for p in cands), reverse=True)
            thr = far[len(victims) - 1]
            table = kp._allrank_table(pol.srow)
            for v in victims:
                err = nu[v] < thr
                res_age = t - pol.adm_t[v]
                pct = kp._allrank(table, pol.srow[v])
                ret = nu[v] - t
                for key in (("res", bin_of(res_age, RES_BINS)), ("pct", bin_of(pct, PCT_BINS)),
                            ("ret", bin_of(ret, RET_BINS) if math.isfinite(ret) else len(RET_BINS) - 1),
                            ("phase", int(t > te)), ("all", 0)):
                    cnt[key] += (1, int(err))

        class RecEMA(kp.ScoreEMA):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.adm_t = {}

            def admit(self, hits, misses):
                super().admit(hits, misses)
                for p in misses:
                    self.adm_t[p] = self.t

            def dropped(self, p):
                super().dropped(p)
                self.adm_t.pop(p, None)

        kp.POLICIES["_rec_ema_b3"] = RecEMA
        kp.simulate(Rl, Sl, B, "_rec_ema_b3", on_evict=cb, alpha=0.5)
        for (kind, b), (n, e) in sorted(cnt.items()):
            err_rows.append([rid, l, kind, b, int(n), int(e)])
        # ---- B6: spatial structure of LRU repeat misses
        prev_sel, seen = set(), set()
        near = np.zeros(4)   # [repeat misses, within1, within2, random-within1, random-within2] handled below
        rnd = np.zeros(3)
        rng = np.random.default_rng(l)
        res = set()
        lru = kp.LRU(K)
        for t in range(T):
            row = tuple(int(p) for p in Rl[t] if p >= 0)
            lru.begin(t, row, None)
            sel = set(row)
            hits = sel & res
            misses = sel - hits
            n = len(res) + len(misses) - B
            if n > 0:
                for v in lru.evict(res - sel, n):
                    res.discard(v)
                    lru.dropped(v)
            res |= sel
            lru.admit(hits, misses)
            if t > 0:
                for p in misses & seen:
                    near[0] += 1
                    near[1] += any(abs(p - q) <= 1 for q in prev_sel)
                    near[2] += any(abs(p - q) <= 2 for q in prev_sel)
                ne = int(np.isfinite(Sl[t]).sum())
                cand = [p for p in rng.integers(0, ne, 8).tolist() if p not in sel]
                for p in cand:
                    rnd[0] += 1
                    rnd[1] += any(abs(p - q) <= 1 for q in prev_sel)
                    rnd[2] += any(abs(p - q) <= 2 for q in prev_sel)
            seen |= sel
            prev_sel = sel
        # neighbour-score AUC among LRU candidates (every 4th step)
        sc_n, lab = [], []
        res2 = set()
        lru2 = kp.LRU(K)
        for t in range(T):
            row = tuple(int(p) for p in Rl[t] if p >= 0)
            lru2.begin(t, row, None)
            sel = set(row)
            hits = sel & res2
            misses = sel - hits
            n = len(res2) + len(misses) - B
            if n > 0:
                if t % 4 == 0:
                    s = Sl[t]
                    ne = int(np.isfinite(s).sum())
                    for p in res2 - sel:
                        nb = [s[q] for q in (p - 1, p + 1) if 0 <= q < ne]
                        sc_n.append(max(nb) if nb else -np.inf)
                        lab.append(next_use(u, p, t) - t <= H and t + H < T)
                for v in lru2.evict(res2 - sel, n):
                    res2.discard(v)
                    lru2.dropped(v)
            res2 |= sel
            lru2.admit(hits, misses)
        from gap_analysis import auc as tie_auc
        a = tie_auc(np.array(sc_n), np.array(lab, bool)) if lab and 0 < sum(lab) < len(lab) else None
        sp_rows.append([rid, l, int(near[0]), int(near[1]), int(near[2]), int(rnd[0]), int(rnd[1]), int(rnd[2]),
                        float("nan") if a is None else a, len(lab)])
        # ---- B7: staleness of the selected-only EMA
        for M in (0, None):
            stale, rhos = [], []

            def cb2(t, victims, pol):
                cands = [p for p in pol.ema if p not in set(pol.ranked)]
                stale.extend(t - pol.upd[p] for p in cands)
                if len(cands) > 3 and t % 4 == 0:
                    a_ = np.array([pol.ema[p] for p in cands])
                    b_ = np.array([kp._z(pol.srow, pol.ranked, p) for p in cands])
                    ra, rb = np.argsort(np.argsort(a_)), np.argsort(np.argsort(b_))
                    if ra.std() > 0 and rb.std() > 0:
                        rhos.append(np.corrcoef(ra, rb)[0, 1])
            kp.simulate(Rl, Sl, B, "scoped_ema", on_evict=cb2, alpha=0.5, M=M)
            st = np.array(stale)
            st_rows.append([rid, l, "sel_only" if M == 0 else "full", float(np.mean(st)), float(np.median(st)),
                            float(np.percentile(st, 90)), float(np.mean(rhos)) if rhos else float("nan")])
        # ---- D4: frequency among LRU candidates
        varis, same, rngs = [], 0, []
        res3, cnt3 = set(), defaultdict(int)
        lru3 = kp.LRU(K)
        nd = 0
        for t in range(T):
            row = tuple(int(p) for p in Rl[t] if p >= 0)
            lru3.begin(t, row, None)
            sel = set(row)
            hits = sel & res3
            misses = sel - hits
            n = len(res3) + len(misses) - B
            if n > 0:
                c = np.array([math.log1p(cnt3[p]) for p in res3 - sel])
                varis.append(c.var())
                same += int(c.min() == c.max())
                rngs.append(c.max() - c.min())
                nd += 1
                for v in lru3.evict(res3 - sel, n):
                    res3.discard(v)
                    lru3.dropped(v)
                    cnt3.pop(v, None)
            res3 |= sel
            lru3.admit(hits, misses)
            for p in sel:
                cnt3[p] += 1
        fq_rows.append([rid, l, nd, float(np.mean(varis)) if varis else float("nan"), same,
                        float(np.mean(rngs)) if rngs else float("nan")])
        # ---- D5: selection share of the first pages and the most recent completed pages
        hp = kp_completed(Sl)
        for t in range(T):
            sel = set(int(p) for p in Rl[t] if p >= 0)
            for i in range(4):
                pins[l, i] += i in sel
            for j in range(1, 5):
                pins[l, 3 + j] += (hp[t] - j) in sel
        pins[l] /= T
    pin_rows = [[rid, l] + pins[l].tolist() for l in range(L)]
    return dict(phase=phase_rows, errors=err_rows, spatial=sp_rows, stale=st_rows, freq=fq_rows, pins=pin_rows)


def kp_completed(Sl):
    return np.isfinite(Sl).sum(1)


HEADERS = dict(phase=["doc", "layer", "phase", "steps", "lru", "belady", "score", "ema"],
               errors=["doc", "layer", "group", "bin", "victims", "errors"],
               spatial=["doc", "layer", "repeat_misses", "within1_prev_sel", "within2_prev_sel", "random_pages",
                        "random_within1", "random_within2", "neighbour_score_auc", "n_candidates"],
               stale=["doc", "layer", "variant", "mean_staleness", "median_staleness", "p90_staleness",
                      "spearman_value_vs_current_z"],
               freq=["doc", "layer", "decisions", "mean_var_log_count", "decisions_all_equal", "mean_range_log_count"],
               pins=["doc", "layer", "page0", "page1", "page2", "page3", "recent1", "recent2", "recent3", "recent4"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--B", type=int, default=96)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    paths = sorted(str(p) for p in a.traces.glob("*.npz"))
    a.out.mkdir(parents=True, exist_ok=True)
    files = {k: open(a.out / f"{k}.csv", "w", newline="") for k in HEADERS}
    writers = {k: csv.writer(f) for k, f in files.items()}
    for k, w in writers.items():
        w.writerow(HEADERS[k])
    with Pool(a.procs) as pool:
        for res in pool.imap_unordered(job, [(p, a.B) for p in paths]):
            for k, rows in res.items():
                writers[k].writerows(rows)
    for f in files.values():
        f.close()
    print(f"{len(paths)} traces -> {a.out}")


if __name__ == "__main__":
    main()
