"""Summaries of part E (followup-v3) and the D2 step-time model.

  python -m analysis_v3.e_summarize DIR [DIR ...] --out results_v3/e/summary

Misses per step everywhere = mean over the timed steps (step_misses[warmup:]) within a run, then mean over rounds.
throughput.csv: other_procs_seen = runs with another process on the card (before or after); clean = none in the row;
max_load = max of the 1-minute load before and after each run; ranks_equal / tokens_equal = n/a when not checked
(nofetch); peak_decode_gib / host_tier_gib = max over rounds of the run's decode peak and host-tier size (GiB).

Inputs: the batch<R>.json files written by analysis_v3.e_bench (one DIR per configuration: model, page size, K, delay).
Per (configuration, R, delay):
  * per policy: median step (ms), mean throughput (tokens/s), misses/step; ratio to LRU's throughput, paired by round,
    with a bootstrap 95 % CI over rounds (2000 resamples, seed 0);
  * E3: per-step time of "allmiss" (K fetched pages per request and layer) minus "nofetch" (no transfer) = the most a
    step can spend on transfer; per missed page = that difference / (K * R * L); upper bound of any policy's gain =
    (its saved misses per step) x (time per missed page);
  * E4: step-time quantiles for steps with / without a page completion and by miss-count tercile;
  * D2: least squares step_ms = a + b * misses_in_step + c * page_step over all LRU/EMA/score/Belady steps of the
    configuration (after warm-up): b is the marginal cost of a missed page, page_bytes / b the effective bandwidth,
    and b * misses / step the transfer share of LRU's step.
Speedup curve (E5): x = LRU's transfer share (D2), y = throughput ratio to LRU per policy, plus the D2 prediction
(a + b m_LRU) / (a + b m_pol). Rows from a slow-tier delay are flagged simulated=True.
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

POLS = ("lru", "score", "ema", "alloc_ema", "belady")


def _batch_files(d):
    """batch<R>.json or batch<R>.json.gz (the committed raw files are gzipped); one file per R."""
    files = {}
    for f in sorted(Path(d).glob("batch*.json*")):
        stem = f.name[:-3] if f.name.endswith(".gz") else f.name
        if stem.endswith(".json"):
            files.setdefault(stem, f)
    return [files[k] for k in sorted(files)]


def _read_json(f):
    return json.load(gzip.open(f, "rt")) if f.name.endswith(".gz") else json.loads(f.read_text())


def timed_misses_per_step(r, warm):
    """Misses per step over the timed steps only (warm-up excluded from numerator and denominator). R11-1: the
    previous code divided the total over all steps (warm-up included) by the number of timed steps."""
    return float(np.mean(r["step_misses"][warm:]))


def mean_timed_misses(runs, warm):
    """Mean over rounds of the per-run timed mean."""
    return float(np.mean([timed_misses_per_step(r, warm) for r in runs]))


def _check(vals):
    """True / False if every run was checked; 'n/a' if no run was checked (None), e.g. nofetch."""
    if all(v is None for v in vals):
        return "n/a"
    return all(v is True for v in vals)


def load_runs(d):
    meta = json.loads((Path(d) / "meta.json").read_text())
    out = []
    for f in _batch_files(d):
        b = _read_json(f)
        for r in b["runs"]:
            r["page_bytes"] = b["page_bytes"]
            r["budget"] = b["budget"]
            out.append(r)
    return meta, out


def boot_ratio(x, y, reps=2000, seed=0):
    """x, y: per-round throughput of policy and LRU (paired); ratio of means and its bootstrap CI over rounds."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    rs = x[idx].mean(1) / y[idx].mean(1)
    return float(x.mean() / y.mean()), float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows, tails, d2, curve = [], [], [], []
    for d in a.dirs:
        meta, runs = load_runs(d)
        args = meta["args"]
        cfg = f"{Path(d).name}"
        warm = int(args["warmup"])
        groups = defaultdict(list)
        for r in runs:
            groups[r["R"], r["delay_ns_per_byte"]].append(r)
        for (R, delay), rs in sorted(groups.items()):
            L = None
            by = defaultdict(dict)
            for r in rs:
                by[r["policy"]][r["round"]] = r
            rounds = sorted(by["lru"])
            # E3 upper bound
            ub = {}
            if "allmiss" in by and "nofetch" in by:
                am = np.median([np.median(by["allmiss"][k]["lat_ms"][warm:]) for k in by["allmiss"]])
                nf = np.median([np.median(by["nofetch"][k]["lat_ms"][warm:]) for k in by["nofetch"]])
                am_m = mean_timed_misses(by["allmiss"].values(), warm)
                ub = dict(allmiss_ms=am, nofetch_ms=nf, transfer_max_ms=am - nf,
                          ms_per_missed_page=(am - nf) / am_m if am_m else float("nan"))
            # D2 fit over the policy runs
            X, Y = [], []
            for p in POLS:
                for k, r in by.get(p, {}).items():
                    lat = np.array(r["lat_ms"][warm:])
                    m = np.array(r["step_misses"][warm:])
                    pg = np.array(r["page_step"][warm:], float)
                    X.append(np.column_stack([np.ones(len(lat)), m, pg]))
                    Y.append(lat)
            if X:
                X, Y = np.vstack(X), np.concatenate(Y)
                coef = np.linalg.lstsq(X, Y, rcond=None)[0]
                page_bytes = rs[0]["page_bytes"]
                lru_m = mean_timed_misses([by["lru"][k] for k in rounds], warm)
                lru_step = np.mean([np.mean(by["lru"][k]["lat_ms"][warm:]) for k in rounds])
                share = coef[1] * lru_m / lru_step
                d2.append(dict(config=cfg, R=R, delay_ns_per_byte=delay, a_ms=coef[0], b_ms_per_miss=coef[1],
                               c_ms_page_step=coef[2], eff_GBps=page_bytes / (coef[1] * 1e-3) / 1e9 if coef[1] > 0 else
                               float("inf"), lru_misses_per_step=lru_m, lru_step_ms=lru_step,
                               transfer_share_lru=share, simulated=delay > 0, **ub))
            for p in POLS + ("allmiss", "nofetch"):
                if p not in by:
                    continue
                ks = [k for k in rounds if k in by[p]]
                tp = [by[p][k]["tokens_per_s"] for k in ks]
                tl = [by["lru"][k]["tokens_per_s"] for k in ks]
                ratio, lo, hi = boot_ratio(tp, tl)
                med = float(np.median([np.median(by[p][k]["lat_ms"][warm:]) for k in ks]))
                mps = mean_timed_misses([by[p][k] for k in ks], warm)
                # R11-2: load = max of before/after; other_procs_seen = number of runs with another process on the
                # card before or after the run; clean = no such run in this row
                loads = [max(by[p][k]["before"]["loadavg1"], by[p][k]["after"]["loadavg1"]) for k in ks]
                others = sum(bool(by[p][k]["before"]["other_pids"] or by[p][k]["after"]["other_pids"]) for k in ks)
                peak = [by[p][k].get("peak_decode") for k in ks]
                host = [by[p][k].get("host_bytes") for k in ks]
                row = dict(config=cfg, R=R, delay_ns_per_byte=delay, simulated=delay > 0, policy=p, rounds=len(ks),
                           median_step_ms=med, tokens_per_s=float(np.mean(tp)), ratio_vs_lru=ratio, ratio_lo=lo,
                           ratio_hi=hi, misses_per_step=mps, max_load=max(loads), other_procs_seen=others,
                           clean=others == 0,
                           ranks_equal=_check([by[p][k]["ranks_equal_prep"] for k in ks]),
                           tokens_equal=_check([by[p][k]["tokens_equal_prep"] for k in ks]),
                           peak_decode_gib=max(peak) / 2**30 if None not in peak else "",
                           host_tier_gib=max(host) / 2**30 if None not in host else "")
                if ub and p in POLS:
                    saved = mean_timed_misses([by["lru"][k] for k in ks], warm) - mps
                    row["upper_bound_gain_ms"] = saved * ub["ms_per_missed_page"]
                rows.append(row)
                if p in POLS and d2:
                    dd = d2[-1]
                    pred = (dd["a_ms"] + dd["b_ms_per_miss"] * dd["lru_misses_per_step"]) / (
                        dd["a_ms"] + dd["b_ms_per_miss"] * mps)
                    curve.append(dict(config=cfg, R=R, delay_ns_per_byte=delay, simulated=delay > 0, policy=p,
                                      transfer_share_lru=dd["transfer_share_lru"], ratio=ratio, lo=lo, hi=hi,
                                      d2_predicted_ratio=pred))
                # E4 tails
                for k in ks:
                    lat = np.array(by[p][k]["lat_ms"][warm:])
                    pg = np.array(by[p][k]["page_step"][warm:], bool)
                    m = np.array(by[p][k]["step_misses"][warm:])
                    for tag, mask in (("page_step", pg), ("no_page_step", ~pg),
                                      ("miss_low", m <= np.percentile(m, 33)), ("miss_high", m >= np.percentile(m, 67))):
                        if mask.any():
                            q = np.percentile(lat[mask], [50, 90, 99])
                            tails.append(dict(config=cfg, R=R, delay_ns_per_byte=delay, policy=p, round=k, tag=tag,
                                              n=int(mask.sum()), p50=q[0], p90=q[1], p99=q[2], max=float(lat[mask].max())))
    a.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("throughput.csv", rows), ("tails.csv", tails), ("d2_model.csv", d2), ("speedup_curve.csv", curve)):
        if data:
            keys = sorted({k for r in data for k in r}, key=lambda k: list(data[0]).index(k) if k in data[0] else 99)
            with open(a.out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(data)
    for r in rows:
        print(f"{r['config']:<16} R={r['R']:>2} d={r['delay_ns_per_byte']:<5} {r['policy']:<9} med {r['median_step_ms']:7.2f} ms "
              f"{r['tokens_per_s']:8.1f} tok/s x{r['ratio_vs_lru']:.3f} [{r['ratio_lo']:.3f},{r['ratio_hi']:.3f}] "
              f"miss/step {r['misses_per_step']:8.1f} load<= {r['max_load']:.1f}")
    for r in d2:
        print(f"D2 {r['config']:<16} R={r['R']:>2} d={r['delay_ns_per_byte']}: a {r['a_ms']:.2f} ms, b {r['b_ms_per_miss']*1e3:.2f} us/miss "
              f"({r['eff_GBps']:.1f} GB/s eff.), LRU transfer share {100*r['transfer_share_lru']:.1f} %")


if __name__ == "__main__":
    main()
