"""R11-7 (review of PR #11): the policies' own GPU time and the net-gain account, plus GPU memory by phase.

  python -m analysis_v3.r11_netgain --pair NAME THROUGHPUT_RAW_DIR R TIMING_RAW_DIR [--pair ...] --out DIR

Per pair (a configuration measured twice: throughput runs without events, and separate CUDA-event timing runs, E6 style):
  * saved transfer (ms/step) = (LRU misses/step - policy misses/step) x ms per missed page, both from the throughput runs
    (timed steps only; per-page cost = (all-miss - no-fetch median step) / all-miss misses per step, as e_summarize);
  * policy overhead (ms/step) = policy-kernel phase (events e2 -> e3: fused policy kernel + slot update, summed over layers)
    of the policy minus LRU's, timing runs, mean over timed steps then rounds; bootstrap CI over rounds (independent);
  * predicted net gain = saved transfer - policy overhead;
  * measured step difference = LRU mean step - policy mean step (timed steps), paired by round, bootstrap 95 % CI over
    rounds (2000 resamples, seed 0); also the median-step difference;
  * consistent = predicted net gain inside the measured CI; otherwise gap = predicted - nearest CI end.
Also: phase times (append, selection, policy, index, attention incl. the wait for the fetch, gather on the side stream) and
GPU memory by phase (prefill forward peak, migration peak, allocated at decode start, decode warm-up peak, steady-state peak;
GiB, max over rounds) from the timing runs.
TIMING_RAW_DIR "-" (no timing runs): overhead and memory are left empty and implied_overhead_ms = saved transfer - measured
step difference is reported (what the policy's own cost and any other effect would have to explain).
Outputs: netgain.csv, phases.csv, memory.csv in --out.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from analysis_v3.e_summarize import load_runs, mean_timed_misses

POLS = ("lru", "score", "ema", "alloc_ema", "belady")
PHASES = ("append", "selection", "policy_kernels", "index", "attention_wait", "spin", "gather")
MEM = ("prefill_forward", "migration", "decode_start", "decode_warmup", "decode_steady")


def _by(runs, R):
    by = {}
    for r in runs:
        if r["R"] == R and r["delay_ns_per_byte"] == 0:
            by.setdefault(r["policy"], {})[r["round"]] = r
    return by


def boot_mean(x, reps=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff(a, b, reps=2000, seed=0):
    """mean(a) - mean(b) for two independent samples (rounds of two policies); bootstrap CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = (a[rng.integers(0, len(a), size=(reps, len(a)))].mean(1) - b[rng.integers(0, len(b), size=(reps, len(b)))].mean(1))
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def phase_means(r, warm):
    """Per-step phase times (ms, summed over layers), mean over the timed steps of one run."""
    b = np.asarray(r["breakdown_ms"], float)
    return b[warm:].mean(0)


def pair_rows(name, tput_dir, R, timing_dir):
    meta, runs = load_runs(tput_dir)
    warm = int(meta["args"]["warmup"])
    by = _by(runs, R)
    if str(timing_dir) == "-":
        tby, twarm = {}, 0
    else:
        tmeta, truns = load_runs(timing_dir)
        twarm = int(tmeta["args"]["warmup"])
        tby = _by(truns, R)
    am = np.median([np.median(r["lat_ms"][warm:]) for r in by["allmiss"].values()])
    nf = np.median([np.median(r["lat_ms"][warm:]) for r in by["nofetch"].values()])
    per_page = (am - nf) / mean_timed_misses(by["allmiss"].values(), warm)
    rounds = sorted(by["lru"])
    lru_m = mean_timed_misses([by["lru"][k] for k in rounds], warm)
    lru_pol = [phase_means(r, twarm)[2] for r in tby["lru"].values()] if "lru" in tby else []
    net, phases, mem = [], [], []
    for p in POLS:
        if p in tby:
            for k, r in sorted(tby[p].items()):
                ph = phase_means(r, twarm)
                phases.append(dict(config=name, R=R, policy=p, round=k, **{n: float(v) for n, v in zip(PHASES, ph)},
                                   clean=not (r["before"]["other_pids"] or r["after"]["other_pids"])))
            mp = [r.get("mem_phases") or {} for r in tby[p].values()]
            mem.append(dict(config=name, R=R, policy=p,
                            **{n: (max(m[n] for m in mp) / 2 ** 30 if all(m.get(n) is not None for m in mp) else "")
                               for n in MEM}))
        if p == "lru" or p not in by:
            continue
        ks = [k for k in rounds if k in by[p]]
        m = mean_timed_misses([by[p][k] for k in ks], warm)
        saved = (lru_m - m) * per_page
        pol = [phase_means(r, twarm)[2] for r in tby[p].values()] if p in tby else []
        ov, ov_lo, ov_hi = boot_diff(pol, lru_pol) if pol and lru_pol else (float("nan"),) * 3
        d = [np.mean(by["lru"][k]["lat_ms"][warm:]) - np.mean(by[p][k]["lat_ms"][warm:]) for k in ks]
        dmed = float(np.mean([np.median(by["lru"][k]["lat_ms"][warm:]) - np.median(by[p][k]["lat_ms"][warm:]) for k in ks]))
        meas, lo, hi = boot_mean(d)
        pred = saved - ov
        eps = 1e-9   # degenerate CIs (identical rounds) compare up to rounding
        gap = (float("nan") if np.isnan(pred) else
               0.0 if lo - eps <= pred <= hi + eps else (pred - hi if pred > hi else pred - lo))
        lru_step = float(np.mean([np.mean(by["lru"][k]["lat_ms"][warm:]) for k in ks]))
        net.append(dict(config=name, R=R, policy=p, lru_misses_per_step=lru_m, misses_per_step=m,
                        ms_per_missed_page=per_page, saved_transfer_ms=saved,
                        policy_ms=float(np.mean(pol)) if pol else float("nan"),
                        lru_policy_ms=float(np.mean(lru_pol)) if lru_pol else float("nan"), overhead_ms=ov, overhead_lo=ov_lo, overhead_hi=ov_hi,
                        predicted_net_ms=pred, measured_ms=meas, measured_lo=lo, measured_hi=hi,
                        measured_median_ms=dmed, implied_overhead_ms=saved - meas, lru_mean_step_ms=lru_step,
                        consistent=gap == 0.0 if not np.isnan(gap) else "", gap_ms=gap,
                        timing_rounds=len(pol), throughput_rounds=len(ks),
                        throughput_clean=all(not (by[q][k]["before"]["other_pids"] or by[q][k]["after"]["other_pids"])
                                             for q in ("lru", p) for k in ks)))
    return net, phases, mem


def write(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=4, action="append", required=True, metavar=("NAME", "TPUT", "R", "TIMING"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    net, phases, mem = [], [], []
    for name, tput, R, timing in a.pair:
        n, p, m = pair_rows(name, Path(tput), int(R), Path(timing))
        net += n
        phases += p
        mem += m
    a.out.mkdir(parents=True, exist_ok=True)
    write(a.out / "netgain.csv", net)
    write(a.out / "phases.csv", phases)
    write(a.out / "memory.csv", mem)
    for r in net:
        print(f"{r['config']:<14} {r['policy']:<9} saved {r['saved_transfer_ms']:6.3f} - overhead {r['overhead_ms']:6.3f} "
              f"= pred {r['predicted_net_ms']:6.3f} ms | measured {r['measured_ms']:6.3f} [{r['measured_lo']:.3f}, "
              f"{r['measured_hi']:.3f}] implied overhead {r['implied_overhead_ms']:.3f} "
              f"{'ok' if r['consistent'] is True else 'gap %.3f' % r['gap_ms']}")


if __name__ == "__main__":
    main()
