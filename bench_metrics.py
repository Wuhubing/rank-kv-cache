"""Torch-free helpers for gpu_bench.py: warm-up handling and raw vs steady-state decode metrics.

Definitions used by gpu_bench.run and its summaries:
- raw: every decode step after the prefill (steps 0 .. steps-1);
- steady: steps warmup .. steps-1, i.e. the first `warmup` decode steps are excluded;
- per-step misses are differences of cumulative per-layer miss counters, so the steady-state miss
  count is the sum over the steady steps only (not the raw total).
"""
import numpy as np


def check_warmup(steps, warmup):
    """0 <= warmup < steps, both integers; raises ValueError otherwise."""
    if int(steps) != steps or int(warmup) != warmup or steps < 1 or not 0 <= warmup < steps:
        raise ValueError(f"need integers 0 <= warmup < steps, got warmup={warmup}, steps={steps}")
    return int(steps), int(warmup)


def warmup_mask(steps, warmup):
    """Boolean [steps] array, True for warm-up steps."""
    steps, warmup = check_warmup(steps, warmup)
    return np.arange(steps) < warmup


def step_misses(cumulative):
    """Per-step, per-layer misses [steps, L] from cumulative counters sampled after each step."""
    c = np.asarray(cumulative)
    if c.ndim != 2:
        raise ValueError(f"cumulative misses must be [steps, L], got shape {c.shape}")
    d = np.diff(np.vstack((np.zeros((1, c.shape[1]), c.dtype), c)), axis=0)
    if (d < 0).any():
        raise ValueError("cumulative miss counters decreased")
    return d


def _lat_stats(x):
    x = np.asarray(x, float)
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                p90=float(np.percentile(x, 90)), p95=float(np.percentile(x, 95)))


def decode_metrics(lat, warmup, step_miss=None, breakdown=None):
    """Raw and steady-state summaries of one run. lat: per-step seconds [steps]; step_miss:
    optional [steps, L]; breakdown: optional [steps, phases] (ms, CUDA events)."""
    lat = np.asarray(lat, float)
    if lat.ndim != 1:
        raise ValueError(f"lat must be one-dimensional, got shape {lat.shape}")
    steps, warmup = check_warmup(lat.size, warmup)
    out = dict(steps=steps, warmup=warmup, lat_raw=_lat_stats(lat), lat_steady=_lat_stats(lat[warmup:]))
    if step_miss is not None:
        m = np.asarray(step_miss)
        if m.ndim != 2 or m.shape[0] != steps:
            raise ValueError(f"step_miss must be [{steps}, L], got shape {m.shape}")
        out.update(misses_raw=m.sum(0).tolist(), misses_steady=m[warmup:].sum(0).tolist(),
                   misses_per_step_raw=float(m.sum() / steps),
                   misses_per_step_steady=float(m[warmup:].sum() / (steps - warmup)))
    if breakdown is not None:
        b = np.asarray(breakdown, float)
        if b.ndim != 2 or b.shape[0] != steps:
            raise ValueError(f"breakdown must be [{steps}, phases], got shape {b.shape}")
        out.update(breakdown_raw_mean=b.mean(0).tolist(), breakdown_steady_mean=b[warmup:].mean(0).tolist())
    return out
