"""F8: warm-up exclusion and raw vs steady-state metrics (torch-free helpers of gpu_bench.py)."""
import numpy as np
import pytest

import bench_metrics as bm


def test_warmup_is_applied_to_latency_samples():
    m = bm.decode_metrics([10.0, 8.0, 1.0, 3.0], warmup=2)
    assert m["lat_raw"]["n"] == 4 and m["lat_steady"]["n"] == 2
    assert m["lat_raw"]["mean"] == 5.5 and m["lat_steady"]["mean"] == 2.0
    assert m["lat_steady"]["median"] == 2.0


def test_zero_warmup_makes_steady_equal_raw():
    m = bm.decode_metrics([1.0, 2.0, 3.0], warmup=0)
    assert m["lat_raw"] == m["lat_steady"]


@pytest.mark.parametrize("steps,warmup", [(4, 4), (4, 5), (4, -1), (0, 0), (4, 1.5)])
def test_invalid_warmup_is_rejected(steps, warmup):
    with pytest.raises(ValueError):
        bm.check_warmup(steps, warmup)


def test_warmup_mask():
    assert bm.warmup_mask(4, 2).tolist() == [True, True, False, False]


def test_step_misses_from_cumulative_counters_and_steady_sum():
    cum = np.array([[5, 7], [6, 9], [6, 12], [8, 12]])
    sm = bm.step_misses(cum)
    assert sm.tolist() == [[5, 7], [1, 2], [0, 3], [2, 0]]
    m = bm.decode_metrics([1.0] * 4, warmup=1, step_miss=sm)
    assert m["misses_raw"] == [8, 12] and m["misses_steady"] == [3, 5]
    assert m["misses_per_step_raw"] == 5.0 and m["misses_per_step_steady"] == 8 / 3


def test_decreasing_cumulative_counter_is_rejected():
    with pytest.raises(ValueError):
        bm.step_misses(np.array([[3], [2]]))


def test_breakdown_steady_mean_excludes_warmup():
    m = bm.decode_metrics([1.0, 1.0, 1.0], warmup=1,
                          breakdown=np.array([[9.0, 9.0], [1.0, 2.0], [3.0, 4.0]]))
    assert m["breakdown_steady_mean"] == [2.0, 3.0] and m["breakdown_raw_mean"] == [13 / 3, 5.0]


def test_shape_mismatches_are_rejected():
    with pytest.raises(ValueError):
        bm.decode_metrics([1.0, 2.0], 0, step_miss=np.zeros((3, 2)))
    with pytest.raises(ValueError):
        bm.decode_metrics([1.0, 2.0], 0, breakdown=np.zeros((2,)))
