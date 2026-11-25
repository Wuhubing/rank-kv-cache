from collections import Counter

import numpy as np
import pytest

from analysis import observations
from replay import replay_stream
from validation.rank_decisions import HORIZON, decision_replay, selected_reuse


@pytest.mark.parametrize("following,outcome", [
    ((2, 7, 0, 6), "rank_first"), ((3, 7, 0, 6), "base_first"),
    ((2, 3, 0, 6), "same_step"), ((9, 7, 0, 6), "neither"),
])
def test_first_exclusive_reuse_and_censoring(following, outcome):
    # After the first three steps both policies have identical state.
    # At step 3 rank keeps page 2, while R+F keeps page 3.
    prefix = [(2, 9, 1, 4), (1, 7, 10, 9), (10, 6, 3, 1), (7, 0, 6, 9)]
    steps = prefix + [following] * HORIZON
    events = decision_replay(steps, 2, budget=7, width=4)
    assert events["complete_changed"] == events[outcome] == 1
    assert decision_replay(steps[:-1], 2, budget=7, width=4)["complete_changed"] == 0
    for key, delta in (("base_repeats", 0), ("rank_repeats", 4)):
        assert events[key] == replay_stream(steps, 7, "score", beta=2, delta=delta)["repeat_misses"]


def test_frequency_counts_match_reference():
    steps = [tuple((i + t*3) % 83 for i in range(64)) for t in range(23)]
    actual = np.zeros((9, 4), dtype=int)
    selected_reuse(steps, actual)
    counts = Counter()
    for row in observations({("dev", 0):steps}, horizons=(8,)):
        q = row["quartile"]
        if q in ("top", "bottom"):
            index = 0 if q == "top" else 2
            counts[row["frequency"], index] += 1
            counts[row["frequency"], index+1] += row["reuse_8"]
    assert all(n == counts[f,j] for (f,j), n in np.ndenumerate(actual))
