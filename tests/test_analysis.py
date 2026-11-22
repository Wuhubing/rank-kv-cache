import random
import unittest
from collections import Counter
from pathlib import Path

from analysis import analyze, observations, quartile, recency_bin, stratified_effect, tally
from replay import iter_replay, read_trace

ROOT = Path(__file__).resolve().parents[1]


def synthetic(seed, keep, rank_by=None, requests=30, steps=40, pages=48, k=8):
    """Each step keeps pages from the previous step with probability keep(rank, freq).

    rank_by=None ranks each selection randomly. Otherwise pages are ranked by
    rank_by(freq), highest first, so rank can be confounded with frequency.
    """
    rng = random.Random(seed)
    streams = {}
    for r in range(requests):
        trace = []
        for t in range(steps):
            def freq(p):
                return sum(p in trace[s] for s in range(max(0, t - 8), t))
            chosen = []
            if trace:
                previous = trace[-1]
                prior = len(trace) - 1
                for rank, page in enumerate(previous, 1):
                    c = sum(page in trace[s] for s in range(max(0, prior - 8), prior))
                    if rng.random() < keep(rank, c):
                        chosen.append(page)
            pool = [p for p in range(pages) if p not in chosen]
            chosen += rng.sample(pool, k - len(chosen))
            if rank_by is None:
                rng.shuffle(chosen)
            else:
                chosen.sort(key=lambda p: (-rank_by(freq(p)), rng.random()))
            trace.append(tuple(chosen))
        streams[f"synthetic/{r}", 0] = trace
    return streams


def pooled(streams, horizon=1, replicates=400, sample="selected"):
    effects, _, _ = analyze(streams, (1, 2, 4, 8), samples=(sample,), replicates=replicates,
                            seed=1)
    return next(r for r in effects if r["group"] == "all" and r["horizon"] == horizon)


class AnalysisTests(unittest.TestCase):
    def test_hand_features(self):
        rows = list(observations(read_trace(ROOT / "examples/handwritten.jsonl"), (1, 2)))
        got = [(r["decode_step"], r["page_id"], r["rank"], r["quartile"], r["recency"],
                r["frequency"], r["reuse_1"], r["reuse_2"]) for r in rows]
        self.assertEqual(got, [(0, 0, 1, "top", None, 0, False, False),
                               (0, 1, 2, "bottom", None, 0, True, True),
                               (1, 1, 1, "top", 1, 1, True, True),
                               (1, 2, 2, "bottom", None, 0, False, False)])
        # Steps 2 and 3 lack two later steps, and no step has eight.
        self.assertEqual(list(observations(read_trace(ROOT / "examples/handwritten.jsonl"))), [])

    def test_frequency_window_and_bins(self):
        steps = [(5,)] * 12 + [(1, 2)] * 9
        rows = [r for r in observations({("a", 0): steps}, (8,)) if r["page_id"] == 5]
        self.assertEqual([r["frequency"] for r in rows], [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8])
        self.assertEqual([recency_bin(x) for x in (None, 1, 2, 3, 4, 5, 8, 9, 16, 17, 64, 65)],
                         ["new", "1", "2", "3-4", "3-4", "5-8", "5-8", "9-16", "9-16",
                          "17-32", "33-64", ">64"])
        self.assertEqual([quartile(r, 64) for r in (1, 16, 17, 48, 49, 64)],
                         ["top", "top", "middle", "middle", "bottom", "bottom"])
        self.assertIsNone(quartile(1, 1))

    def test_stratified_weights(self):
        # Stratum A: 1/2 vs 0/2 over 4 pages. Stratum B: 3/4 vs 3/4 over 8 pages. C lacks bottom.
        tables = [{"A": [2, 1, 2, 0], "B": [4, 3, 4, 3]}, {"C": [5, 5, 0, 0]}]
        effect, used, top, bottom = stratified_effect(tables)
        self.assertAlmostEqual(effect, 4 * 0.5 / 12)
        self.assertEqual((used, top, bottom), (2, 9 / 11, 3 / 6))

    def test_exact_recency_removes_within_bin_confound(self):
        rows = []
        # At recency 3 every page reuses; at recency 4 none does. Top pages are
        # concentrated at 3 and bottom pages at 4, creating a binned-recency effect.
        for recency, top_n, bottom_n, reused in ((3, 9, 1, True), (4, 1, 9, False)):
            for quartile, n in (("top", top_n), ("bottom", bottom_n)):
                for _ in range(n):
                    rows.append(dict(request_id="r", layer_id=0, frequency=1,
                                     recency=recency, recency_bin="3-4", quartile=quartile,
                                     reuse_1=reused))
        binned = stratified_effect(tally(rows, 1).values())[0]
        exact = stratified_effect(tally(rows, 1, recency_strata="exact").values())[0]
        self.assertAlmostEqual(binned, 0.8)
        self.assertAlmostEqual(exact, 0)

    def test_detects_injected_rank_signal(self):
        streams = synthetic(3, lambda rank, c: 0.8 if rank <= 2 else 0.3)
        row = pooled(streams)
        self.assertGreater(row["ci_low"], 0.2)
        self.assertEqual(row["clusters"], 30)

    def test_no_signal_interval_covers_zero(self):
        for seed in (4, 5):
            row = pooled(synthetic(seed, lambda rank, c: 0.5))
            with self.subTest(seed=seed):
                self.assertLess(row["ci_low"], 0)
                self.assertGreater(row["ci_high"], 0)

    def test_strata_remove_frequency_confound(self):
        # Reuse depends only on frequency, and rank follows frequency. A large page pool
        # makes random refills rarely reuse a page. With a small pool, a step full of
        # frequent pages leaves fewer refill slots, so relative rank gains a real effect.
        streams = synthetic(6, lambda rank, c: min(0.9, 0.1 + 0.2 * c), rank_by=lambda c: c,
                            pages=4000)
        row = pooled(streams)
        self.assertGreater(row["raw_top"] - row["raw_bottom"], 0.15)
        self.assertLess(row["ci_low"], 0)
        self.assertGreater(row["ci_high"], 0)

    def test_candidate_and_lru_samples_by_hand(self):
        steps = {("toy", 0): [(0, 1), (1, 2), (2, 3), (0, 3)]}
        fields = ("decode_step", "page_id", "rank", "n_selected", "recency", "frequency",
                  "reuse_1")
        rows = [tuple(r[f] for f in fields)
                for r in observations(steps, (1,), sample="candidate")]
        # t=1: page 0 last ranked 1 at t=0. t=2: page 0 (from t=0) and page 1 (rank 1 at t=1).
        self.assertEqual(rows, [(1, 0, 1, 2, 1, 1, False),
                                (2, 0, 1, 2, 2, 1, True),
                                (2, 1, 1, 2, 1, 2, False)])
        # With two slots, page 0 is resident when t=1 arrives (LRU then evicts it), so it
        # is a candidate at t=1 but gone by t=2, where only page 1 remains.
        lru2 = [(r["decode_step"], r["page_id"])
                for r in observations(steps, (1,), sample="lru", budget=2)]
        lru3 = [(r["decode_step"], r["page_id"])
                for r in observations(steps, (1,), sample="lru", budget=3)]
        self.assertEqual(lru2, [(1, 0), (2, 1)])
        self.assertEqual(lru3, [(1, 0), (2, 0), (2, 1)])
        with self.assertRaises(ValueError):
            list(observations(steps, (1,), sample="lru"))

    def test_lru_sample_matches_replay_residents(self):
        rng = random.Random(9)
        for _ in range(50):
            steps = [tuple(rng.sample(range(12), 4)) for _ in range(rng.randint(3, 15))]
            streams = {("r", 0): steps}
            residents = [r for _, _, r in iter_replay(steps, 6, "lru")]
            expected = {(t, p) for t in range(1, len(steps) - 1)
                        for p in residents[t - 1] - set(steps[t])}
            lru = {(r["decode_step"], r["page_id"])
                   for r in observations(streams, (1,), sample="lru", budget=6)}
            candidate = {(r["decode_step"], r["page_id"])
                         for r in observations(streams, (1,), sample="candidate")}
            seen = {(t, p) for t in range(1, len(steps) - 1)
                    for p in set().union(*steps[:t]) - set(steps[t])}
            self.assertEqual(lru, expected)
            self.assertEqual(candidate, seen)
            self.assertLessEqual(lru, candidate)

    def test_selected_signal_need_not_reach_candidates(self):
        # Rank decides which selected pages stay selected, but a page that drops out
        # returns only through rank-blind refills. Only H1-selected should fire.
        streams = synthetic(10, lambda rank, c: 0.9 if rank <= 2 else 0.4, pages=60)
        self.assertGreater(pooled(streams, sample="selected")["ci_low"], 0.2)
        candidate = pooled(streams, sample="candidate")
        self.assertLess(candidate["ci_low"], 0)
        self.assertGreater(candidate["ci_high"], 0)

    def test_detects_candidate_rank_signal(self):
        # Dropped pages come back with a probability set by their last rank.
        rng = random.Random(11)
        streams = {}
        for r in range(30):
            trace, last_rank = [], {}
            for t in range(40):
                current = set(trace[-1]) if trace else set()
                back = [p for p, rank in sorted(last_rank.items()) if p not in current
                        and rng.random() < (0.3 if rank <= 2 else 0.05)]
                chosen = rng.sample(back, min(len(back), 4))
                chosen += [p for p in (trace[-1] if trace else ()) if rng.random() < 0.3][:2]
                chosen = list(dict.fromkeys(chosen))
                pool = [p for p in range(200) if p not in chosen]
                chosen += rng.sample(pool, 8 - len(chosen))
                rng.shuffle(chosen)
                last_rank.update((p, i) for i, p in enumerate(chosen, 1))
                trace.append(tuple(chosen))
            streams[f"synthetic/{r}", 0] = trace
        row = pooled(streams, sample="candidate")
        self.assertGreater(row["ci_low"], 0.05)
        self.assertEqual(row["clusters"], 30)

    def test_band_groups_and_top_share(self):
        # Pages 1-8 drop out of the qa stream after t=2 and stay candidates; ranks 1-2 are
        # the top quartile of eight. The sum stream never drops a page.
        streams = {("qa/4-8k/1", 0): [tuple(range(1, 9))] * 3 + [tuple(range(9, 17))] * 12,
                   ("sum/8-16k/2", 1): [tuple(range(8, 0, -1))] * 15}
        effects, _, sizes = analyze(streams, (1,), samples=("selected", "candidate"),
                                    by=("band", "cell"), replicates=10)
        rows = {(r["sample"], r["group"]): r for r in effects}
        self.assertEqual(sizes, {"selected": 224, "candidate": 88})
        self.assertEqual(rows["candidate", "band 4-8k"]["observations"], 88)
        self.assertEqual(rows["candidate", "cell qa/4-8k"]["top_share"], 0.25)
        self.assertEqual(rows["candidate", "all"]["bottom_share"], 0.25)
        self.assertNotIn(("candidate", "band 8-16k"), rows)
        self.assertEqual(rows["selected", "band 8-16k"]["observations"], 112)

    def test_groups_and_single_cluster(self):
        streams = {("qa/1", 0): [(1, 2, 3, 4)] * 10, ("sum/2", 3): [(4, 3, 2, 1)] * 10}
        effects, curve, sizes = analyze(streams, (1, 8), by=("task", "layer-group"),
                                        layer_groups=2, replicates=50)
        self.assertEqual(Counter(r["group"] for r in effects),
                         Counter({"all": 2, "task qa": 2, "task sum": 2,
                                  "layers 0-1": 2, "layers 2-3": 2}))
        single = next(r for r in effects if r["group"] == "task qa")
        self.assertIsNone(single["ci_low"])
        self.assertEqual(sizes, {"selected": 2 * 2 * 4})
        rows = list(observations(streams, (1, 8)))
        self.assertTrue(all(r["reuse_rate"] == 1 for r in curve))
        with self.assertRaises(ValueError):
            analyze(streams, (0,))
        stream_level = tally(rows, 1, cluster="stream")
        self.assertEqual(sorted(stream_level), [("qa/1", 0), ("sum/2", 3)])


if __name__ == "__main__":
    unittest.main()
