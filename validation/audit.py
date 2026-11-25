"""Recompute published results and check the two cache simulators independently."""

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import kvpolicy
from efficient_dev import auc
from gap_analysis import auc as tie_correct_auc
from replay import read_trace, replay_stream


def rows(path):
    with (ROOT / path).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def json_file(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def check_split(path, id_field, hash_field, expected_sizes):
    prompts = json_file(path)["prompts"]
    groups = {
        name: {row[id_field] for row in prompts if row["split"] == name}
        for name in expected_sizes
    }
    assert {name: len(group) for name, group in groups.items()} == expected_sizes
    assert len({p[id_field] for p in prompts}) == len(prompts) == sum(expected_sizes.values())
    hashes = {name: {p[hash_field] for p in prompts if p["split"] == name} for name in groups}
    assert hashes["dev"].isdisjoint(hashes["heldout"])
    return groups


def audit_stage_one(docs):
    prereg = json_file("splits/preregistration.json")
    split_digest = hashlib.sha256((ROOT / "splits/longbench_v1.json").read_bytes()).hexdigest()
    assert prereg["split_sha256"] == split_digest
    tuning = {(int(r["delta"]), int(r["beta"])): int(r["misses"])
              for r in rows("docs/results/dev_full/tuning.csv")}
    assert tuning == {tuple(map(int, k.split(","))): v for k, v in prereg["tuning"]["misses"].items()}
    for policy, grid in (("full", tuning), ("rf", [k for k in tuning if k[0] == 0]),
                         ("rrank", [k for k in tuning if k[1] == 0])):
        entry = next(p for p in prereg["policies"] if p["name"] == policy)
        assert (entry["delta"], entry["beta"]) == min(grid, key=lambda k: (tuning[k], k))
    for split, group in docs.items():
        root = f"docs/results/{'dev_full' if split == 'dev' else 'heldout'}"
        manifest = json_file(f"{root}/manifest.json")
        assert manifest["split_sha256"] == split_digest
        assert set(manifest["request_ids"]) == group
        assert (manifest["gpu"], manifest["page_size"], manifest["top_k"]) == ("NVIDIA L4", 16, 64)
        for suffix in ("eval", "eval_truncated"):
            folder = f"{root}/{suffix}"
            measured = defaultdict(dict)
            for row in rows(f"{folder}/policies_by_doc.csv"):
                key = int(row["budget"]), row["policy"]
                doc = row["doc"]
                assert doc not in measured[key]
                counts = tuple(int(row[k]) for k in ("accesses", "misses", "compulsory_misses", "repeat_misses", "steps"))
                accesses, misses, compulsory, repeats, steps = counts
                assert 0 <= compulsory <= misses <= accesses
                assert repeats == misses - compulsory
                assert accesses == 28 * 64 * steps
                measured[key][doc] = counts
            assert all(set(per_doc) == group for per_doc in measured.values())
            for row in rows(f"{folder}/policies.csv"):
                budget, policy = int(row["budget"]), row["policy"]
                totals = [sum(col) for col in zip(*measured[budget, policy].values())]
                accesses, misses, compulsory, repeats, steps = totals
                assert totals[:4] == [int(row[k]) for k in ("accesses", "misses", "compulsory_misses", "repeat_misses")]
                assert math.isclose(float(row["miss_rate"]), misses / accesses)
                assert math.isclose(float(row["h2d_mib_per_token"]), misses * 65_536 / steps / 2**20)
                for doc in group:
                    baseline = measured[budget, "lru"][doc]
                    current = measured[budget, policy][doc]
                    assert baseline[::2] == current[::2]
                    assert measured[budget, "belady"][doc][1] <= current[1]
            session = sorted(group)
            rng = random.Random(prereg["bootstrap"]["seed"])
            lru, rf, full = (measured[128, name] for name in ("lru", "rf", "full"))

            def differences(sample):
                l, f, p = (sum(policy[doc][3] for doc in sample) for policy in (lru, rf, full))
                return (l - p) / l, f - p

            samples = [differences([rng.choice(session) for _ in session]) for _ in range(2000)]
            ci = [tuple(sorted(col)[j] for j in (50, 1950)) for col in zip(*samples)]
            result = json_file(f"{folder}/success.json")
            drop, saved = differences(session)
            assert math.isclose(drop, result["s1_full_vs_lru_repeat_reduction"])
            assert saved == result["s2_repeat_misses_saved_full_vs_rf"]
            assert all(math.isclose(x, y) for x, y in zip(ci[0], result["s1_ci"]))
            assert list(ci[1]) == result["s2_ci"]
            print(f"Rank {split}/{suffix}: repeat miss reduction {drop:.4%}; saved vs RF {saved} {ci[1]}")


def audit_long_context(docs, previously_seen):
    frozen = json_file("splits/heldout_frozen_v1.json")
    assert frozen == json_file("docs/results/longctx_heldout/frozen.json")
    assert {name.replace("__", "/", 1).removesuffix(".npz") for name in frozen["dev_docs"]} == docs["dev"]
    dev = {}
    for row in rows("docs/results/longctx_dev/misses.csv") + rows("docs/results/longctx_dev/misses_ext.csv"):
        key = row["config"], int(row["B"]), row["doc"], int(row["layer"])
        assert key not in dev
        assert key[2] in docs["dev"]
        dev[key] = tuple(int(row[k]) for k in ("accesses", "misses", "compulsory"))
    for name, config in (("lru", "lru"), ("ema", "ema:0.5"), ("belady", "belady")):
        totals = [sum(col) for col in zip(*(dev[config, 96, doc, layer] for doc in docs["dev"] for layer in range(28)))]
        assert frozen["dev_check"][name] == totals[:2]
    for row in rows("docs/results/longctx_dev/lodo.csv"):
        b, doc, chosen = int(row["B"]), row["doc"], row["chosen"]
        assert sum(dev[chosen, b, doc, layer][1] for layer in range(28)) == int(row["misses"])
    dev_lru = frozen["dev_check"]["lru"][1]
    dev_gap = dev_lru - frozen["dev_check"]["belady"][1]
    allocation = next(r for r in rows("docs/results/longctx_dev/checks_alloc.csv")
                      if r["B"] == "96" and r["base"] == "ema")
    print(f"Long dev B=96: fixed EMA G {(dev_lru-frozen['dev_check']['ema'][1])/dev_gap:.4f}; "
          f"LODO allocation G {(dev_lru-int(allocation['alloc_lodo']))/dev_gap:.4f}")
    for b, slots in frozen["alloc"].items():
        assert len(slots) == 28 and min(slots) >= 64 and sum(slots) == int(b) * 28
        config = f"ema:{frozen['alpha'][b]}"
        try:
            curves = {layer: {size: sum(dev[config, size, doc, layer][1] for doc in docs["dev"])
                              for size in (64, 72, 80, 88, 96, 104, 112, 128, 160, 192, 256)}
                      for layer in range(28)}
        except KeyError:
            print(f"Long dev B={b}: allocation verified for capacity only; full alpha grid absent from committed counts")
        else:
            assert list(kvpolicy.allocate(curves, 28, int(b) * 28, sorted(curves[0])).values()) == slots
    layer_data, pooled = {}, defaultdict(Counter)
    for row in rows("docs/results/longctx_heldout/misses.csv"):
        b, doc, layer, config = int(row["B"]), row["doc"], int(row["layer"]), row["config"]
        key = b, doc, layer, config
        acc, miss, comp = (int(row[k]) for k in ("accesses", "misses", "compulsory"))
        assert key not in layer_data and doc in docs["heldout"] and 0 <= comp <= miss <= acc
        layer_data[key] = acc, miss, comp
        pooled[b, doc, config].update(accesses=acc, misses=miss, rep=miss - comp)
    for key, (acc, miss, comp) in layer_data.items():
        b, doc, layer, config = key
        base = layer_data[b, doc, layer, "lru"]
        assert (acc, comp) == (base[0], base[2])
        if config != "alloc_ema":
            assert miss >= layer_data[b, doc, layer, "belady"][1]
        if config == "alloc_ema" and b in (72, 256):
            assert miss == layer_data[b, doc, layer, "ema"][1]
    group_layers = Counter((b, d, c) for b, d, _, c in layer_data)
    assert set(group_layers.values()) == {28}
    assert len(layer_data) == 22344
    for row in rows("docs/results/longctx_heldout/doc_level.csv"):
        key = int(row["B"]), row["doc"], row["config"]
        assert pooled[key] == {k: int(row[k]) for k in ("accesses", "misses", "rep")}
    ordered = sorted(docs["heldout"])

    def vector(b, config, key="misses"):
        return np.array([pooled[b, doc, config][key] for doc in ordered])

    lru, best, ema, both = (vector(96, name) for name in ("lru", "belady", "ema", "alloc_ema"))
    samples = np.random.default_rng(0).integers(0, len(ordered), (2000, len(ordered)))
    gap = (lru[samples] - best[samples]).sum(axis=1)
    ema_ci = np.percentile((lru[samples] - ema[samples]).sum(axis=1) / gap, [2.5, 97.5])
    delta_ci = np.percentile((ema[samples] - both[samples]).sum(axis=1) / gap, [2.5, 97.5])
    assert np.allclose(ema_ci, [0.277, 0.285], atol=0.0005)
    assert np.allclose(delta_ci, [0.113, 0.128], atol=0.0005)
    assert np.count_nonzero(ema < lru) == 21
    print(f"Long held-out: EMA G CI {ema_ci.round(4)}, allocation delta G CI {delta_ci.round(4)}; like-for-like EMA wins {(ema < lru).sum()}/{len(ordered)}")
    for b in (72, 80, 96, 128, 256):
        l, o = vector(b, "lru"), vector(b, "belady")
        denom = (l - o).sum()
        n = vector(b, "lru", "accesses").sum()
        gains = [((l - vector(b, cfg)).sum() / denom) for cfg in ("ema", "alloc_ema")]
        repeats = [(l - vector(b, cfg)).sum() / vector(b, "lru", "rep").sum() for cfg in ("ema", "alloc_ema")]
        if b == 96:
            assert np.allclose(gains, [0.281, 0.402], atol=0.0005)
            assert np.allclose(repeats, [0.116, 0.166], atol=0.0005)
        print(f"  B={b}: LRU {l.sum()/n:.3%}, Belady {o.sum()/n:.3%}; G {gains[0]:.4f}/{gains[1]:.4f}, repeat savings {repeats[0]:.3%}/{repeats[1]:.3%}")
    overlaps = [d for d in ordered if d in previously_seen]
    novel = np.array([d not in previously_seen for d in ordered])
    if overlaps:
        gain = (lru - best)[novel].sum()
        print(f"Cross-experiment source sensitivity: omit {len(overlaps)} previously used document(s); "
              f"EMA G {(lru-ema)[novel].sum()/gain:.4f}, allocation + EMA G {(lru-both)[novel].sum()/gain:.4f}; "
              f"EMA wins {np.count_nonzero(ema[novel] < lru[novel])}/{novel.sum()}")


def audit_gpu_tables(heldout):
    plain, fused = rows("results_gpu/bench_summary.csv"), rows("results_gpu/bench_fused_summary.csv")
    for row in plain:
        if row["miss"]:
            assert math.isclose(float(row["MB_tok"]), float(row["miss"]) * 65_536 / 384 / 10**6)
    distinct = {r["doc"] for r in fused}
    assert len(distinct) == 3 and all(sum(d.startswith(p) for d in heldout) == 1 for p in distinct)
    runs = {(r["doc"], r["cfg"], r["B"], r["rep"]): r for r in fused}
    assert len(runs) == len(fused)
    errors = []
    for row in fused:
        assert row["rank_eq"] == "1.0" and row["gen_len"] == "385"
        if row["miss"]:
            error = abs(int(row["miss"]) - int(row["sim"]))
            if row["cfg"] in ("lru", "belady"):
                assert error == 0
            errors.append(error)
    diffs = [float(runs[d, "ema", "96", r]["med_ms"]) - float(runs[d, "lru", "96", r]["med_ms"])
             for d in sorted(distinct) for r in ("0", "1")]
    assert max(errors) == 39
    print(f"GPU saved summaries: max simulator miss error = {max(errors)}; fused paired EMA minus LRU at B=96 mean {mean(diffs):+.3f} ms, range [{min(diffs):+.2f}, {max(diffs):+.2f}]")
    for table, b, field in ((plain, "96.0", "med"), (fused, "96", "med_ms")):
        lru = [float(r[field]) for r in table if r["cfg"] == "lru" and r["B"] == b and r.get("timing") != "True"]
        print(f"  LRU {field}: mean of run medians {mean(lru):.4f}, median of run medians {median(lru):.4f}")
    for b in ("96", "128"):
        counts = [sum(int(runs[d, cfg, b, "0"]["miss"]) for d in distinct) for cfg in ("lru", "ema", "alloc_ema")]
        print(f"  B={b}: demand-byte savings {1-counts[1]/counts[0]:.3%}/{1-counts[2]/counts[0]:.3%}")


def exact_optimum(trace, budget):
    costs = {frozenset(): 0}
    for row in trace:
        selected = frozenset(row)
        new = {}
        for old, cost in costs.items():
            eligible = old - selected
            for keep in combinations(eligible, min(budget - len(selected), len(eligible))):
                resident = selected.union(keep)
                score = cost + len(selected - old)
                new[resident] = min(new.get(resident, score), score)
        costs = new
    return min(costs.values())


def reference_ema(trace, scores, budget, alpha):
    resident = {}
    hits = 0
    for step, selection in enumerate(trace):
        scores_now = scores[step]
        cutoff = scores_now[selection[-1]]
        scaled = (scores_now - cutoff) / (scores_now[selection[0]] - cutoff + 1e-6)
        protected = set(selection) & resident.keys()
        hits += len(protected)
        updated = {p: (alpha * v + (1 - alpha) * scaled[p], t, h) for p, (v, t, h) in resident.items()}
        keep = sorted(updated.keys() - protected, key=lambda p: (*updated[p], -p), reverse=True)
        resident = {p: updated[p] for p in keep[:budget - len(selection)]}
        for p in selection:
            resident[p] = (updated[p][0] if p in protected else scaled[p], step, p in protected)
    return sum(map(len, trace)) - hits


def audit_policies():
    rng = np.random.default_rng(4722)
    rank_differences = Counter()
    for _ in range(400):
        scores = rng.integers(-5, 6, (9, 7)).astype(float)
        k, budget = int(rng.integers(1, 4)), int(rng.integers(4, 7))
        trace = np.argsort(-scores, axis=1, kind="stable")[:, :k].tolist()
        exact = exact_optimum(trace, budget)
        assert replay_stream(trace, budget, "belady")["misses"] == exact
        assert kvpolicy.simulate(trace, None, budget, "belady")["misses"] == exact
        for delta, beta in ((0, 0), (0, 2), (4, 2), (8, 2)):
            one = replay_stream(trace, budget, "score", delta=delta, beta=beta)
            two = kvpolicy.simulate(trace, None, budget, "rank", delta=delta, beta=beta)
            assert min(one["misses"], two["misses"]) >= exact
            rank_differences[delta, beta] += one["misses"] != two["misses"]
        assert kvpolicy.simulate(trace, scores, budget, "ema", alpha=0.5)["misses"] == reference_ema(trace, scores, budget, 0.5)
    for _ in range(100):
        grid = (4, 6, 8)
        curves = {layer: {b: int(rng.integers(0, 20)) for b in grid} for layer in range(4)}
        budget = int(rng.integers(16, 33))
        exact = min(sum(curves[layer][b] for layer, b in enumerate(config))
                    for config in product(grid, repeat=4) if sum(config) <= budget)
        plan = kvpolicy.allocate_dp(curves, 4, budget, grid)
        assert sum(plan.values()) <= budget
        assert sum(curves[layer][b] for layer, b in plan.items()) == exact
    print("Bélády and EMA matched independent references on 400 traces; exact allocation matched exhaustive search on 100 grids")
    print(f"Rank-policy differences across 400 traces (delta, beta): {dict(rank_differences)}")


def audit_known_limitations():
    tied = np.ones(2)
    positive_first, positive_last = np.array([True, False]), np.array([False, True])
    print(f"Long-dev AUC for equal scores: {auc(tied, positive_first)}, {auc(tied, positive_last)} after reversal; "
          f"tie-correct reference {tie_correct_auc(tied, positive_first)}")
    dev_name = json_file("splits/heldout_frozen_v1.json")["dev_docs"][0]
    rid = dev_name.replace("__", "/", 1).removesuffix(".npz")
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        np.savez(folder / "renamed.npz", rid=rid,
                 ranked=np.tile(np.arange(64, dtype=np.int16), (2, 1, 1)),
                 scores=np.tile(np.arange(64, 0, -1, dtype=np.float16), (2, 1, 1)))
        evaluation = subprocess.run(
            [sys.executable, "heldout.py", "eval", str(folder), "--frozen", "splits/heldout_frozen_v1.json",
             "--out", str(folder / "out"), "--procs", "1"], cwd=ROOT, capture_output=True, text=True,
        )
        print(f"Held-out driver accepted a renamed, one-layer synthetic trace carrying a dev document ID: "
              f"{evaluation.returncode == 0 and (folder / 'out/misses.csv').exists()}")


def replay_exported_document(data):
    with TemporaryDirectory() as temp:
        path = Path(temp) / "trace.jsonl"
        path.write_bytes(data)
        streams = read_trace(path)
    counts = Counter()
    for steps in streams.values():
        for label, delta, beta in (("lru", 0, 0), ("rf", 0, 2), ("rank", 4, 2)):
            counts[label] += replay_stream(steps, 128, "score", delta=delta, beta=beta)["repeat_misses"]
        counts["belady"] += replay_stream(steps, 128, "belady")["repeat_misses"]
    return counts


def audit_colab(path):
    with ZipFile(path) as bundle, ProcessPoolExecutor(max_workers=4) as pool:
        entries = sorted(n for n in bundle.namelist() if n.endswith(".jsonl"))
        assert len(entries) == 40
        result = Counter()
        for counts in pool.map(replay_exported_document, (bundle.read(n) for n in entries)):
            result.update(counts)
    assert result == {"lru": 162369, "rf": 161850, "rank": 161712, "belady": 58764}, result
    print(f"User's Colab trace, replayed on upstream implementation at B=128: {dict(result)}; matched-beta rank saved {result['rf']-result['rank']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colab-zip", type=Path, help="Optional original 40-document Colab export")
    args = parser.parse_args()
    short = check_split("splits/longbench_v1.json", "request_id", "context_sha256", {"dev": 20, "heldout": 40})
    long = check_split("splits/longctx_v1.json", "rid", "ctx", {"dev": 14, "heldout": 21})
    audit_stage_one(short)
    previously_seen = {f"{t}/{doc}" for group in short.values() for t, _, doc in (r.split("/") for r in group)}
    audit_long_context(long, previously_seen)
    audit_gpu_tables(long["heldout"])
    audit_policies()
    audit_known_limitations()
    if args.colab_zip:
        audit_colab(args.colab_zip)


if __name__ == "__main__":
    main()
