"""Fixed batched path (gpu_bench_batch2, followup-v3 E1): per-request misses equal the simulator for LRU, Belady and
EMA; ranks and tokens equal the old batched path; all-miss fetches K pages per step; the slow-tier spin only adds
time. CUDA only."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

import gpu_bench_batch as gbb  # noqa: E402
import gpu_bench_batch2 as gb2  # noqa: E402
import kvpolicy as kp  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(1234)
    cfg = Qwen3Config(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                      max_position_embeddings=4096, attention_dropout=0.0)
    return Qwen3ForCausalLM(cfg).to(device="cuda", dtype=torch.bfloat16).eval()


def prompts(R, N, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(3, 256, N).tolist() for _ in range(R)]


@pytest.mark.parametrize("N", [2048, 2055])
def test_misses_match_simulator_and_old_path(model, N):
    P = prompts(3, N)
    steps, B = 24, [96, 80]
    lru = gb2.run(model, P, "lru", steps, B)
    old = gbb.run(model, P, "lru", steps, B)
    assert np.array_equal(lru["ranks"], old["ranks"]) and lru["gen"] == old["gen"]
    assert np.array_equal(lru["misses"], old["misses"])
    nxt = gbb.next_use_batch(lru["ranks"])
    bel = gb2.run(model, P, "belady", steps, B, nxt=nxt)
    ema = gb2.run(model, P, "ema", steps, B, alpha=0.5, record_scores=True)
    assert np.array_equal(lru["ranks"], bel["ranks"]) and np.array_equal(lru["ranks"], ema["ranks"])
    assert lru["gen"] == bel["gen"] == ema["gen"]
    for r in range(3):
        for l in range(2):
            R = lru["ranks"][:, r, l]
            assert lru["misses"][r, l] == kp.simulate(R, None, B[l], "lru")["misses"]
            assert bel["misses"][r, l] == kp.simulate(R, None, B[l], "belady")["misses"]
            S = np.full((steps, N // 16 + steps), np.nan, np.float32)
            for t in range(steps):
                row = ema["scores"][t][l][r]
                S[t, :len(row)] = row
            sim = kp.simulate(R, S, B[l], "ema", alpha=0.5)["misses"]
            assert abs(int(ema["misses"][r, l]) - sim) <= 2  # float32 GPU vs float64 replay near-ties
    assert lru["step_misses"].sum() == lru["misses"].sum()
    assert lru["bytes"] == int(lru["misses"].sum()) * 2 * 2 * 16 * 32 * 2


def test_allmiss_fetches_k_pages_and_keeps_tokens(model):
    P = prompts(2, 2050, seed=3)
    lru = gb2.run(model, P, "lru", 12, 96)
    am = gb2.run(model, P, "allmiss", 12, 96)
    assert np.array_equal(am["misses"], np.full((2, 2), 12 * 64))
    assert am["gen"] == lru["gen"] and np.array_equal(am["ranks"], lru["ranks"])
    nf = gb2.run(model, P, "nofetch", 12, 96)
    assert nf["lat"].shape == (12,)


def test_slow_tier_spin_adds_time_only(model):
    P = prompts(2, 2048, seed=5)
    fast = gb2.run(model, P, "lru", 16, 96, warmup=4)
    slow = gb2.run(model, P, "lru", 16, 96, warmup=4, delay_ns_per_byte=20.0)
    assert slow["gen"] == fast["gen"] and np.array_equal(slow["misses"], fast["misses"])
    # 20 ns/byte x 2 KiB pages: every missed page adds ~41 us; the first steps miss many pages
    assert slow["lat"][:4].sum() > fast["lat"][:4].sum()


@pytest.mark.parametrize("overlap", [False, True])
def test_overlap_modes_agree(model, overlap):
    P = prompts(2, 2048, seed=7)
    a = gb2.run(model, P, "ema", 20, 96, alpha=0.5, overlap=overlap)
    b = gbb.run(model, P, "ema", 20, 96, alpha=0.5)
    assert a["gen"] == b["gen"] and np.array_equal(a["ranks"], b["ranks"])


def test_timing_breakdown(model):
    P = prompts(2, 2048, seed=6)
    res = gb2.run(model, P, "ema", 10, 96, alpha=0.5, timing=True, warmup=2)
    assert res["breakdown"].shape == (10, 7) and (res["breakdown"] >= 0).all()
