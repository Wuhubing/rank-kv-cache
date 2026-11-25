"""Batched two-tier cache (gpu_bench_batch): per-request misses equal the simulator, selection and
tokens identical across policies, batch = 1 equals gpu_bench's resident Quest path. CUDA only."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

import gpu_bench as gb  # noqa: E402
import gpu_bench_batch as gbb  # noqa: E402
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
def test_misses_match_simulator_and_policies_agree(model, N):
    P = prompts(3, N)
    steps, B = 24, [96, 80]
    lru = gbb.run(model, P, "lru", steps, B)
    nxt = gbb.next_use_batch(lru["ranks"])
    bel = gbb.run(model, P, "belady", steps, B, nxt=nxt)
    ema = gbb.run(model, P, "ema", steps, B, alpha=0.5, record_scores=True)
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


def test_batch_one_matches_resident_quest(model):
    P = prompts(1, 2050, seed=3)
    ref = gb.run(model, P[0], "resident", steps=16)
    got = gbb.run(model, P, "lru", 16, [96, 96])
    assert got["gen"][0] == ref["gen"]
    assert np.array_equal(got["ranks"][:, 0], ref["ranks"])


def test_bytes_count(model):
    P = prompts(2, 2048, seed=5)
    res = gbb.run(model, P, "lru", 12, [96, 96])
    page_bytes = 2 * 2 * 16 * 32 * 2  # K+V, kvh=2, 16 tokens, head_dim 32, bf16
    assert res["bytes"] == int(res["misses"].sum()) * page_bytes
