"""F7/F8 on a real CUDA device (tiny random Qwen3). Skipped without torch, transformers or CUDA.
Ported from PR #4 validation/test_gpu_contracts.py and validation/run_gpu.py."""
import weakref

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(1234)
    config = transformers.Qwen3Config(vocab_size=256, hidden_size=128, intermediate_size=256,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                      head_dim=32, max_position_embeddings=4096, attention_dropout=0.0)
    return transformers.Qwen3ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()


@pytest.fixture(scope="module")
def probe(model):
    import gpu_bench as gb
    cache_ref, tensor_refs, observations = [], [], []

    def after_prefill(module, args, kwargs, output):
        if kwargs["input_ids"].shape[1] > 1:
            cache_ref.append(weakref.ref(output.past_key_values))
            tensor_refs.extend(weakref.ref(t) for t in output.past_key_values.key_cache)
            tensor_refs.extend(weakref.ref(t) for t in output.past_key_values.value_cache)

    def before_decode(module, args, kwargs):
        if kwargs["input_ids"].shape[1] == 1:
            observations.append(dict(prefill_cache_alive=cache_ref[0]() is not None,
                                     prefill_kv_tensors_alive=sum(r() is not None for r in tensor_refs)))

    pre = model.register_forward_pre_hook(before_decode, with_kwargs=True)
    post = model.register_forward_hook(after_prefill, with_kwargs=True)
    try:
        result = gb.run(model, [3, 4, 5, 6] * 512, "lru", steps=4, budgets=[96, 96], warmup=2)
    finally:
        pre.remove()
        post.remove()
    return observations, result


def test_tiered_decode_releases_prefill_cache_before_memory_measurement(probe):
    observations, _ = probe
    assert not observations[0]["prefill_cache_alive"]
    assert observations[0]["prefill_kv_tensors_alive"] == 0


def test_warmup_is_applied_to_latency_samples(probe):
    _, r = probe
    assert len(r["lat"]) == 4 and r["warmup_mask"].tolist() == [True, True, False, False]
    assert r["metrics"]["lat_steady"]["n"] == 2
    assert np.array_equal(r["step_miss"].sum(0), r["misses"])
    assert r["metrics"]["misses_steady"] == r["step_miss"][2:].sum(0).tolist()


def test_memory_phases_are_reported_separately(probe):
    _, r = probe
    for key in ("peak_prefill", "peak_migration", "peak_decode_warmup", "peak_decode_steady", "mem_start"):
        assert isinstance(r[key], int)
    assert r["peak_mem"] == max(r["peak_decode_warmup"], r["peak_decode_steady"])
