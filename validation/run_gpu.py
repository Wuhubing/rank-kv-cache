"""CUDA correctness and memory-lifetime checks without external models or traces.

python -m validation.run_gpu --out validation/results/gpu_experiments.json
Uses real 64 KiB KV pages and a separate tiny random Qwen3 model. These are
synthetic correctness measurements, not pretrained-model latency benchmarks.
"""
import argparse
import importlib.metadata
import json
import platform
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import gpu_bench as gb
import kvpolicy as kp


@torch.inference_mode()
def component_checks():
    torch.manual_seed(2718)
    device, dtype = "cuda", torch.bfloat16
    n, kvh, dim, heads, steps = 140, 8, 128, 16, 40
    module = SimpleNamespace(layer_idx=0)
    records = []
    for tail in (0, 7, 15):
        initial_k = torch.randn(1, kvh, n * gb.PS + tail, dim, device=device, dtype=dtype)
        initial_v = torch.randn_like(initial_k)
        queries = torch.randn(steps, 1, heads, 1, dim, device=device, dtype=dtype)
        keys = torch.randn(steps, 1, kvh, 1, dim, device=device, dtype=dtype)
        values = torch.randn_like(keys)
        resident = gb.Bench("resident")
        resident.rb = {}
        resident.rec = [None]
        full_k, full_v = initial_k, initial_v
        references, ranks, score_rows = [], [], []
        pmax = n + (tail + steps) // gb.PS + 1
        for t in range(steps):
            full_k = torch.cat((full_k, keys[t]), 2)
            full_v = torch.cat((full_v, values[t]), 2)
            output, _ = resident.quest_resident(module, queries[t], full_k, full_v, None, dim ** -0.5)
            references.append(output)
            ranks.append(resident.rec[0].cpu().numpy())
            lo, hi = resident.rb[0]
            q = queries[t, 0, :, 0].reshape(kvh, heads // kvh, dim).float()
            scores = (torch.einsum("hgd,hpd->hgp", q.clamp_min(0), hi)
                      + torch.einsum("hgd,hpd->hgp", q.clamp_max(0), lo)).mean((0, 1))
            row = np.full(pmax, np.nan, dtype=np.float32)
            row[:scores.numel()] = scores.cpu().numpy()
            score_rows.append(row)
        ranked, scores = np.stack(ranks), np.stack(score_rows)
        pk = initial_k[0, :, :n * gb.PS].reshape(kvh, n, gb.PS, dim).transpose(0, 1).cpu()
        pv = initial_v[0, :, :n * gb.PS].reshape(kvh, n, gb.PS, dim).transpose(0, 1).cpu()
        for budget in (64, 96):
            for policy in ("lru", "ema", "belady"):
                bench = gb.Bench(policy)
                tier = gb.Tier(pk, pv, initial_k[0, :, n * gb.PS:].clone(),
                               initial_v[0, :, n * gb.PS:].clone(), budget, pmax, policy, 0.5, None)
                bench.tiers[0] = tier
                if policy == "belady":
                    bench.nxt_all = gb.next_use(ranked[:, None])
                maximum_error, mismatched_ranks = 0.0, 0
                for t in range(steps):
                    bench.step, bench.rec, bench.evs = t, [None], []
                    actual, _ = bench.quest_tiered(module, queries[t], keys[t], values[t], None, dim ** -0.5)
                    maximum_error = max(maximum_error, (actual - references[t]).abs().max().item())
                    mismatched_ranks += int(not np.array_equal(bench.rec[0].cpu().numpy(), ranked[t]))
                    # Verify every resident slot against the authoritative full KV store.
                    pos = tier.pos.cpu().numpy()
                    occupied = np.flatnonzero(pos >= 0)
                    expected_k = full_k[0, :, :tier.np * gb.PS].reshape(kvh, tier.np, gb.PS, dim).transpose(0, 1)
                    expected_v = full_v[0, :, :tier.np * gb.PS].reshape(kvh, tier.np, gb.PS, dim).transpose(0, 1)
                    assert torch.equal(tier.sk[occupied], expected_k[pos[occupied]])
                    assert torch.equal(tier.sv[occupied], expected_v[pos[occupied]])
                    assert len(set(pos[occupied].tolist())) == len(occupied)
                simulated = kp.simulate(ranked, scores, budget, policy)["misses"]
                assert mismatched_ranks == 0 and maximum_error == 0.0
                if policy != "ema":
                    assert tier.misses == simulated
                records.append(dict(tail=tail, B=budget, policy=policy, steps=steps,
                                    page_bytes=2 * kvh * gb.PS * dim * 2,
                                    gpu_misses=tier.misses, simulator_misses=simulated,
                                    rank_mismatches=mismatched_ranks, max_attention_error=maximum_error,
                                    resident_kv_exact=True))
                bench.tiers.clear()
    return records


def tiny_model():
    torch.manual_seed(1234)
    config = Qwen3Config(vocab_size=256, hidden_size=128, intermediate_size=256,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=32, max_position_embeddings=4096, attention_dropout=0.0)
    return Qwen3ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()


def model_checks(model):
    records = []
    rng = np.random.default_rng(1234)
    for length in (2048, 2055, 2063):
        prompt = rng.integers(3, 256, length).tolist()
        reference = gb.run(model, prompt, "resident", steps=24)
        for mode in ("lru", "ema", "belady"):
            result = gb.run(model, prompt, mode, steps=24, budgets=[96, 96], alpha=0.5,
                            trace=reference["ranks"])
            tokens_equal = result["gen"] == reference["gen"]
            ranks_equal = np.array_equal(result["ranks"], reference["ranks"])
            assert tokens_equal and ranks_equal
            records.append(dict(prompt_tokens=length, policy=mode, decode_steps=24,
                                tokens_equal=tokens_equal, rankings_equal=ranks_equal,
                                misses=result["misses"], raw_latency_samples=len(result["lat"])))
    return records


def memory_lifetime_probe(model):
    cache_ref, tensor_refs, observations = [], [], []
    def after_prefill(module, args, kwargs, output):
        if kwargs["input_ids"].shape[1] > 1:
            cache_ref.append(weakref.ref(output.past_key_values))
            tensor_refs.extend(weakref.ref(t) for t in output.past_key_values.key_cache)
            tensor_refs.extend(weakref.ref(t) for t in output.past_key_values.value_cache)
    def before_decode(module, args, kwargs):
        if kwargs["input_ids"].shape[1] == 1:
            observations.append(dict(decode_step=len(observations),
                                     prefill_cache_alive=cache_ref[0]() is not None,
                                     prefill_kv_tensors_alive=sum(r() is not None for r in tensor_refs),
                                     allocated_bytes=torch.cuda.memory_allocated()))
    pre = model.register_forward_pre_hook(before_decode, with_kwargs=True)
    post = model.register_forward_hook(after_prefill, with_kwargs=True)
    try:
        result = gb.run(model, [3, 4, 5, 6] * 512, "lru", steps=4, budgets=[96, 96], warmup=2)
    finally:
        pre.remove()
        post.remove()
    return {"observations": observations, "warmup_argument": 2,
            "steps": 4, "returned_latency_samples": len(result["lat"]),
            "reported_peak_bytes": result["peak_mem"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this runner; CPU validation is separate.")
    started = time.perf_counter()
    components = component_checks()
    model = tiny_model()
    integration = model_checks(model)
    memory = memory_lifetime_probe(model)
    result = dict(environment={"python": platform.python_version(), "platform": platform.platform(),
                               "torch": torch.__version__, "cuda": torch.version.cuda,
                               "transformers": importlib.metadata.version("transformers"),
                               "gpu": torch.cuda.get_device_name(), "dtype": "bfloat16"},
                  component_runs=components, tiny_random_qwen3_runs=integration,
                  memory_lifetime_probe=memory,
                  zero_copy="not executed: Triton unavailable on this Windows environment" if gb.triton is None else "available but not exercised by this synchronous-path runner",
                  wall_seconds=time.perf_counter() - started,
                  scope="Synthetic correctness only; no external traces, pretrained weights, or general speedup claims.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"component_runs": len(components), "model_runs": len(integration),
                      "memory_lifetime_probe": memory, "zero_copy": result["zero_copy"]}, indent=2))


if __name__ == "__main__":
    main()
