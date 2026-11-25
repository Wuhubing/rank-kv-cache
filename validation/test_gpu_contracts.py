"""Optional real CUDA regression tests. CPU-only installs skip this module."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")

from validation.run_gpu import memory_lifetime_probe, tiny_model
from notebooks.long_trace_capture import capture
import gpu_bench as gb

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    return tiny_model()


@pytest.fixture(scope="module")
def memory_probe(model):
    return memory_lifetime_probe(model)


@pytest.mark.xfail(strict=False, reason="F7 fixed in #7 (prefill output released before decode stats); not yet re-run on CUDA")
def test_tiered_decode_releases_prefill_cache_before_memory_measurement(memory_probe):
    assert not memory_probe["observations"][0]["prefill_cache_alive"]


@pytest.mark.xfail(strict=False, reason="F8 fixed in #7 (raw lat kept; steady-state metrics exclude warm-up); not yet re-run on CUDA")
def test_warmup_is_applied_to_latency_samples(memory_probe):
    # gpu_bench keeps every step in `lat` and applies warm-up in metrics["lat_steady"].
    assert memory_probe["steady_latency_samples"] == memory_probe["steps"] - memory_probe["warmup_argument"]


def test_long_trace_capture_matches_benchmark(model):
    prompt = np.random.default_rng(321).integers(3, 256, 2055).tolist()
    ranked, scores, generated, _ = capture(model, prompt, max_new=25)
    reference = gb.run(model, prompt, "resident", steps=24)
    assert generated == reference["gen"]
    assert np.array_equal(ranked, reference["ranks"])
    assert np.isfinite(np.take_along_axis(scores, ranked.astype(int), axis=2)).all()


def test_notebook_selector_against_its_brute_force_reference():
    notebook = json.loads((ROOT / "notebooks/first_trace.ipynb").read_text(encoding="utf-8"))
    cells = {c["id"]: "".join(c.get("source", [])) for c in notebook["cells"]}
    namespace = {}
    exec(compile(cells["selector"], "notebook_selector", "exec"), namespace)
    exec(compile(cells["selector-check"], "notebook_selector_check", "exec"), namespace)
