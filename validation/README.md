# Comprehensive systems validation

Read [the evaluation](REPORT.md) for conclusions, defects, measured results, and the remaining experimental limits. This branch evaluates `efficient-policy` at `24e5018341f0a0910a2c6e7f41bd7081522a2bc0`.

Production Python and notebook code are unchanged. The only corrective change outside `validation/` is `.gitattributes`, which preserves LF bytes in JSON files so frozen split hashes survive Windows checkouts. Existing split contents and hashes are unchanged in Git.

## Reproduce the CPU evaluation

Use a fresh Python environment; Python 3.13.5 was used for this evaluation. Run from the repository root.

```bash
python -m pip install -r validation/requirements-cpu.txt
python -X utf8 -m pytest -q -ra
python -X utf8 -m validation.run_cpu --out validation/results/cpu_experiments.json
```

Tests marked `xfail(strict=True)` reproduce unresolved defects. They are not successful correctness checks. `--runxfail` runs their assertions normally and exits with failures. A future fix causes an unexpected pass until its marker is removed.

```bash
python -X utf8 -m pytest validation --runxfail -q
```

CPU-only installations skip the GPU module. Use pytest: the upstream README's unittest command discovers only 50 tests, omitting the 15 parametrized/function-based policy tests. If site-installed pytest plugins interfere, set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

## Reproduce the CUDA evaluation

The measured setup was Windows 11, NVIDIA RTX 5080 Laptop GPU, driver 592.27, Python 3.13.5, PyTorch 2.7.1 with CUDA 12.8, Transformers 4.53.3, and BF16. The driver-reported supported CUDA version was 13.1; the PyTorch runtime was 12.8. The environment was installed under this task's `work/` directory without changing the user's base Python installation.

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==4.53.3
python -X utf8 -m validation.run_gpu --out validation/results/gpu_experiments.json
python -X utf8 -m validation.bench_layer --out validation/results/layer_benchmark.json
python -X utf8 -m validation.plot_results
```

`run_gpu` checks the synchronous path using 64 KiB pages and a randomly initialized two-layer Qwen3 model. It downloads no weights or datasets. `bench_layer` measures one layer with correlated synthetic queries, fixed input data, randomized configuration order, seven repeats, and explicit exclusion of the first 16 of 192 steps. Its static resident baseline preallocates KV storage. It does not measure full-model throughput, batch scaling, model quality, or Triton zero-copy behavior.

The original long-context trace and GPU-result folders were excluded at the user's request. The committed 40-document rank-study counts can be reaggregated; the newer 21-document score-EMA claim cannot be independently reproduced from the repository alone.

## Evidence files

| File | Evidence |
| --- | --- |
| `results/test_results.txt` | Complete suite result, including unresolved defects |
| `results/defect_reproductions.txt` | Ordinary failing assertions with xfail disabled |
| `results/upstream_initial.txt` | Initial Windows checkout failures |
| `results/upstream_lf.txt` | All 65 upstream tests after restoring LF bytes |
| `results/cpu_experiments.json` | Source hashes, committed-table audit, synthetic regimes, exact counterexample |
| `results/gpu_experiments.json` | CUDA correctness and memory-lifetime evidence |
| `results/layer_benchmark.json` | All per-step timings, misses, run summaries, transfer-bandwidth samples |
| `results/systems_evaluation.png` | Plot of measured synthetic results |

The validation branch deliberately preserves production defects for review. The report gives a specific remedy and a regression check for each one.
