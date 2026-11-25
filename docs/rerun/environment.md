# S1 环境（wulab2）

记录于 2026-11-24。

| 项目 | 值 |
|---|---|
| 机器 | wulab2，Ubuntu 22.04.5 LTS，内核 5.15.0-176，48 核，503 GB 内存 |
| 项目目录 | `/data/weibing/projects/rank-kv-cache`（https 克隆，分支 `replication-v2`） |
| 原始 trace | `/data/weibing/projects/rank-kv-cache/data/`（`.gitignore` 中，不提交、不上传） |
| 缓存 | `HF_HOME=/data/weibing/cache/huggingface`（由 `~/.wulab_env.sh` 设置，已确认），`TMPDIR=/data/weibing/tmp` |
| GPU（固定） | **卡 0**，NVIDIA RTX 6000 Ada Generation，UUID `GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3`；所有采集与 benchmark 都用 `CUDA_VISIBLE_DEVICES=0` |
| 驱动 | 550.90.12（最高 CUDA 12.4） |
| Python | 3.11.18（`uv venv`，uv 0.13.10，`.venv` 5.5 GB） |
| torch | 2.6.0+cu124（来自 download.pytorch.org/whl/cu124） |
| transformers | 4.53.3（tokenizers 0.21.4，safetensors 0.8.0） |
| 其他 | numpy 2.4.6，scipy 1.17.1，pytest 9.2.0，rouge 1.0.1，matplotlib 3.11.2，datasets 5.0.1 |
| 模型 | `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`（与 `splits/longctx_v1.json` 一致） |
| 数据 | `zai-org/LongBench@5e628be450b7e67fb7ae6e201bd6d8f7056f7672` 的 `data.zip` |

与原 L4 环境的差别：原来用 Colab L4 和 torch 2.14.1+cu130；wulab2 驱动只支持 CUDA 12.4，所以换成 torch 2.6.0+cu124。数值行为可能不同，这本来就是跨硬件复现的一部分。

后来补装（同一个 `.venv`）：`setuptools`（triton 3.2.0 编译 kernel 需要它；没有它时 `gpu_bench.py` 的 `import triton` 会静默失败，fused 路径不可用）、`pandas` 和 `pyarrow`（读 RULER 的 parquet）。S6 另外下载了 `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`（16 GB）、`microsoft/SCBench@283310bb…` 的 `scbench_summary`/`scbench_qa_eng`、`simonjegou/ruler@24adceac…` 的 16384 配置，都在 `HF_HOME` 下。

## 检查
- mini（uv Python 3.11，仅 CPU）：`pytest tests validation` → 213 passed，2 skipped（需要 CUDA）。
- wulab2（`CUDA_VISIBLE_DEVICES=0`）：`pytest tests validation` → 218 passed，2 xpassed（`test_gpu_contracts.py` 的 F7/F8 在 #7 修复后预期通过，标记仍是 xfail(strict=False)），0 skipped。
- 选择器自检：`validation/test_gpu_contracts.py::test_notebook_selector_against_its_brute_force_reference` 在 CUDA 上通过；S2 的 `collect/long.py` 另有自己的暴力比对测试。
