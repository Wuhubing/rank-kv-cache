# 本地（M1 Max）运行说明

分支 `xiaoyang/score-ema-final`。trace 采集和策略回放都在 Mac 上做；只有 GPU 卸载延迟测试需要 Colab（Mac 是统一内存，没有 PCIe 传输）。

## 1. 环境（一次）

```bash
cd ~/Desktop/MIT/6.5940/rank-kv-cache-final
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt
python -m pytest -q          # 应全部通过
```

## 2. 自检（约 2–5 分钟，第一次会下载模型和 LongBench）

```bash
python local/capture.py check
```

在第一篇 dev 文档的 4K token 片段上解码 48 步，跑三遍。必须三项 PASS：
- `selector_bounds`：增量维护的 page 上下界与整段重算一致；
- `stable_sort`：MPS 上的排序是稳定降序（同分时小 page id 在前）；
- `rerun_identical`：同一台机器重跑，排名和 token 逐位一致。

另外会打印：完整 prompt 的 token 数是否与 split 一致、分块 prefill 与一次性 prefill 的差别（仅供参考）、每步解码耗时。

## 3. Pilot（一篇 26K token 的 QMSum，约 740 步）

```bash
python local/capture.py run --rids qmsum/cb5b1672ca426597251b6def8cbab8ec5249c9662a0c9888 --out traces/pilot
python local/quick_replay.py traces/pilot --budgets 96 128 --configs lru belady score ema:0.5 rank:8:2 \
    --compare docs/results/longctx_dev/misses.csv
```

第二条会并排打印本地（MPS）和 L4 上同一篇文档的 LRU/Bélády miss rate 和各策略 G。

## 4. 全部 14 篇 dev（可中断，重跑会跳过已完成的）

```bash
python local/capture.py run --subset dev --out traces/mps_dev
python local/quick_replay.py traces/mps_dev --budgets 96 --out results/mps_dev/quick.csv \
    --compare docs/results/longctx_dev/misses.csv
```

trace 在 `traces/`（已 gitignore，每篇约 100 MB）。每篇旁边的 `.json` manifest 记录 prompt 哈希、生成的 token、停止原因、耗时、版本和数组哈希。

## 注意

- 不同硬件采集的 trace 不能混用（bf16 近似平局会翻转排名）。所有比较都在同一台 Mac 的 trace 之间做。
- 默认 bf16、prefill 分块 1024 token。所有 trace 用相同设置。
- 若 `prompt_tokens` 与 split 不符，先确认 `transformers==4.53.3`。
