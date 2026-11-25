# 预注册：Mac（MPS）上的 held-out 检验

**状态：在采集任何 held-out trace 之前提交。** 每组只采集一次、按下面的命令只评估一次。

## 冻结的内容

- 冻结文件 `splits/heldout_frozen_mps.json`（sha256 `f4bf17a4b6645640ce31d58b2b9f0fe5b516cb2b38538e8780434602e05a8e33`），
  由 `python heldout.py freeze traces/mps_dev --out splits/heldout_frozen_mps.json` 只用 14 篇 MPS dev trace 生成：
  - EMA α：B=72/80/96/128/256 时分别为 0.4/0.4/0.5/0.7/0.95（与 L4 冻结值相同）；
  - 每个总 budget 下的按层 slot 分配（贪心，总量 28·B）；
  - B=96/128/256 的线性下次使用回归系数（两轮 DAgger）；
  - 复核：dev 上 B=96 的 LRU / EMA / Bélády miss 为 1,065,097 / 959,392 / 684,401，与 `docs/results/mps_dev` 一致。
- 采集设置与 dev 相同：`local/capture.py`，Qwen3-0.6B（rev c1899de），bf16，MPS，prefill 分块 1024，贪心解码，停止规则不变。
- 策略、指标和 bootstrap（按文档，2000 次，seed 0）与 `heldout.py` / `recompute_heldout.py` 现有代码相同。

## 数据与分组

**P（主检验，判定成功）**：`splits/longctx_v1.json` 的 21 篇 held-out（15 GovReport、6 QMSum），
split sha256 `1b4c763e09e48061d0605ee945ccd989e5ab082fcb7b85e488400a371449d590`。
这些文档团队已在 L4 上评估过（docs/heldout_results.md），但从未在这台 Mac 上采集，冻结配置也只来自 MPS dev。
所以 P 检验的是：**换硬件、重新从 dev 冻结之后，预注册结论是否仍然成立**；它不是对新文档的检验。

**F（新文档，单独报告，不判定成功）**：`splits/longctx_v2.json`（sha256 `e3bfe54c9d111c71e563d3eea24f6d143e91222c7fcd28b6781a3ed586f0a3bd`），
所有文档都不在任何早先的 split 中。
- F1 `heldout`：3 篇 GovReport，原文超过 32K token，按 LongBench 方式保留首尾、截掉中间，prompt 27.8–29.6K token。
  LongBench 里 GovReport/QMSum 剩下的长文档只有这 3 篇（QMSum 0 篇），所以只报告逐篇结果。
- F2 `heldout_ood`：8 篇 NarrativeQA，"总结整个故事"，新任务、新 prompt，prompt 16.5–30.0K token。

## 成功标准（P，B=96，与 longctx_v1 预注册相同）

1. score EMA 的 G 的 95% CI 下界 > 0.15；
2. score EMA 的 miss 少于 LRU 的文档 ≥ 18/21；
3. 按层分配 + EMA 相对 EMA 的 ΔG，配对 95% CI 下界 > 0。

三条全部满足，才在 poster 上称"在第二种硬件上通过了预注册检验"。任何一条不满足，就如实报告未通过的那条。
F1、F2 报告同样的量（G、CI、优于 LRU 的文档数、budget 曲线），F2 再按任务单独说明。

## 命令（每条只运行一次）

```bash
# 采集
python local/capture.py run --split splits/longctx_v1.json --subset heldout     --allow-heldout --out traces/mps_heldout
python local/capture.py run --split splits/longctx_v2.json --subset heldout     --allow-heldout --out traces/v2_fresh
python local/capture.py run --split splits/longctx_v2.json --subset heldout_ood --allow-heldout --out traces/v2_ood
# 评估
python heldout.py eval traces/mps_heldout --frozen splits/heldout_frozen_mps.json --split splits/longctx_v1.json --subset heldout     --out results/mps_heldout --procs 8
python heldout.py eval traces/v2_fresh    --frozen splits/heldout_frozen_mps.json --split splits/longctx_v2.json --subset heldout     --out results/v2_fresh    --procs 8
python heldout.py eval traces/v2_ood      --frozen splits/heldout_frozen_mps.json --split splits/longctx_v2.json --subset heldout_ood --out results/v2_ood      --procs 8
# 判定（只读 CSV）
python recompute_heldout.py results/mps_heldout/misses.csv
python recompute_heldout.py results/v2_fresh/misses.csv
python recompute_heldout.py results/v2_ood/misses.csv
```

采集中途断开时重跑同一条命令（已完成的文档会跳过），不算重新采集。采集失败的文档如实记录，不替换。
