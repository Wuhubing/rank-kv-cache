# 最终阶段计划（分支 `xiaoyang/score-ema-final`）

**目标**：一周内交 poster。主线是正面结果，但先在 Mac 上独立复现 Wuhubing 的 score-EMA 结论，复现成立才走正面主线。

**故事**：替换策略需要什么信息？
1. proposal 原题：selection rank 有信号，但策略层面无效（预注册负面结果，L4 已完成）。
2. 原因：v1 trace 只有排名，缺少"未选中驻留 page 的当前分数"。消融：rank / 当前分数 / EMA；EMA 只更新被选中 page 与更新全部驻留 page。
3. 解法：score EMA（每 page 一个浮点数）+ 按层分配，填上 LRU–Bélády 差距的约 28% / 40%。
4. 系统：融合 kernel 后开销与 LRU 相同、字节少 10–19%，batch 1 下延迟持平；按实测单 page 代价估算更大 batch 下的节省。

| 阶段 | 内容 | 设备 | 关口 |
|---|---|---|---|
| P0 | 新分支；held-out 协议修复（复核第 3 项）；本地采集脚本 | Mac | 测试全过 |
| P1 | `capture.py check` + 单篇 pilot，与 L4 同文档对比 | Mac | 三项自检 PASS；pilot 的 LRU–Bélády 差距与 EMA G 量级与 L4 相近 |
| P2 | 14 篇 dev 在 MPS 上采集并复现：EMA、分配+EMA、信息消融、rank 对照 | Mac | **EMA G 的 CI 下界 > 0.15 且多数文档优于 LRU → 正面主线；否则改写为负面结论** |
| P3 | 从未用过的新文档做 held-out：先冻结 split、配置与评估命令并提交，再采集、只评估一次 | Mac | 沿用 longctx_v1 的三条标准 |
| P4 | 开销模型；时间允许时 Colab 上做 batch 扫描 | Mac / Colab | 可选 |
| P5 | poster 与图；每个数字对应到结果文件 | Mac | — |

**新 held-out 的来源**：longctx_v1 的 16–32K GovReport/QMSum 候选池已全部用完（14 dev + 21 held-out）。候选方案（P3 前定）：超过 32K 的 GovReport/QMSum 截断到 16–32K，或加一个新任务（如 NarrativeQA 全文摘要）检验泛化。
