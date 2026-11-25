# Dev 复现：M1 Max（MPS）对 L4

同样的 14 篇 longctx_v1 dev 文档、同样的选择器、prompt、停止规则和分析脚本（`efficient_dev.py`、`checks.py`），
只换采集硬件：Apple M1 Max，MPS，bf16，torch 2.15.0，transformers 4.53.3，prefill 分块 1024 token。
采集命令 `python local/capture.py run --subset dev --out traces/mps_dev`；每篇的 manifest 在 `manifests/`。

两边生成的 token 不同（硬件间 bf16 近似平局会翻转选择），所以这是一套独立的 trace，不是逐位复现。
14 篇里 13 篇在 EOS 结束，1 篇（gov_report/8ea6dbec…，879 步）触发 32-gram 循环停止；L4 上 14 篇都是 EOS。
解码步数 338–978（L4：473–1016）。

## B=96 主要结果（G 为填上 LRU–Bélády 差距的比例，按文档 bootstrap 的 95% CI；LODO 选参）

| 策略 | L4 miss / G | M1 Max miss / G | 优于 LRU 的文档 |
|---|---|---|---|
| LRU | 6.03% / 0 | 6.07% / 0 | — |
| Bélády | 3.86% / 1 | 3.90% / 1 | — |
| score EMA（z，更新全部驻留 page） | 5.42% / +0.279 [0.274, 0.284] | 5.47% / **+0.278 [0.275, 0.281]** | 14/14 |
| 按层分配 + EMA | +0.409 | **+0.400** | — |
| 当前分数（不平均） | +0.244 | +0.240 | 14/14 |
| InfLLM 式衰减求和（softmax，全部驻留 page） | +0.269 | +0.264 | 14/14 |
| 线性下次使用回归（DAgger） | +0.300 | +0.297 | 14/14 |
| EMA 只在被选中时更新 | −1.596 | −1.595 | 0/14 |
| EMA 用 rank 代替分数 | +0.016 | +0.016 | 14/14 |
| rank 策略（第一部分） | +0.027 | +0.030 | 14/14 |
| 衰减选中计数（不看分数） | +0.044 | +0.040 | 12/14 |

B=128：EMA +0.269 对 +0.269；分配 + EMA +0.540 对 +0.530。LODO 在 B=96 的 14 折里都选 α=0.5（与 L4 相同）。
按层分配的 LODO 方案拿到 oracle 分配收益的 90%（B=96，EMA）。

## 关口（docs/FINAL_PLAN.md，P2）

EMA 的 G 置信区间下界 0.275 > 0.15，14/14 篇优于 LRU：**通过**，按正面主线继续。
消融的结论也不变：有用的信息是"未选中驻留 page 的当前分数"，只有排名或只在被选中时更新都不行。
