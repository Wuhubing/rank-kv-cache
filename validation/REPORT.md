# Wuhubing 分支复核

复核基线为 `efficient-policy` 的 `083f758`，包含 `stage1-2` 的 `14c1578`。覆盖原 proposal 的 rank 实验、长上下文 score EMA、按层分配与 L4 benchmark。初始审计提交保留了上游的实验实现、参数和结果，新增可重跑的审计脚本。Olivia 的旁支有独立实现，本文仅用它说明固定频率权重的补充比较。

本文记录 `083f758` 的审计结果。后续提交 `7bc0f2c` 修复了第 1、2 项，并新增 [定向重跑入口](rerun_longctx_dev.py) 和 [Colab notebook](../notebooks/colab_rank_followup.ipynb)。

## 14 篇 dev trace 的修正后结果

在原始 `longctx_v1` 的 14 篇 dev trace 上重跑 `B=96`。LRU 和 Bélády 的基准计数通过核对，Colab 保存了 4,312 条逐层结果。Rank 策略按原实验的 leave-one-document-out 流程选择参数。

| 指标 | 修正前 | 修正后 |
|---|---:|---:|
| Rank 策略总 miss | 1,166,769 | 1,165,593 |
| 相对 LRU 的重复 miss 降幅 | 1.1408% | 1.2561% |
| 填补 LRU 到 Bélády 的差距 `G` | 2.7480% | 3.0257% |

修正后总 miss 减少 1,176 次，历史 rank 在这批 dev trace 上仍只带来小幅收益。历史 rank 的 AUC 从 0.41015 变为 0.41022，再限定有完整 32 步观察窗口的样本后为 0.41144。在同一完整窗口上，当前分数和 EMA 分数的 AUC 分别为 0.67481 和 0.69754。`last_rank` 数值越小代表排名越高，所以其方向应反过来解释。

原始输出为 `summary.json`、`rank_by_layer.csv`、`rank_lodo.csv` 和 `auc.csv`，保存在 `My Drive/rank-kv-followup/full-20261124-044821/`。 [已运行的 Colab notebook](https://colab.research.google.com/drive/1x6V5ABv6rou0Ep4zrUQW_DavuN6iVt05)

## 可以复现的结果

| 项目 | 审计结果 | 证据 |
|---|---|---|
| CPU 测试 | `pytest` **65 项通过**。smoke test 有一条 `All-NaN slice` 警告，发生在合成样本的零 oracle 差距设置。 | 两套 replay、trace schema、split、H1、模型辅助逻辑、长文本模拟器和 dev 端到端 smoke。 |
| Bélády、EMA 和精确分配 | **400** 条小轨迹上，两个 Bélády 实现的 miss 数均等于枚举最优，EMA 与独立实现一致。**100** 个小型分配问题上，精确分配与穷举一致。 | 随机种子固定并包含分数平局。 |
| 原 proposal 的 rank 策略 | 从逐文档结果重算了 dev、held-out 和两套截断版的主结果及配对 CI。正式 held-out 只比 LRU 少 **0.3862%** 重复 miss，截断后少 **0.3120%**。完整策略对 R+F 少 **330** 次，95% CI **83–600**。 | 20/40 划分、manifest 中的文档 ID、split 哈希、调参表和冻结权重一致。核对了访问数、首次 miss、oracle 下界及每 token 传输量。 |
| 同一策略跑另一套真实 trace | 用 `stage1-2` 的回放实现读取 Xiaoyang 的原始 40 篇 Colab export。在 `B=128`，LRU / Bélády / 固定 `β=2` 的 R+F / 加 `δ=4` 的 rank 分别有 **162,369 / 58,764 / 161,850 / 161,712** 次重复 miss，与原结果完全一致。 | 在同一批 trace 和同一组参数上跨实现验证。Rank 单独增加后少 **138** 次重复 miss。 |
| 长文本 dev | `B=96` 上固定 EMA 的 `G=0.2790`，逐文档留一法的层分配结果 `G=0.4088`。重新汇总每个已选配置的逐层结果，与 `lodo.csv` 中的 miss 数一致。 | 从逐层 CSV 核验冻结配置。`B=96`、`B=128` 的分配可从已保存的训练曲线重算。其他三个预算的完整 α 网格未提交，验证了槽位总数与记录的一致性。 |
| 长文本 held-out | **22,344** 条逐层记录与逐文档表一致。`B=96` 时 EMA 的 `G=0.2814`，95% CI **[0.2773, 0.2852]**，**21/21** 篇胜过 LRU。加分配后 `G=0.4016`，比 EMA 高约 **0.120**，配对 CI **[0.1127, 0.1284]**。相对 LRU 的重复 miss 降幅是 **11.63% / 16.60%**。 | 从保存的 miss 计数独立重算 bootstrap 和全部五个预算的曲线。分配与均匀预算使用相同总槽位数。 |
| L4 保存的 benchmark 汇总 | 融合版记录中，LRU 和 Bélády 的 miss 数与模拟器完全一致，EMA 类最大绝对差 **39** 次。`B=96` 两轮共六个配对，EMA − LRU 的每轮中位延迟差平均 **+0.030 ms**，范围 **−0.47 至 +1.22 ms/token**。`B=96/128` 按 miss 数计算的需求传输量分别少 **9.33% / 9.77%**，加分配后少 **13.15% / 19.18%**。 | 从提交的 benchmark CSV 重算。字节是 miss × 每页 64 KiB，属于需求 KV 传输估计。|

`G` 表示填补 LRU 到 Bélády 差距的比例。上表 `G=0.4016` 和重复 miss 下降 16.60% 使用不同分母。按层分配改变了各层预算，所以逐层 oracle 下界仅对相同层预算的策略直接适用。

## 确认的问题

### 1. 长文本分支的 rank baseline 少了同一步 hit 的平局判定

原 `replay.py` 把有效时间相同的候选按 `(last_step, was_hit, -page_id)` 决定去留。`kvpolicy.RankFull.key()` 缺了 `was_hit`。因此 `delta=beta=0` 时，长文本分支里的 `RankFull` 有时与 LRU 不同。固定种子的 400 条小轨迹里，两个实现产生不同 miss 数的次数分别是 `(0,0)` **35**、`(0,2)` **4**、`(4,2)` **7**、`(8,2)` **1**。

**影响**集中在长文本实验的 `rank(stage1-2)` 对照。原 `stage1-2` 的正式结果使用 `replay.py`。长文本 held-out 的 EMA、allocation、LRU 和 Bélády 使用各自的策略实现。建议在后续修复时恢复共同 tie-break，并重算长文本 rank 对照。 [原 replay](../replay.py) · [长文本策略](../kvpolicy.py)

### 2. 长文本 dev 的 AUC 对分数平局处理错误

`efficient_dev.auc()` 对相同的分数照样分配不同名次。两个样本的分数完全一样，交换正负标签的顺序会使函数返回 **0 或 1**，正确答案两种顺序均为 **0.5**。离散的 rank、recency、frequency 特征会大量同分，因此 `gap.csv` 中这些特征的 AUC 需要用 tie-aware 实现重算。`gap_analysis.py` 已有正确的平均名次实现。该 AUC 用于机制说明，EMA 的主要 miss 数不依赖它。 [现有实现](../efficient_dev.py) · [tie-aware 参考](../gap_analysis.py)

另外，长文本的“32 步内复用”诊断把最后 32 步中未再出现的页记为 negative，其中有些页没有完整的观察窗口。原 H1 会排除未来窗口不足的样本。建议长文本 AUC 诊断也排除这些样本，或者明确注明终止截尾。此项不改变已记录的实际 trace miss。

### 3. Held-out 驱动没有强制执行冻结协议

`heldout.py eval` 只用文件名检查 dev 重叠。复核脚本构造了一个带真实 dev `rid`、只有一层的合成 NPZ，改成 `renamed.npz` 后程序照常生成 38 行结果。驱动也从模块常量取 EMA α，传入的冻结 JSON 中的 `alpha` 不控制实际运行。

**已提交的结果仍然通过文档 ID 和预算核对。** 长文本的 dev/held-out 在各自划分内没有共享文档或 context hash，冻结 JSON 中的 α 也与当前代码一致。后续复现应校验 NPZ 内的 `rid`、唯一性、完整的预注册 ID 集合、28 层与 top-K 维度，读取冻结 JSON 的 α，并记录原始 trace 的哈希。 [驱动](../heldout.py) · [冻结配置](../splits/heldout_frozen_v1.json)

### 4. GPU 内存和 warm-up 的结论需要缩小范围

- `run(..., warmup=8)` 的 `warmup` 参数未被读取。函数返回所有 decode 步的延迟。外部 Colab 汇总可能切掉了 warm-up，但逐步数据与汇总驱动没有随分支提交，因此无法核实切片。
- `reset_peak_memory_stats()` 时，prefill 的 `out.past_key_values` 仍持有完整 GPU cache。最后一层的 `kf` / `vf` 也仍有引用。当前 `peak_mem` 会计入这些本应释放的 prefill 张量。要报告纯 decode 显存，需要先释放引用，再记录 warm-up 后的稳态和峰值。
- 未融合 benchmark 的 `86.6 ms` 来自**三个文档各自中位数的均值**。直接取三个文档中位数的中位数约是 `86.24 ms`。融合版只有三个文档，LRU 和 EMA 各有两轮。保存的结果支持“这个设置没有观察到稳定的加速”。正式的等效性界限、其他 batch size 或其他 GPU 的结论需要另外测量。 [GPU 实现](../gpu_bench.py) · [原报告](../docs/gpu_benchmark.md)

## 方法论判断

- **原研究问题与实际目标分开。** H1 的预测结果和两套固定权重实验均支持历史 rank 有小信号。正式 H2 对照同时改变了 rank 和频率权重，所以 **330 次**描述的是两个调优配置的差，而 rank 单独的增量应引用固定 β 的实验。原 proposal 的 **10%** 目标由同 trace、同预算的 LRU 对照直接判断，已确认未达到。
- **H1 的一个措辞要精确。** `candidate` 包含所有曾入选且当前不在 top-K 的页，其中有些已经离开 GPU。`lru` 样本只看 LRU 实际仍驻留的页，但两者都包含没有发生驱逐的步。特别是 **0.73%** 属于前者，并非只在实际发生换出时的候选占比。实际 LRU 驻留样本中对应占比约 **0.74%**，8 步效应约 **+13.01 个百分点**。随结果提交的 H1 表没有 proposal 所列的 task × layer group。 [H1 实现](../analysis.py) · [正式结果](../docs/results/heldout/eval/h1_effects.csv)
- **长文本验证为何仍有说服力。** 在新实验中，选择器、访问序列和预算都匹配。EMA 只读当前和过去分数，Bélády 单独读未来。Git 历史显示 split、主要判据和冻结配置先于结果提交。短文本与长文本两个实验的 held-out 之间有 **1 篇**相同源文档，提示词与 trace 不同。把它排除后，EMA 的 `G=0.2817`，加分配后 `G=0.4004`，仍有 **20/20** 篇优于 LRU。把结果表述为这套 Quest 长摘要 workload 的发现，符合证据范围。现有 InfLLM 对照用 Quest bounds 的 softmax 代理 attention mass。
- **Sparse vs. dense 的质量检查。** 按任务用的 F1、ROUGE-L 和对多个参考答案取最大值与 LongBench 的评分规则相符。结果来自一组按文档配对的评分，适合 sanity check。通篇解码相似或 sparse 与 dense 等效，需要独立指标和验证。 [LongBench 评分代码](https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py) · [项目质量汇总](../docs/results/heldout/longbench/longbench_summary.json)

## 未能独立复现的部分

短文本 JSONL、长文本 dev/held-out NPZ 和 GPU 逐步计时未随 Git 提交。长文本 dev NPZ 已从共享 Drive 获取，并用于上面的 Colab 重跑。GPU 归档也已找到，模型采集、逐页 payload 和 token 对照以及修正测量口径后的硬件计时仍待独立复现。报告里的三条长文本主要结果可以从提交的逐层 miss 表重算。 [GPU benchmark 归档](https://drive.google.com/drive/folders/125FYYpCIZYnJJK5wBuOWBW9B6qutw_cv)

## 复现命令

```bash
python3 -m pytest -q
python3 validation/audit.py
python3 recompute_heldout.py docs/results/longctx_heldout/misses.csv
python3 validation/audit.py --colab-zip /path/to/heldout-20261120T043814Z.zip
```

最后一条使用 Xiaoyang 的原始 Colab 导出文件。沿用上游结果和参数时，其余三条可以在没有 GPU 或 raw trace 的情况下运行。脚本打印几个已知问题的复现实验，便于后续修复前后对照。
