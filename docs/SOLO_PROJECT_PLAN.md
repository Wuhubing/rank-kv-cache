# 6.5940 项目大纲与实验进度

**研究问题**　在固定模型、固定每步访问页面、固定 KV 容量时，哪种页面信息有助于减少重复加载？减少的传输量足以抵消策略自身的运行开销吗？

以课程目录中的 `6_5940_Proposal.pdf` 为准。原题是 *Can Selection Rank Improve KV Cache Replacement?*。原 proposal 的交付包括 trace 研究、单层 GPU cache-path benchmark、poster 和可复现代码。长上下文 score 实验作为有独立数据和指标的扩展。完整模型的 GPU 测试作为补充。独立分支从 `b6b75d5` 开始，保留原项目的代码、结果与贡献记录。

## 实验大纲

| 顺序 | 问题与方法 | 交付与验收 | 环境 |
|---|---|---|---|
| 1 · 验证起点 | 对照 proposal、两份 checkout 和本地归档。在真实 trace 上交叉检查 LRU、R+F、R+F+rank、Bélády。 | 跑测试、检查 oracle、重现四个 repeat-miss 计数，从已有逐层表重算长上下文主结果。已完成。 | Mac CPU |
| 2 · 从预测到驱逐 | 首先在个人 20 篇 dev 上跑 H1。固定过去 8 步的访问次数，比较本步 top/bottom quartile 在未来 8 步至少再出现一次的概率。再缩到实际发生驱逐时、已驻留且未入选的候选页，用上次入选时的 rank 比较。 | 先输出 pooled、任务 × 层组、有效文档数、可比较样本数、加权概率差和文档 bootstrap CI。标注末尾不足 8 步的样本。对同一批驱逐记录统计 rank 改变 LRU 决定的比例和改选后谁先复用。主分析与机制分析分表。[20 篇 dev 已完成](results/solo_rank_dev/README.md)。 | Mac CPU |
| 3 · 回答原 proposal | 在个人 dev 上复算 25 组 δ、β；核对已存锁定结果。按原有设计在个人 40 篇上重放。比较 FIFO、LRU、rank-only、R+F、R+rank、full 和 oracle。用固定 β 的成对实验量化 rank 的增量。 | B=128 主表、B=256 对照，B=64 做正确性检查。给出总 miss、首次和重复 miss、配对 CI、需求 MiB/token。原版 20/40 正式实验根据其冻结配置单列。补充原版 dense/sparse 的任务分数作为 selector 质量检查。[个人 20/40 篇已完成](results/solo_short_heldout/README.md)。 | Mac CPU |
| 4 · 验证更有用的信号 | 在 GovReport/QMSum 的 16–32K 长摘要 trace 上，比较历史 rank、当前 Quest 分数、只在入选时更新的 EMA、每步更新的 EMA。按层分配的权重只从 dev 学习；在相同总槽位下做 uniform/allocated × LRU/EMA 的 dev 对照。 | 先取得 1 篇开发 NPZ 与归档基准逐层对齐。随后跑 14 篇 dev；按原冻结协议重放 21 篇既有 held-out。报告 B=96 主结果及 B=72/80/96/128/256 曲线。表格写明数据来源、模型、prompt、生成长度与精度。 | Mac CPU 回放；原始 NPZ 无法取回时，用 Colab L4 按既有采集设置生成并单独记录 |
| 5 · 验证硬件收益 | 优先完成原题的 LRU 对 rank 单层硬件测试。沿用个人 checkout 中的 `gpu_layer_benchmark.py` 和对应 replay 语义，将入口和依赖移植到本分支。已有 L4 smoke 包覆盖 4 个文档，但其两个策略各有 0 次重复 miss；正式测试应覆盖有实际驱逐差异的层。 | 固定 trace、BF16 页、缓存初始化和 GPU，核对每次 miss 与回放一致、payload 与直接索引一致。预热后交错至少 3 轮，输出 wall、resolver、host pack、H2D、配对文档区间和原始逐步计时。EMA 的整模型融合数据作为单独对照。 | Colab L4；Mac 汇总 |
| 6 · 提交成品 | 串起 rank 的预测力、真正影响的驱逐次数、score 与分层预算的增益、GPU 实际耗时。 | Poster、最终报告、可复现入口。四张主图分别展示 H1、短 trace 策略、长 trace 预算曲线、配对 GPU 时间分解。图注明确“本次原始回放”“已有计数表重算”“既有 GPU 记录”“新 GPU 测量”。 | Mac |

### 固定比较口径

- 一个 page 含 16 个 token。每层每步必须访问相同的 64 个历史 page，当前未完成的 page 单独保留。每个策略都装入所有 miss，保护本步选择。B=128 指每层可驻留 128 个历史 page。
- repeat miss 指曾经加载、后来被换出、本步又需要加载的 page。原题成功线是 B=128 时相对 LRU 少至少 10% 的 repeat misses。H2 单独检验相对 R+F 节省的次数，其文档配对 95% CI 下界是否大于零。
- G = (LRU misses − policy misses) / (LRU misses − Bélády misses)，分母为零时记为 N/A。分层分配的 G 沿用 uniform LRU/Bélády 作为参考，同时报告每层容量之和 28B 和原始 miss 率。uniform Bélády 只在固定各层容量时构成 oracle 下界。
- 需求传输量按每次 miss 64 KiB 计算，另列实测 copy 时间和端到端时间。报告层级耗时以 µs/layer-step 为单位，整模型测试以 ms/token 为单位。
- 已发布的 held-out 结果属于复现。开发和选方法使用 dev；新方法若需要独立确认，先冻结规则，再评估新的未见文档。对短、长实验间重叠的 1 篇来源文档，保留已有剔除敏感性分析。

## 第一阶段实测 · 2026-11-24

在本地 Darwin arm64、Python 3.13.0 上完成。以下命令从本分支根目录执行。

```bash
python3 -m pytest -q
python3 validation/audit.py --colab-zip "$HOME/Downloads/heldout-20261120T043814Z.zip"
python3 recompute_heldout.py docs/results/longctx_heldout/misses.csv
```

| 输入与验证层级 | 本次结果 | 可支持的结论 |
|---|---|---|
| 实现测试、独立枚举、模拟器交叉比较 | 70 项测试通过；一条来自合成零差距样本的 NumPy warning。400 条小轨迹的 EMA 和两个 oracle 对齐独立参考，100 个小型分配问题与穷举一致。 | replay 基础行为通过已覆盖用例。 |
| 个人 40 篇原始 trace，本机重新运行替换策略 | B=128 的重复 miss 为 LRU 162,369、固定 β=2 的 R+F 161,850、加 δ=4 的 rank 161,712、Bélády 58,764。 | 固定频率权重后，rank 额外少 138 次，约为 R+F 的 0.085%。个人 20 篇 dev 采集用 A100、40 篇用 L4，两种策略的比较共享同一份 held-out trace。 |
| 仓库正式 rank 实验，已归档逐文档计数 | B=128，full 比 LRU 少 0.3862% 的重复 miss，95% CI [0.1963%, 0.5725%]；比单独调优的 R+F 少 330 次，CI [83, 600]。 | 达到 H2 的操作判据，低于 10% 实用目标。两种正式配置的 β 不同，固定 β 的增量另见上一行。 |
| 长上下文实验，已归档 21 篇逐层计数 | B=96 的重复 miss 比 LRU 少 11.633%（EMA）和 16.602%（分层 + EMA）。G 为 0.2814 和 0.4016；EMA 在 21/21 篇胜出。 | 支持该长摘要 workload 上的回放收益。此阶段尚未从 21 篇原始 NPZ 重算。 |
| 已有 L4 融合版整模型计时汇总 | B=96 时 EMA 和 LRU 六个配对的中位耗时差，平均 +0.030 ms/token，范围 −0.47 至 +1.22。 | 该 batch=1 测试没有显示稳定加速。准确的预热切片、显存和独立计时留给硬件阶段。 |

个人原始 ZIP 的 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `h2-development-20261120T045605Z.zip` | `0d0656528229a10f0cb034780bd39388b97548840434b7bf3dea26bd9f89e306` |
| `heldout-20261120T043814Z.zip` | `24e5cfc913352ac8b85451dc77f0e4f72052b76584f449ad82b1ec8cc05c82f3` |
| `gpu-layer-smoke-20261122T044930617682Z.zip` | `b6021c1e61f0c2a6f8cb880d8ff58a945c73d40947dc6caae87017244d9837c2` |

## 开发集上的驱逐例子

已定位第一个教学例子。个人开发 ZIP 的 `01.jsonl`（GovReport，ID 前缀 `eaf2acd9`），层 6、第 37 步，层和步均从零计。这是该层固定 β=2 的两种策略第一次驻留集合不同。此前均有 128 页，本步 55 次命中、9 次缺页。R+F 留下 page 447，加 rank 后留下 page 457。它们上次都在第 29 步出现，分别排第 62 和第 13。在随后的第 38–45 步，两页均未被选中。这次选了不同的页，在这 8 步里没有直接的复用收益。

20 篇开发文档的复用图、任务 × 层组比较、同状态驱逐对照和完整回放见[第二阶段结果](results/solo_rank_dev/README.md)。第 3 阶段已完成，完整策略在 B=128 相对 LRU 少 0.405% 的重复 miss，固定 β 后 rank 单独贡献 138 次；B=256 的增量区间跨零。[主表与图](results/solo_short_heldout/README.md)。第 4 阶段的[单篇原始 trace 回放](results/solo_longctx_pilot/README.md)已完成，112 行逐层计数全部对齐。下一步在全部 14 篇开发文档上按同一配置回放，以每篇文档配对汇报结果。长上下文正式重跑前，修复 `heldout.py` 的输入校验并从冻结 JSON 读取 α。
