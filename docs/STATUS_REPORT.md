# 项目近况报告：KV Cache 替换策略（rank 检验 → 高效替换策略）

**截至 2026-11-23 · 分支 `efficient-policy`（draft PR #3，叠在 `stage1-2` / PR #2 之上）**

本报告分两部分：

- **第一部分（§1–§8）：selection rank 能否改进替换。** 按 proposal 完成正式检验：20 篇 dev 调参并预注册（`2f6a9bd`），40 篇 held-out 采集一次、评估一次。结论是负面的。
- **第二部分（§9）：新主线"高效替换策略"。** 在更长的 trace 上找极小、因果、低代价的替换策略。21 篇 held-out 预注册检验，三条标准全部通过；之后在 L4 上做了真实卸载 cache 的延迟测试。

每一步做了什么、用什么实现、得到什么数据、每条结论怎么推出来的，都写在对应小节里。第一部分的原始输出在 [`docs/results/`](results/) 下；第二部分的详细文档是 [`efficient_policy_dev.md`](efficient_policy_dev.md)、[`heldout_results.md`](heldout_results.md)、[`gpu_benchmark.md`](gpu_benchmark.md)。两部分的复现命令分别在 §8 和 §9.8。

---

## 0. 一页摘要

### 0.1 新主线：高效替换策略（§9）

**正式结论（21 篇 held-out，预注册，只评估一次）：一个每个 page 只存一个浮点数的 score EMA 策略，稳定地填上了 LRU 到 Bélády 差距的 28%；再加上按层分配 slot，能填上 40%。但在 L4 + Qwen3-0.6B 上，省下的传输换不成延迟。**

1. **为什么换主线。** 第一部分发现 rank 没用；消融（§9.4）说明原因是 trace v1 只记录了排名，丢掉了未选中 page 的当前分数，而这个分数恰恰是驱逐最需要的信息。
2. **新设定。** trace v2 记录每一步所有历史 page 的 Quest 分数；上下文 16–32K token；带思考的长摘要，生成 473–1016 步。主 budget 是 B=96（B/K=1.5），此时 LRU 的 miss rate 约 6%，Bélády 约 3.9%。
3. **held-out 结果（B=96）**：

   | 标准 | 阈值 | 结果 |
   |---|---|---|
   | score EMA 的 G，CI 下界 | > 0.15 | **G 0.281**，CI [0.277, 0.285] |
   | EMA 优于 LRU 的文档数 | ≥ 18/21 | **21/21** |
   | 分配 + EMA 显著优于 EMA | 配对 CI > 0 | **G 0.402**，ΔG +0.120 [0.113, 0.128] |

   dev 和 held-out 几乎一致（EMA 0.279 对 0.281，分配 + EMA 0.409 对 0.402），策略排名不变。
4. **定位。** score EMA 本质上是"InfLLM 的衰减分数规则，用在所有驻留 page 的 Quest 上界分数上"，不是新规则。贡献在于 page 级的系统比较、衰减随 budget 的扫描、找出所需信息的消融，以及按层分配和分数规则可以叠加。
5. **GPU 延迟（L4）。**
   - 真实卸载 cache 选出的 page 和生成的 token 都与 trace 完全一致，miss 数与模拟器吻合。
   - PCIe 流量减少：EMA 少 9–10%，分配 + EMA 少 13–19%。
   - **延迟没有改善**：LRU 86.6 ms/token，EMA 93.1 ms/token，Bélády 87.3 ms/token。这个规模下真正的 PCIe 时间不到 1 ms，时间主要花在每层的固定开销上。
   - 后续测试换成免同步、zero-copy 的实现，延迟仍然不变，瓶颈是 CPU 派发。只算 GPU 实际忙的时间，Bélády 也只值约 3%，EMA 不到 1%。要让 miss 减少变成延迟收益，最直接的办法是加大 batch（未实测）。

### 0.2 第一部分：selection rank（§1–§8）

**正式结论（40 篇 held-out，按预注册执行一次）：selection rank 不能实质改进 KV cache 替换。proposal 的主成功标准（S1）不满足。**

1. **缓存确实有改进空间。** held-out 上 B=128 时 LRU 的 miss rate 是 4.05%，Bélády 是 2.97%，绝对差 1.08 pp、相对差 26.7%；Bélády 比 LRU 少 65% 的 repeat miss。dev 上是 4.13% 对 3.01%，几乎一样。
2. **两条成功标准（B=128，held-out）：**
   - **S1 不满足**：完整策略（dev 调出 δ=8、β=2）只比 LRU 少 **0.39%** 的 repeat miss，CI [0.20%, 0.57%]，目标是 10%。
   - **S2 在统计上满足，但效应极小**：完整策略比 R+F 少 330 次 repeat miss，配对 CI [83, 600] 整体大于 0，只占 R+F repeat miss 的 0.22%。rank 项确实有作用，只是小到没有实际意义。
3. **H1：rank 对驱逐候选有真实但很小的增量信号。** 控制 frequency × recency 后，驱逐候选上 top 与 bottom 四分位的复用率差为 +3.1 pp（1 步内）到 +10.8 pp（8 步内），40 个文档 cluster 的 CI 都在 0 以上。但候选中 top 四分位只占 0.73%，所以 rank 加分几乎改变不了驱逐决策。
4. **非 rank 特征能挽回的也有限。** 在 dev 上训练、系数固定的逻辑回归在 held-out 上 G = **11.9%** [6.7%, 17.2%]，repeat miss 少 7.8%。收益几乎全部来自 MultiFieldQA（G 26%，GovReport 只有 2.4% 且 CI 跨 0）；截断 MFQA 的重复回答后，G 降到 2.4% [−1.2%, 6.0%]。
5. **LongBench sanity check：** 稀疏参考运行比 dense 略差（GovReport ROUGE-L 17.4 对 18.7；MFQA F1 11.6 对 12.4），说明稀疏 trace 是 dense 解码的合理近似。
6. **对照蓝图的决策关口：** Gate A 通过；Gate B 通过（信号真实，但弱且罕见）；**Gate C 不通过**。结论应按蓝图预设的负面结论加现象刻画来写。

---

## 1. 背景：proposal 与蓝图

- **Proposal**：在 Qwen3-0.6B + Quest 式 page 选择（16-token page，K=64）下，检验 page 上次被选中时的排名能否在 recency 和 frequency 之外预测复用（H1），以及把 rank 加进 recency+frequency 策略能否减少 miss（H2）。打分公式为 sᵢ = τᵢ + δ·1{rᵢ ≤ ⌈K/4⌉} + β·min(fᵢ, 4)。成功标准：held-out 上 B=128 时 repeat miss 比 LRU 少至少 10%；并且相对 δ=0 的增益，配对 95% CI 整体大于 0。
- **蓝图**：在 proposal 上加了"先验证再投入"的顺序：A 文献核实 → B 信号验证（H1）→ C 策略验证（H2）→ D 系统 benchmark，每阶段末尾设决策关口；并预先写好正面和负面两种结论的说法。

---

## 2. 经过（时间线）

| 时间 | 步骤 | 产出 / commit |
|---|---|---|
| 11-19 | 已有的起点：回放器（LRU、Bélády）和 Colab pilot notebook | `7c73d51`、`9da5c9a` |
| 11-22 13:35 | **Phase A 文献核实**：读 ArkVale、HiSparse 全文和 ArkVale 的 kernel 代码，概览 Quest、H₂O、PARROT | `5848a7f` [`related_work_matrix.md`](related_work_matrix.md) |
| 13:35 | **H2 回放策略**：在 `replay.py` 中加入 FIFO、rank-only、打分策略（R+F、R+rank、完整策略） | `e11e652` |
| 13:39 | **H1 分析**：`analysis.py`，分层四分位差加聚类 bootstrap | `ff24fc6` |
| 13:40 | README 更新；团队决定生成长度统一为 128 | `78927c6`、`0f35894` |
| ~13:50 | **GPU pilot**：在 Colab 的 A100 上跑一篇 GovReport（截成 4K），127 步 × 28 层 | 结果在 [`results/pilot/`](results/pilot/) |
| 14:02 | 发现原 H1 口径测的不是驱逐候选，**改口径**：新增 H1-candidate 和 LRU-candidate | `0946844` |
| 14:06 | **冻结 60 篇划分**（20 dev / 40 held-out），notebook 改为批量采集 | `6f269a0` [`splits/longbench_v1.json`](../splits/longbench_v1.json) |
| ~14:06 | **采集 8 篇 dev 文档**（A100，两个任务 × 两个长度段 × 2 篇） | 结果在 [`results/dev_pilot/`](results/dev_pilot/) |
| 14:07 | 按任务、长度段分组报告，加入 top 四分位占比 | `1af9d98` |
| 14:15 | 发现 MFQA 答完后反复输出答案，加入截断工具 | `1daecbd` |
| 14:31 | **差距分析**：坏驱逐、特征画像、可学习策略的上界 | `b926dd5` |
| 14:55 | notebook 改为从固定 commit 读取划分文件、按 `SUBSET` 选文档；manifest 记录标准答案和 dense 生成 | `a19ada8` |
| 14:58 | **第二轮文献核实**：InfLLM、Scissorhands、H₂O 附录、HiSparse 与 Quest 代码、PARROT；补进矩阵 | `d529def` [`related_work_matrix.md`](related_work_matrix.md) |
| ~15:06 | **采集全部 20 篇 dev**（L4）；发现 A100 与 L4 的选择结果不同（见 §4.7） | 结果在 [`results/dev_full/`](results/dev_full/) |
| 15:13 | **在 20 篇 dev 上调参并预注册**：完整策略 δ=8、β=2，R+F β=4，R+rank δ=4，逻辑回归系数固定；评估脚本随预注册一起提交 | `2f6a9bd` [`preregistration.md`](preregistration.md) |
| ~15:24 | **采集 40 篇 held-out**（L4，sparse trace + dense 生成），随后按预注册**只评估一次** | 结果在 [`results/heldout/`](results/heldout/) |
| 11-22 晚 | 团队决定换主线到"高效替换策略"，不使用 RL；原 kvrl 蒸馏方案放弃（`distill_kvrl.py` 未跟踪、未使用） | — |
| 11-23 00:11 | **trace schema v2**：每步记录所有历史 page 的分数；`replay.py` 兼容 v1/v2 | `db51faa` |
| 00:39 | **长 trace pilot**：一篇 26K QMSum，741 步；确定 prompt 和停止规则 | `537a196` [`long_trace_pilot.md`](long_trace_pilot.md) |
| 00:47 | **冻结 `longctx_v1` 划分和 budget**（14 dev / 21 held-out，主 budget B=96） | `950cb37` [`splits/longctx_v1.json`](../splits/longctx_v1.json) |
| 01:06–02:44 | **策略族模拟与 dev 研究**：`kvpolicy.py`、`efficient_dev.py`、`checks.py`；EMA 消融、稳健性检查、相关工作 baseline（InfLLM 式、KVDrive 式、knapsack、HiSparse tie 规则） | `ab1921e`–`056c3ad` [`efficient_policy_dev.md`](efficient_policy_dev.md) |
| 11:35 | held-out 预注册标记为已批准并冻结 | `bb0ce87` |
| 12:34 | **只用 dev 冻结**按层分配和线性模型系数，在读到任何 held-out 输出之前提交 | `f87e982` [`splits/heldout_frozen_v1.json`](../splits/heldout_frozen_v1.json) |
| 12:43 | **采集 21 篇 held-out，按预注册只评估一次**：三条标准全部通过 | `91f7041` [`heldout_results.md`](heldout_results.md) |
| 13:29 | **GPU 延迟测试**：L4 上的真实两级卸载 cache | `f8ca5ab` [`gpu_benchmark.md`](gpu_benchmark.md) |
| 13:48 | **GPU 后续测试**：免同步、zero-copy 实现，以及 GPU 忙碌时间分析；延迟仍不变，瓶颈是 CPU 派发 | `e327cfb` [`gpu_benchmark.md`](gpu_benchmark.md) |

GPU：pilot 和 8 篇 dev pilot 用的是 NVIDIA A100-SXM4-40GB；20 篇 dev（`dev_full`）和 40 篇 held-out 都用 NVIDIA L4。第二部分的全部采集和延迟测试也都在 L4 上。其余设置相同：bf16，torch 2.14.1+cu130，transformers 4.53.3，贪心解码，batch=1。

---

## 3. 实现

| 文件 | 作用 |
|---|---|
| [`notebooks/first_trace.ipynb`](../notebooks/first_trace.ipynb) | Colab 采集。`QuestPages` 是自定义的 attention 后端：对每个历史 page 存 post-RoPE key 的逐坐标 min/max，用 query 算上界分数，在 query head 上取平均得到每层一个排名，选 top-64 个 page 加上当前 page 做 attention。每步每层记录排好序的 page id。内含选择器自检（与暴力计算比对）。 |
| [`replay.py`](../replay.py) | 核心只用标准库。对每个 (request, layer) 流独立回放：LRU（同步命中排在新填入之前，再按 page id 打破平局）、FIFO、rank-only、打分策略 `score:δ:β`、Bélády。所有策略都保护当前选择集、每个 miss 都放入缓存。 |
| [`analysis.py`](../analysis.py) | H1 分析。三种口径：selected（原口径）、candidate（之前被选过、本步没被选中的 page）、lru（B=128 时 LRU 缓存中实际驻留且本步没被选中的 page）。在 recency × frequency 分层内算 top 与 bottom 四分位的复用率差，按层大小加权，按文档聚类做 bootstrap。 |
| [`split.py`](../split.py) | 确定性抽样（seed 6594010）：每个任务 × 长度段 15 篇，其中 5 dev / 10 held-out；pilot 文档在 dev 里。 |
| [`truncate_trace.py`](../truncate_trace.py) | 在 MFQA 的第一个换行处截断 trace，去掉重复的答案尾部。 |
| [`gap_analysis.py`](../gap_analysis.py) | 差距分析：在 LRU 的每个驱逐决策上标出坏驱逐；计算候选特征及其 AUC；按文档留一交叉验证训练逻辑回归、深度 3 决策树和启发式规则，并把它们当作驱逐策略回放。 |
| [`preregister.py`](../preregister.py) | 在 20 篇 dev 上按 proposal 规则调 δ、β，训练并固定逻辑回归，写出 [`splits/preregistration.json`](../splits/preregistration.json)。 |
| [`evaluate.py`](../evaluate.py) | 严格按预注册配置评估一个运行目录：各策略、各 budget 的 repeat/total miss、G、H2D 字节/token，两条成功标准，按任务和长度段分组，H1 三种口径；CI 都是按文档的 bootstrap。 |
| [`longbench_score.py`](../longbench_score.py) | LongBench 官方指标（MFQA 用 `qa_f1_score`，GovReport 用 ROUGE-L F）给稀疏参考生成和 dense 生成打分，并做配对 bootstrap。 |
| `tests/` | 41 个单元测试（系统 Python 下跳过 2 个需要 numpy 的）：回放的 tie 规则、Bélády 与穷举最优一致、δ=β=0 与 LRU 逐步一致、H1 样本定义、合成信号能被检出、划分确定性、notebook 配置与划分文件一致且不含 held-out ID、截断位置、成功标准计算、LongBench F1、逻辑回归系数的导出与读入等。 |

---

## 4. 数据与结论推导

### 4.1 Phase A：文献核实 → Gate A 通过

**做法**：分两轮。
- 第一轮：读 ArkVale（NeurIPS 2024）和 HiSparse（arXiv:2608.07009）全文，以及 ArkVale 的驱逐 kernel（`select.cu`）；概览 Quest、H₂O、PARROT。
- 第二轮（补齐未核实条目）：读 InfLLM、Scissorhands 全文，H₂O 的附录，HiSparse 在 SGLang 中的 kernel 代码，Quest 的代码，以及 PARROT 的相关章节。

**发现**：
- **ArkVale** 的驱逐只用**当前步**的 query 分数（`select.cu` 第 127、131 行），没有 recency、frequency 或历史排名状态；camera-ready 中也没有驱逐策略消融、oracle 或 miss rate。
- **HiSparse** 的设置和我们几乎相同（固定选择、host KV、B 个 slot、对比 FIFO/random/LRU/Bélády），但用的是纯 LRU，并明确把 LRU 与 Bélády 之间的差距留作"预测式替换策略能挽回的空间"（§4.3）。代码显示即使搭配 Quest，它的缓存和 LRU 仍按 **token** 管理。Fig. 6 的数字全部复核无误。
- **InfLLM** 是最接近的已有缓存策略：KV 块放在 CPU、按需召回，GPU 缓存用"按注意力加权、带衰减的频率分数"驱逐（论文称之为 LRU），并在 GovReport 上报告了 miss rate、和 FIFO/Random 比较过。但它不用选择分数或排名，也没有和 Bélády 比。
- **Scissorhands** 是最接近 H1 的说法（"过去重要的将来也重要"），但证据只是前后两半句子的重合率，不是逐步预测，也没有控制 recency 和 frequency。**H₂O** 从未测量 heavy hitter 在解码步之间是否稳定。
- **Quest** 的代码按每个 query head 各自选 page，我们把各 head 的分数平均成一个排名，所以应称为"Quest 式"。

**推导**：没有论文在选择固定的前提下，用"上次观测到的 top-K 排名"决定 page 是否留在缓存，也没有论文检验它在 recency 和 frequency 之外的预测力 → Gate A 通过。但只能声称增量问题，不能说"首个基于重要性的驱逐"，也不能说"首个用过去重要性做可召回缓存替换"（InfLLM 已经做过）。完整矩阵和"必须避免的说法"见 [`related_work_matrix.md`](related_work_matrix.md)。

### 4.2 Pilot：一篇 4K 文档

GovReport 第 0 篇截成 4,096 token，127 步 × 28 层 = 3,556 行。

| B | 策略 | miss rate | repeat miss |
|---|---|---|---|
| 64 | LRU = Bélády | 14.83% | 28,967 |
| 128 | FIFO | 4.09% | 4,538 |
| 128 | rank-only | 4.49% | 5,437 |
| 128 | LRU | 2.94% | 1,909 |
| 128 | 完整策略（δ=4，β=2） | 2.94% | 1,907 |
| 128 | Bélády | 2.36% | 582 |
| 256 | LRU = Bélády | 2.10% | 0（只剩 compulsory） |

- B=64=K 时所有策略相同，B=256 时只剩 compulsory miss，模拟器的检查通过。
- **问题出现了**：原口径 H1 效应很强（下一步内 +27.1 pp），但完整策略只少 2 次 miss。

### 4.3 H1 口径修正：为什么原口径不对

**原口径**比较的是"本步被选中的 page"里排名高和排名低的谁更可能再被选中。pilot 中 top 四分位下一步再被选中的概率是 99.4%：这些 page 几乎每步都在选择集里，本来就受保护，LRU 从不驱逐它们。

**驱逐只发生在"在缓存里、但本步没被选中"的 page 之间**，所以 H1 应该在这批 page 上检验。pilot 上改口径后：

| 口径 | 下一步内 | 8 步内 |
|---|---|---|
| selected（原口径） | +27.1 pp | +6.4 pp |
| candidate | +2.0 pp | +4.5 pp |
| LRU-candidate | +4.6 pp | +8.3 pp |

**推导**：一旦控制 recency 和 frequency，rank 在驱逐候选上只剩几个百分点的增量信号 → 这解释了完整策略为什么几乎无效。

### 4.4 Dev 8 篇：结果是否稳定

**划分**：60 篇按 source document 冻结，20 dev / 40 held-out；本次只采集其中 8 篇 dev。

| 任务 | 长度段 | prompt token 数 |
|---|---|---|
| GovReport | 4–8K | 4,388；7,694 |
| GovReport | 8–16K | 11,022（pilot 那篇，未截断）；8,425 |
| MultiFieldQA-en | 4–8K | 5,163；6,362 |
| MultiFieldQA-en | 8–16K | 15,631；9,085 |

每篇都生成满 128 token（127 个稀疏步），合计 1,820,672 次历史 page 访问。

**B=128 回放（全部 8 篇）**

| 策略 | miss rate | repeat miss | 相对 LRU |
|---|---|---|---|
| FIFO | 6.03% | 65,335 | +89% |
| rank-only | 6.33% | 70,851 | +105% |
| LRU | 4.34% | 34,579 | — |
| R+F（δ=0，β=2） | 4.33% | 34,474 | −0.30% |
| R+rank（δ=4，β=0） | 4.34% | 34,504 | −0.22% |
| 完整策略（δ=4，β=2） | 4.33% | 34,424 | **−0.45%** |
| Bélády | 3.09% | 11,837 | **−66%** |

compulsory miss 为 44,450 次，占全部访问的 2.44%，任何策略都无法避免。

**分组看（B=128，LRU / Bélády 的 miss rate）**

| 分组 | LRU | Bélády | 差距占比 |
|---|---|---|---|
| GovReport | 4.43% | 3.20% | — |
| MultiFieldQA-en | 4.25% | 2.98% | — |
| 4–8K | 3.66% | 2.63% | 41% |
| 8–16K | 5.02% | 3.56% | 59% |

两个任务几乎相同；长文档差距更大。

**δ 扫描（β=0，B=128）**：δ = 1/2/4/8/16/32 时 repeat miss 分别为 34,541 / 34,527 / 34,504 / 34,490 / 34,542 / 34,604（LRU 是 34,579）。最好的 δ=8 也只少 0.26%，δ 再大反而更差。

**H1（按文档聚类，8 个 cluster）**

| 口径 | h=1 | h=2 | h=4 | h=8 | top 四分位占比 |
|---|---|---|---|---|---|
| selected | +28.4 [27.2, 29.7] | +18.8 | +12.0 | +7.9 [7.4, 8.4] | 25% |
| candidate | +2.5 [1.7, 3.9] | +4.5 [2.7, 7.1] | +7.7 [4.3, 12.2] | +11.6 [6.7, 17.7] | **0.8%** |
| LRU-candidate | +3.6 [2.4, 5.4] | +6.8 [4.2, 9.9] | +11.7 [6.8, 16.6] | +15.9 [8.8, 23.1] | 0.8% |

单位为 pp（控制 frequency × recency 后的 top 减 bottom 复用率差），方括号是 95% CI。各分组中 top 四分位占比都在 0.4%–1.0% 之间，长文档上也没有变大。

**推导**：
- candidate 口径的 CI 都在 0 以上，**rank 的增量信号是真实存在的**（Gate B 勉强通过）。
- 但排名高的 page 几乎从不掉出选择集（只占候选的 0.8%），给它们加分几乎改变不了驱逐决策 → 策略层面无效（Gate C 不通过）。

### 4.5 MFQA 重复问题与截断

贪心解码下，MFQA 答完第一行后会反复输出 "Answer: ..."。4 篇 MFQA 的第一个答案分别在第 22、13、18、72 步结束，之后的步骤都在重复，让 page 复用看起来更多、更好预测。`truncate_trace.py` 在第一个换行处截断（GovReport 不受影响）。

截断后 B=128：LRU 的 miss rate 是 5.08%，Bélády 是 3.95%，绝对差 1.13 pp、相对差 22.2%；repeat miss 18,957 对 6,145；完整策略 18,843（−0.6%）。H1-candidate 为 +2.1 到 +7.6 pp，top 占比 0.9%。**结论不变。**

### 4.6 差距分析：那 66% 的差距由什么构成

**LRU 的坏驱逐**（B=128，全部 8 篇）
- 共 14,730 次驱逐决策、50,636 个被驱逐的 page，其中 **68.2% 是坏驱逐**，CI [64.8%, 71.2%]；每篇文档在 60%–75% 之间（截断后 62.3%）。
- 31.7% 的被驱逐 page 之后再也没被选中，这些驱逐是对的。
- 坏驱逐的 page 多在 5–32 步后又被选中（约 58%），1–4 步内的约 30%。
- 按层很不均匀：第 11 层占差距的 9.8%；第 0、3–5、7、10 层几乎为 0（这些层的选择集几乎不变）。

**候选特征的预测力**（标签为 16 步内是否被重新选中，AUC 越偏离 0.5 越有用）

| 特征 | AUC |
|---|---|
| 近 32 步被选次数 | **0.72** |
| 近 8 步被选次数 | 0.67 |
| 历史总被选次数 | 0.67 |
| recency | 0.35（反过来看是 0.65） |
| last rank | 0.435（反过来看是 0.565） |
| 选择间隔、掉出间隔（周期性） | 0.43–0.51 |
| 位置、开头 sink、层 | ≈0.50 |

深度 3 的决策树只用到了"近 32 步被选次数"和 recency（规则见 [`tree_rules.txt`](results/dev_pilot/gap/tree_rules.txt)）。

**可学习策略的上界**（按文档留一交叉验证，B=128；G = 填上 LRU–Bélády 差距的比例）

| 策略 | 完整 trace：G | repeat miss 比 LRU 少 | 截断后：G |
|---|---|---|---|
| 逻辑回归 | **11.7%** [4.6, 20.0] | 7.7% | 7.3% [3.4, 10.6] |
| 逻辑回归，去掉 rank | 11.6% [5.0, 19.4] | 7.6% | 7.8% [4.2, 10.7] |
| 深度 3 决策树 | 11.4% [1.3, 22.6] | 7.5% | 3.8% [−2.6, 8.5] |
| 只用近 32 步被选次数 | 7.7% [−5.1, 21.9] | 5.1% | −7.3% |
| 只用 last rank | −159% | 比 LRU 差一倍 | −185% |
| 启发式：下次使用 = τ + 上次间隔 | −161% | — | — |

- **rank 的贡献**：带和不带 rank 的逻辑回归 G 只差 +0.04 pp，配对 CI [−0.6, +0.8]。
- **收益集中在 MFQA**：3 篇 MFQA 上逻辑回归的 G 是 16%–35%，4 篇 GovReport 只有 9%、8%、−4%、2%（见 [`policies_by_doc.csv`](results/dev_pilot/gap/policies_by_doc.csv)）。截断 MFQA 重复后 G 降到 7.3%，说明这部分收益很大程度来自重复输出。
- **顺带的发现**：proposal 的完整策略用的是"近 8 步被选次数、封顶 4 次"，而学出来的模型主要依赖"近 32 步被选次数"，这也是完整策略几乎无效的原因之一。按约定没有据此设计新策略。

**推导**：
- 可学习的部分只占差距的约 7–12%，没达到"repeat miss 少 10%"的目标。
- 这部分几乎全部来自 frequency 和 recency，rank 的贡献为 0。
- frequency + recency 正是 LRU-K、LRFU、ARC 这类经典策略，把研究问题换成"frequency"没有新意。

### 4.7 正式检验：20 篇 dev 调参，预注册，40 篇 held-out 只评估一次

**流程**
1. **补齐 dev。** 在 L4 上用同一协议（MAX_NEW=128，不截断，选择器不变）采集全部 20 篇 dev。
2. **调参并预注册。** 按 proposal 在 B=128 上以 dev miss rate 选参，平局取较小值。之后写 [`preregistration.md`](preregistration.md) 和 [`splits/preregistration.json`](../splits/preregistration.json)，连同 `evaluate.py`、`longbench_score.py` 先在 dev 上试运行，再一起 commit（`2f6a9bd`）。
3. **采集 held-out。** 预注册之后，在同一型号的 GPU（L4）上采集 40 篇 held-out 的稀疏 trace 和 dense 生成。Colab 里只导出，不做回放、不打印生成文本。
4. **只评估一次。** 按预注册里的四条命令各运行一次。没有偏离预注册的地方（见文末"偏离"）。

**一个意外发现：Quest 式选择依赖 GPU 型号。** 同一篇 prompt，A100 与 L4 的选择从第 0 步就不同：生成 token 完全相同的 5 篇文档里，只有 32%–75% 的（层，步）选出同一组 page，排序完全相同的不到 5%。同一块 L4 上重跑则逐行一致（[`gpu_consistency.txt`](results/dev_full/gpu_consistency.txt)）。原因是 bf16 下 page 分数存在大量近似平局，不同 GPU 的数值舍入会翻转排名。所以预注册要求 held-out 与 dev 使用同一型号 GPU，实际也确实都是 L4；之前 A100 上的 8 篇 dev pilot 不参与调参和评估。

**dev 调参结果**（B=128 的 miss 数，行是 δ，列是 β）

| δ \ β | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| 0 | 187,866 | 187,733 | 187,689 | 187,672 | 187,951 |
| 4 | 187,729 | 187,634 | 187,600 | 187,593 | 187,879 |
| 8 | 187,739 | 187,622 | **187,568** | 187,573 | 187,856 |

（完整 5×5 表见 [`tuning.csv`](results/dev_full/tuning.csv)。）

- 选出的参数：完整策略 δ=8、β=2；R+F 的 β=4；R+rank 的 δ=4。
- 最好的组合也只比 δ=β=0（即 LRU）少 0.16% 的 miss。
- 逻辑回归在 20 篇 dev 的 LRU 驱逐候选上训练，系数固定；权重最大的是 freq32（+0.53）和 recency（−0.41）。

**held-out 主结果**（40 篇，127 步 × 28 层，共 9,103,360 次历史 page 访问；G 和"相对 LRU"都按 repeat miss 计算）

| B | 策略 | miss rate | repeat miss | 相对 LRU | G [95% CI] | H2D MiB/token |
|---|---|---|---|---|---|---|
| 128 | FIFO | 5.51% | 284,460 | +87.8% | −134.9% [−147.4, −125.6] | 6.17 |
| 128 | rank-only | 5.80% | 310,599 | +105.0% | −161.4% [−177.2, −147.1] | 6.50 |
| 128 | **LRU** | 4.05% | 151,478 | — | 0 | 4.54 |
| 128 | R+F（β=4） | 4.05% | 151,223 | −0.17% | 0.26% [−0.24, 0.73] | 4.54 |
| 128 | R+rank（δ=4） | 4.05% | 151,213 | −0.17% | 0.27% [0.18, 0.35] | 4.54 |
| 128 | **完整策略（δ=8，β=2）** | 4.05% | 150,893 | **−0.39%** | 0.59% [0.30, 0.88] | 4.53 |
| 128 | 逻辑回归（系数固定） | 3.92% | 139,712 | −7.8% | **11.9%** [6.7, 17.2] | 4.39 |
| 128 | Bélády | 2.97% | 52,915 | −65.1% | 100% | 3.33 |
| 256 | FIFO | 2.57% | 16,843 | +216% | −225.8% | 2.88 |
| 256 | rank-only | 2.53% | 12,623 | +137% | −143.1% | 2.83 |
| 256 | LRU | 2.45% | 5,323 | — | 0 | 2.74 |
| 256 | R+F | 2.45% | 5,323 | 0 | 0 | 2.74 |
| 256 | R+rank | 2.45% | 5,316 | −0.13% | 0.14% [−0.08, 0.33] | 2.74 |
| 256 | 完整策略 | 2.45% | 5,316 | −0.13% | 0.14% [−0.15, 0.35] | 2.74 |
| 256 | 逻辑回归 | 2.44% | 5,096 | −4.3% | 4.5% [2.1, 7.1] | 2.74 |
| 256 | Bélády | 2.39% | 221 | −95.8% | 100% | 2.68 |

- compulsory miss 占全部访问的 2.39%，占 B=128 时 LRU miss 的 58.9%。B=256 时 LRU 的 miss 有 97.6% 是 compulsory，几乎没有可挽回的空间。
- H2D 按每层每个 page 64 KiB（16 token × 8 KV head × 128 维 × K、V × bf16）计算。B=128 时 LRU 每生成一个 token 搬 4.54 MiB，Bélády 是 3.33 MiB，完整策略只省 0.01 MiB。

**成功标准**（[`success.json`](results/heldout/eval/success.json)）

| 标准 | 结果 | 是否满足 |
|---|---|---|
| S1：完整策略比 LRU 少 ≥10% repeat miss | 少 **0.39%**，CI [0.20%, 0.57%] | **不满足** |
| S2：完整策略相对 R+F 节省的 repeat miss，配对 95% CI 整体大于 0 | 节省 330 次，CI [83, 600]，占 R+F 的 0.22% | **满足**（但效应极小） |

**H1**（控制 frequency × recency 后的 top 减 bottom 复用率差，单位 pp；按文档聚类，40 个 cluster）

| 口径 | 1 步内 | 2 步内 | 4 步内 | 8 步内 | top 四分位占比 |
|---|---|---|---|---|---|
| selected | +28.1 [27.2, 28.9] | +18.6 | +11.9 | +7.6 [7.1, 8.1] | 25% |
| candidate | +3.1 [2.5, 3.9] | +5.0 [4.1, 6.3] | +7.7 [6.3, 9.5] | +10.8 [9.1, 13.0] | **0.73%** |
| LRU-candidate | +4.5 [3.5, 6.2] | +7.0 [5.2, 10.2] | +9.9 [7.4, 13.7] | +13.0 [10.2, 18.7] | 0.74% |

按组看，驱逐候选口径 8 步内的效应：GovReport +8.7 [7.3, 10.5]，MFQA +16.3 [13.4, 19.5]，4–8K +12.7 [10.0, 16.0]，8–16K +10.4 [8.3, 13.3]。各组 top 四分位占比在 0.51%–0.91% 之间。

**分组结果**（B=128，[`policies_by_group.csv`](results/heldout/eval/policies_by_group.csv)）

| 分组 | 文档数 | LRU | Bélády | 完整策略 G | 逻辑回归 G |
|---|---|---|---|---|---|
| GovReport | 20 | 4.72% | 3.42% | 0.5% [0.1, 0.9] | 2.4% [−1.4, 6.0] |
| MultiFieldQA-en | 20 | 3.38% | 2.52% | 0.8% [0.4, 1.2] | 26.3% [21.4, 30.2] |
| 4–8K | 20 | 3.34% | 2.54% | 0.5% [0.1, 0.8] | 14.5% [6.9, 22.2] |
| 8–16K | 20 | 4.76% | 3.40% | 0.6% [0.3, 1.1] | 10.4% [3.6, 17.6] |

长文档的绝对差距更大（8–16K 为 1.36 pp，4–8K 为 0.80 pp）。逻辑回归的收益集中在 MFQA。

**次要分析：截断 MFQA 的重复回答**（20 篇 MFQA 都在第一个换行处截断，保留 79,716 行）

| B=128 | LRU | Bélády | 完整策略 | 逻辑回归 |
|---|---|---|---|---|
| miss rate | 5.47% | 4.29% | 5.46% | 5.44% |
| G | 0 | 100% | 0.50% [0.09, 0.91] | 2.4% [−1.2, 6.0] |

截断后 S1 仍不满足（只少 0.31%，CI [0.06%, 0.58%]）；S2 仍满足（节省 281 次，CI [66, 551]）。逻辑回归的优势基本消失，说明它在 MFQA 上的收益主要来自重复输出的周期性。

**LongBench sanity check**（官方指标，×100；[`longbench_summary.json`](results/heldout/longbench/longbench_summary.json)）

| 任务 | 稀疏参考 | dense | 稀疏 − dense [95% CI] |
|---|---|---|---|
| GovReport（ROUGE-L） | 17.42 | 18.70 | −1.28 [−2.31, −0.18] |
| MultiFieldQA-en（F1） | 11.65 | 12.42 | −0.77 [−1.89, +0.30] |

- 两者差距很小，稀疏参考运行可以看作 dense 解码的合理近似。GovReport 上稀疏略差且显著；MFQA 上不显著。
- 绝对分数偏低的原因：MFQA 生成了 128 token，重复回答会拉低 F1（LongBench 官方的 MFQA 上限是 64 token）；GovReport 的摘要只有 128 token（官方上限 512）。

**dev 与 held-out 对比**（B=128；dev 上的逻辑回归是样本内结果）

| 指标 | dev（20 篇） | held-out（40 篇） |
|---|---|---|
| LRU / Bélády miss rate | 4.13% / 3.01% | 4.05% / 2.97% |
| 相对差距 | 27.0% | 26.7% |
| 完整策略相对 LRU（S1） | −0.38% [0.26, 0.49] | −0.39% [0.20, 0.57] |
| S2：完整策略相对 R+F 节省 | 104 [−33, 268]，不显著 | 330 [83, 600]，显著 |
| 完整策略 G | 0.59% | 0.59% |
| 逻辑回归 G | 9.8%（样本内） | 11.9%（样本外） |
| H1-candidate，8 步内 | +9.5 [7.1, 12.6] | +10.8 [9.1, 13.0] |
| 候选中 top 四分位占比 | 0.76% | 0.73% |

dev 与 held-out 几乎一致。S2 在 dev 上不显著、在 held-out 上显著，主要是因为 held-out 的文档数是 dev 的两倍，CI 更窄；效应本身同样极小。

**偏离预注册：无。** 策略、参数、系数、budget、指标、bootstrap 设置和 MFQA 规则都没有改；held-out 只采集一次、评估一次；GPU 型号与 dev 相同。唯一的操作差异是 Colab 里 held-out 的结果单元格不做回放（预注册中已写明）。

---

## 5. 总结论与依据

| 问题 | 结论 | 依据 |
|---|---|---|
| 问题 | 结论（held-out，40 篇） | 依据 |
|---|---|---|
| Gate A：问题有没有被回答过 | **通过**（问题需收窄为增量问题） | §4.1：ArkVale 只用当前分数，HiSparse 只用 LRU |
| 缓存有没有改进空间 | **有**：B=128 相对差距 26.7%（绝对 1.08 pp），Bélády 消掉 65% 的 repeat miss；B=256 时几乎没有空间 | §4.7 |
| H1 / Gate B：rank 在驱逐候选上有没有增量信号 | **有，但弱且罕见**：+3.1 到 +10.8 pp（40 个 cluster，CI 都大于 0），top 四分位只占候选的 0.73% | §4.7 |
| S1：完整策略比 LRU 少 ≥10% repeat miss | **不满足**：少 0.39% [0.20, 0.57] | §4.7 |
| S2：完整策略相对 R+F 的节省 CI > 0 | **满足但可忽略**：330 次 [83, 600]，占 0.22% | §4.7 |
| Gate C：rank 策略能否超过强 baseline | **不能（实际意义上）**：完整策略 G = 0.59%，H2D 只省 0.2% | §4.7 |
| 差距能被简单特征挽回多少 | 固定系数的逻辑回归 G = 11.9%（主要是 MFQA）；截断 MFQA 重复后只有 2.4% 且 CI 跨 0 | §4.6、§4.7 |
| 稀疏参考运行是否可信 | **基本可信**：相对 dense，ROUGE-L 低 1.3 分，F1 低 0.8 分 | §4.7 |

**拟定的对外说法**（蓝图的负面结论版本，已由预注册的 held-out 检验支持）：

> 在固定 Quest 式选择、Qwen3-0.6B、page 级可召回缓存的设定下，B=2K 时 LRU 与 Bélády 之间有约 27% 的相对差距。sparse-attention selection rank 对驱逐候选的复用有统计上显著的增量预测力，但高排名 page 很少成为驱逐候选（0.7%），因此把 rank 加进 recency+frequency 策略只能减少 0.4% 的 repeat miss（预注册目标 10%）。在 dev 上训练的简单特征模型能挽回约 12% 的差距，但主要来自贪心解码下的重复输出；去掉重复后只剩约 2%。

---

## 6. 局限

- **样本量**：正式检验用 20 篇 dev 和 40 篇 held-out，按文档聚类的 bootstrap 分别有 20 和 40 个 cluster。每个任务 × 长度段的 held-out 只有 10 篇，这一级的 CI 仍然较宽。§4.2–4.6 的 pilot 和 8 篇 dev 结果只是方向性的。
- **只有一个模型和一种选择器**：Qwen3-0.6B + Quest 式上界，K=64、16-token page。不能外推到 HiSparse 的 GLM-5.1 token 级设置。
- **trace 依赖 GPU 型号**：同一 prompt 在 A100 与 L4 上的选择差别很大（§4.7）。正式检验全部在 L4 上进行；结论是否适用于其他 GPU 没有验证，但 A100 上 8 篇 dev pilot 的结论方向与 L4 一致。
- **trace 是固定的**：回放没有在线反馈，不影响生成；没有测延迟。
- **MFQA 重复**：贪心解码下的重复输出会夸大复用；停止规则会影响结果。按预注册，主分析用原样 trace，截断版作为次要分析，两者对 rank 策略的结论一致，但逻辑回归的收益在截断后基本消失。
- **文献**：第二轮已补齐大部分未核实条目。仍未核实的是 ArkVale 的 OpenReview 评审（页面有反爬虫验证），以及几处小尾巴（H₂O 附录 D 末尾、HiSparse 代码版本是否与论文评测一致）；Scissorhands 和 H₂O 的引文需对照 PDF 抽查。详见 [`related_work_matrix.md`](related_work_matrix.md) 末尾。
- **选择器与 Quest 不同**：Quest 按 query head 各自选择，我们把各 head 的上界平均成一个排名，每层共用一个缓存。
- **缺少 InfLLM 式 baseline**：按注意力加权的衰减频率分数需要每个 page 的注意力质量，trace v1 没有记录。
- **ArkVale 式 baseline 缺失**：按"当前分数最低"驱逐需要记录未选中 page 的当前分数，trace v1 没有这个字段。

---

## 7. 待团队决定

> **注**：以下是第一部分结束时的待办。第 2 项（layer-level benchmark）和第 4 项（记录未选中 page 的分数）已由第二部分解决：trace v2 记录了全部分数，GPU 测试改为针对新策略做。新主线的待决事项见 §9.7。

已在本轮解决：MFQA 停止规则（主分析按 proposal 生成 128 token 或 EOS，截断版作为次要分析）；held-out 上比较的策略（见预注册）。

1. **按负面结论加现象刻画来写最终报告和 poster**。正式检验支持这个方向：S1 不满足，S2 显著但只有 0.22%。
2. **layer-level GPU benchmark 是否还做**。完整策略每 token 只省 0.01 MiB 的 H2D（约 0.2%），benchmark 预期测不出可见差异，建议降为可选或不做。
3. **是否把"选择对 GPU 型号敏感"写进报告**，作为附带发现。
4. **是否在 trace 中记录未选中 page 的分数**，以便加入 ArkVale 式 baseline。这需要新采集，而 held-out 已经用过一次，只能在 dev 上做探索性分析。
5. **成员名单**：proposal 写 Xihang Yu，README 写 Jimmy Yu。

---

## 8. 复现

结果摘要（不含原始 trace）放在 [`docs/results/`](results/)：
- `pilot/`：单篇 4K pilot；
- `dev_pilot/`：A100 上的 8 篇 dev；
- `dev_full/`：L4 上的 20 篇 dev，含调参表、试运行评估和跨 GPU 对比；
- `heldout/`：40 篇 held-out 的正式评估。

原始 trace（pilot 1.2 MB、dev_pilot 10 MB、dev_full 27 MB、heldout 54 MB）没有放进仓库，可用 notebook 重新采集（`SUBSET` 选 `dev` 或 `heldout`，需 L4）。manifest 中记录了 trace 的 sha256：dev_full 是 `a96cb40d…eb5f`，heldout 是 `f5f7d812…63ae`。

正式检验（预注册之后的四条命令，各运行一次）：

```bash
python3 preregister.py results/dev_full --tuning-csv results/dev_full/tuning.csv
python3 evaluate.py results/heldout --out results/heldout/eval
python3 truncate_trace.py results/heldout
python3 evaluate.py results/heldout --trace trace_truncated.jsonl --out results/heldout/eval_truncated
python3 longbench_score.py results/heldout --out results/heldout/longbench
```

第一条是调参（在预注册之前运行）。`evaluate.py`、`gap_analysis.py` 和 `preregister.py` 需要 numpy；`truncate_trace.py`、`split.py` 和 `longbench_score.py` 需要 transformers（只用 tokenizer），后者还需要 `rouge==1.0.1`。

探索阶段的命令：

```bash
python3 -m unittest discover -s tests -v
python3 replay.py results/dev_pilot/trace.jsonl --budgets 64 128 256 --policies lru fifo rank rf rrank full belady --delta 4 --beta 2 --by task band cell --out results/dev_pilot/replay
python3 analysis.py results/dev_pilot/trace.jsonl --samples selected candidate lru --by task band cell --out results/dev_pilot/h1
python3 gap_analysis.py results/dev_pilot --out results/dev_pilot/gap
python3 truncate_trace.py results/dev_pilot
python3 gap_analysis.py results/dev_pilot --trace trace_truncated.jsonl --out results/dev_pilot/gap_truncated
```

δ 扫描用 `--policies lru belady score:1:0 score:2:0 ...`（`score:δ:β`）。

各脚本的完整参数见 [README](../README.md)。`gap_analysis.py` 需要 numpy。

---

## 9. 第二部分：高效替换策略

**目标**（对齐课程 Project 10 原题）：在同一个可召回的 Quest page cache 上，找一个极小、因果、几乎不花代价的替换策略，拿到 Bélády 相对 LRU 优势的一大部分；然后测它在 GPU 上能换来什么。不使用 RL。第一部分的负面结论保留，作为消融。

### 9.1 为什么换主线、换设定

- **rank 失败的原因要先查清。** 第一部分的 trace v1 每步只记录 top-K 的排名。驱逐时要比较的却是**没被选中**的驻留 page，它们在 v1 里没有任何当前信息。所以 trace v2（`db51faa`）记录每一步所有历史 page 的 Quest 分数（float16，npz）。Quest 每步本来就给所有 page 打分，在线策略拿到这个分数没有额外代价。
- **差距要足够大。** 第一部分在 4–16K、128 步下，LRU 与 Bélády 只差 1.08 pp，低于 proposal 定的 2 pp 门槛。新设定把上下文加长到 16–32K，并用带思考的长摘要让生成更长。

### 9.2 长 trace pilot 与冻结（[`long_trace_pilot.md`](long_trace_pilot.md)）

**prompt 和停止规则怎么定的：**
- 不带 chat template 的 prompt 永远不输出 EOS，约 250 token 后开始逐字重复答案，会虚高命中率，弃用。
- 带思考的具体问题在 304 步就结束，太短。
- **最终选择**：chat template、开启思考、要求"至少八段"的全文摘要。4 篇探测文档分别在 741–873 步正常输出 EOS，没有循环。
- 停止规则：EOS、1024 token，或最新的 32-gram 在输出里已经出现过。

**pilot 结果**（一篇 26K token 的 QMSum，741 步）：B=96 时 LRU 5.70%、Bélády 3.54%，差 2.15 pp；B=128（B/K=2）时只差 1.35 pp。
- 把 page 从 16 token 改成 8 或 4 token，差距只略微变大，粒度不是主要原因。
- 差距集中在少数几层（L11、L16、L24、L25）。
- 解释：HiSparse 按 token 缓存，我们按 16-token page 缓存，page 级的选择在相邻步之间更稳定，所以 B/K=2 时空间天然更小。这一点作为发现写进报告，不为了凑数改 B 的定义。

**团队决定**：主 budget 用 B/K=1.5（B=96），完整曲线用 B/K ∈ {1.125, 1.25, 1.5, 2, 4}。连同划分一起冻结在 [`splits/longctx_v1.json`](../splits/longctx_v1.json)（`950cb37`）：**14 篇 dev**（10 GovReport、4 QMSum）、**21 篇 held-out**（15 GovReport、6 QMSum）。

### 9.3 实现

| 文件 | 作用 |
|---|---|
| [`notebooks/first_trace.ipynb`](../notebooks/first_trace.ipynb)、[`notebooks/long_trace_capture.py`](../notebooks/long_trace_capture.py) | schema v2 采集：每步每层的 top-K 排名（int16）和所有 page 的分数（float16），存成 npz（26K token、1K 步的文档约 97 MB） |
| [`replay.py`](../replay.py)、[`truncate_trace.py`](../truncate_trace.py) | 兼容 v1 和 v2（gzip） |
| [`kvpolicy.py`](../kvpolicy.py) | 策略族模拟器。语义与 HiSparse 一致：每步的 top-K page 必须驻留，策略只决定驱逐哪些未选中的驻留 page |
| [`efficient_dev.py`](../efficient_dev.py) | dev 研究：参数网格、按文档留一（LODO）选参、按层分配、学习式模型、差距分析 |
| [`checks.py`](../checks.py) | 稳健性检查、消融、机制分析 |
| [`heldout.py`](../heldout.py) | `freeze` 只用 dev 冻结分配和线性系数；`eval` 对 held-out 按预注册评估 |
| [`gpu_bench.py`](../gpu_bench.py) | 真实两级卸载 cache 和计时（在 Colab 上运行） |
| `tests/test_kvpolicy.py`、`tests/test_schema_v2.py` | 新增测试；整个测试集 50 个全部通过（需要 numpy） |

**score EMA 的定义**：每个驻留 page 存一个浮点数 e。每步更新 `e ← α·e + (1−α)·z`，其中 `z = (s − s_K) / (s_1 − s_K)`，s 是该 page 当前的 Quest 分数，s_1 和 s_K 是本步第 1 名和第 K 名的分数。驱逐 e 最小的 page。新装入的 page 以当前 z 作为初值。

### 9.4 dev 结果与结论推导（[`efficient_policy_dev.md`](efficient_policy_dev.md)）

所有超参数都用 14 篇 dev 的 LODO 选择。G = (LRU miss − 策略 miss) / (LRU miss − Bélády miss)，CI 是按文档聚类的 2000 次 bootstrap。

**主表（B=96，LRU 6.03%，Bélády 3.86%）：**

| 策略 | miss rate | G [95% CI] | 优于 LRU 的文档 |
|---|---|---|---|
| **score EMA**（α=0.5，14 折都选了它） | 5.42% | **+0.279** [0.274, 0.284] | 14/14 |
| 当前分数，不平均（KVDrive 式） | 5.50% | +0.244 [0.236, 0.251] | 14/14 |
| **按层分配 + score EMA** | 5.14% | **+0.409** | — |
| 按层分配 + LRU | 5.75% | +0.127 | — |
| 线性下次使用距离回归（DAgger） | 5.38% | +0.300 [0.295, 0.305] | 14/14 |
| InfLLM 式衰减求和，所有驻留 page | 5.44% | +0.269 | 14/14 |
| 衰减选中计数，不看分数（MoE 式） | 5.93% | +0.044 | 14/14 |
| 第一部分的 rank 策略 | 5.97% | +0.027 | 14/14 |
| ARC / LRU-2 / LFU / FIFO | 6.04 / 6.13 / 7.85 / 8.18% | −0.01 / −0.05 / −0.84 / −1.00 | — |

**结论 1：有用的信息是"未选中驻留 page 的当前分数"。** 推导来自消融（B=96 的 G）：
- EMA 每步更新所有驻留 page：+0.279。只在 page 被选中时更新：−1.596，远差于 LRU，因为过时的高分把已经不在 top-K 附近的 page 钉住了。
- 把分数换成排名（v1 里唯一的信息）：+0.016，基本等于 LRU。**这解释了第一部分的负面结果：v1 恰好丢掉了有用的信号。**
- 不做平均、只用当前分数：+0.244。平均带来约 +0.035。

**结论 2：这条规则不是新规则。** InfLLM 的衰减求和（用所有驻留 page 上 Quest 上界的 softmax 代替注意力概率）得到 +0.269，每个 budget 下都和 EMA 相差 0.01–0.04。所以报告里把它写成"InfLLM 式衰减分数规则，用在 Quest 分数上"，贡献是测量和消融。

**结论 3：按层分配和分数规则可以叠加。**
- 差距集中在少数几层，所以按每层的 miss 曲线贪心分配 slot（总量保持 28·B）。
- LODO 分配能拿到 oracle 分配收益的 89–94%，不同折之间几乎一样。
- B=96 时 EMA 0.279 加上分配 + LRU 0.127，合计 0.406，实际的分配 + EMA 是 0.409，接近相加。
- KVDrive 式精确 knapsack 分配和贪心结果相同；按 LRU–Bélády 差距比例分配反而更差（−0.32）。

**结论 4：效果稳定。**
- EMA 在每篇文档、每个任务、每个长度段、每个 budget 上都优于 LRU。B=96 时最差的一篇也比 LRU 少 9.3% 的 miss。
- α 在 0.3–0.7 之间时 G 都在 0.268–0.279 之间。最优 α 随 budget 变化（B/K≤1.25 时 0.4，4 时 0.95），所以 held-out 按 budget 分别冻结 α。
- 外部有效性检查（只在 dev 上）：Qwen3-1.7B 的 4 篇 G +0.281，另一种摘要 prompt 的 3 篇 G +0.276。

**结论 5：学习式策略只多一点。**
- 6 个特征的线性回归加两轮 DAgger，G 为 0.300（B=96）和 0.317（B=128、256）。DAgger 比行为克隆只多约 0.005。
- 它在 B/K=4 时明显好于 EMA（0.317 对 0.220），但每个 page 要存约 5 个浮点数。

**差距分析**：B=96 时 LRU 的被驱逐 page 中，68.5% 在 32 步内又被选中，而 LRU 留下的候选是 78.9%，LRU 只比随机略好。单特征 AUC：z_ema 0.70、当前 z 0.67、缓存内选中次数 0.65、最后排名 0.41。

**"不准入"变体**：HiSparse 语义下选中的 page 必须装入，所以"不准入"只能是"用完就驱逐"，这已经在所有策略的动作空间里。Bélády 的驱逐中有 18.2% 是这种用一次就走的（LRU 只有 0.3%），对应 MoE 缓存文献的发现。

**开销**（每层每步，B 个驻留 page）：

| 策略 | 每个 page 的状态 | 每步计算 |
|---|---|---|
| LRU | 1 个整数 | 更新 K 个时间戳，取最小 |
| 当前分数 | 0 | 取 B 个分数，取最小 |
| **score EMA** | 1 个浮点数 | 取 B 个分数，B 次乘加，归一化，取最小 |
| 线性回归 | 约 5 个浮点数 | 每个候选约 6 次乘加 |
| 按层分配 | 0 | 0（一张 28 项的 budget 表，离线算好） |

Quest 本身每层每步要算约 6–8 MFLOP，EMA 只加约 400 FLOP，不到 0.01%；每个请求的状态约 11 KB。

### 9.5 held-out 正式检验（[`heldout_results.md`](heldout_results.md)）

**流程：**
1. 预注册写在 `efficient_policy_dev.md` 末尾，11:35 标记为已批准（`bb0ce87`）。
2. 只用 14 篇 dev 冻结按层分配和线性系数，12:34 提交（`f87e982`），之后才读任何 held-out 输出。
3. 21 篇 held-out 用同样的模型、prompt、选择器、解码和停止规则采集一次，按预注册评估一次（`91f7041`）。
4. sanity check：在新 runtime 上重新采集 dev，完全复现 dev 报告（LRU 6.03%，EMA G 0.279）。

**成功标准（B=96）：全部通过。**

| 标准 | 阈值 | 结果 |
|---|---|---|
| (i) score EMA 的 G，CI 下界 | > 0.15 | G 0.281，CI [0.277, 0.285] |
| (ii) EMA 优于 LRU 的文档数 | ≥ 18/21 | 21/21 |
| (iii) 分配 + EMA 优于 EMA（配对 CI 不含 0） | CI > 0 | ΔG +0.120，CI [0.113, 0.128] |

单篇文档的 EMA G 在 0.26–0.30 之间。

**budget 曲线（21 篇合并）：**

| B（B/K） | LRU miss | Bélády | EMA G | 分配 + EMA G | InfLLM 式 G | 当前分数 G | 线性 G |
|---|---|---|---|---|---|---|---|
| 72（1.125） | 11.50% | 8.81% | 0.257 | 0.257* | 0.239 | 0.229 | – |
| 80（1.25） | 8.88% | 6.19% | 0.276 | 0.307 | 0.257 | 0.249 | – |
| **96（1.5）** | **5.98%** | **3.85%** | **0.281** | **0.402** | 0.270 | 0.246 | 0.303 |
| 128（2） | 3.56% | 2.19% | 0.271 | 0.528 | 0.266 | 0.224 | 0.314 |
| 256（4） | 1.38% | 0.98% | 0.242 | 0.242* | 0.216 | 0.145 | 0.336 |

\* B=72 和 256 时分配网格到了边界，冻结的分配是均匀的，所以等于 EMA。

B=96 时 repeat miss 比 LRU 少：EMA 11.6%，分配 + EMA 16.6%，Bélády 41.3%。dev 和 held-out 几乎一致，策略排名不变。

### 9.6 GPU 延迟测试（[`gpu_benchmark.md`](gpu_benchmark.md)）

**设置：**
- Colab L4，Qwen3-0.6B，3 篇 held-out 文档（20–30K token），每篇 384 个解码步。held-out 只用来计时，没有在上面调任何参数。
- prefill 之后，所有完整 page 移到 pinned host memory；每层在 GPU 上有 B 个 slot，按需装入。
- 每步每层：给所有 page 打分，取 top-K，策略选驱逐对象，把缺失的 page 从 host 拷到 GPU，再做 attention。
- 拷贝是同步的，没有预取，也没有和计算重叠。

**正确性**：
- 24 次卸载运行每一步每一层选出的 page 都和 trace 一致，生成的 token 也一致。
- LRU 和 Bélády 的 miss 数和模拟器完全相同。EMA 差不到 0.1%，因为 trace 存的是 fp16 分数，在线运行用 fp32，少数近似打平的情况结果不同。

**延迟（3 篇的中位数）：**

| 配置 | ms/token（B=96） | ms/token（B=128） | PCIe MB/token（B=96） |
|---|---|---|---|
| Quest，KV 全在 GPU（不卸载） | 57.1 | – | 0 |
| 卸载 + LRU | 86.6 | 84.5 | 8.50 |
| 卸载 + score EMA | 93.1 | 89.5 | 7.71 |
| 卸载 + 分配 + EMA | 93.0 | 90.3 | 7.38 |
| 卸载 + Bélády（oracle） | 87.3 | – | 5.61 |

**结论：在这个规模下，miss 少了不会变快。** 推导：
- 实测 host→GPU 带宽 12.2 GB/s，LRU 每 token 的 8.5 MB 只需约 0.7 ms。但传输阶段实际花了 11.4 ms，九成以上是每层的固定开销（CPU gather、kernel 启动、同步）。
- 策略阶段同理：EMA 为了省约 0.07 ms 的传输，多了约 5 ms 的小 kernel 启动。
- Bélády 比 LRU 少 34% 的 miss，延迟却只差 1 ms 以内，直接说明 miss 在这里几乎不影响延迟。

所以"几乎零开销"对算法本身成立（每个 page 一个浮点数），但对这个未融合的 PyTorch 实现不成立，EMA 每 token 慢约 7.5%。要让 miss 减少变成延迟收益，传输必须占每步时间的很大一部分：更大的模型或 batch、更慢的链路，或者融合 kernel 去掉固定开销。这些都还没测。

**后续测试：免同步、zero-copy 实现（`e327cfb`）。** 目的是排除"每层同步和 CPU 拷贝掩盖了策略收益"这种可能。`gpu_bench.run(..., zc=True)` 不再做 host 同步，驱逐对象用固定大小（K）的张量操作选出；一个 Triton kernel 直接从映射的 pinned host memory 读取缺失的 page（微基准约 12 GB/s），不经过中转拷贝。

- **正确性**（gov_report/00dfb，128 步）：token 和排名完全一致。LRU 19,538 次、Bélády 13,604 次 miss，与模拟器完全相同；EMA 17,929 对模拟器 17,937，分配 + EMA 17,380 对 17,377。
- **延迟仍然不变**：LRU 89.1、EMA 95.5、分配 + EMA 95.5、Bélády 89.4 ms/token。同步不是瓶颈，瓶颈是 **CPU 派发**：Python 和 HF 的开销，加上每层约 60 次小 kernel 启动。
- **只算 GPU 实际忙的时间**（`torch.profiler` 统计所有 kernel 和拷贝的耗时，减去 prefill，40 步，B=96）。它近似一个没有启动开销的实现（CUDA graph 或融合 kernel）：

  | 配置 | GPU ms/步 | miss/步 |
  |---|---|---|
  | Quest，KV 全在 GPU（HF DynamicCache） | 35.0 | 0 |
  | 卸载 + LRU | 16.7 | 194 |
  | 卸载 + score EMA | 17.4 | 184 |
  | 卸载 + 分配 + EMA | 17.8 | 182 |
  | 卸载 + Bélády | 16.3 | 155 |

  推导：
  - 一次 miss 在 GPU 上很便宜。Bélády 每步少 39 次 miss，只省 0.45 ms，约每个 64 KiB page 11.5 µs。
  - EMA 每步少约 10 次 miss（约 0.12 ms），但它没合并的几个逐元素 kernel 更贵，GPU 时间净多 0.7 ms。合并成每层一个 kernel 后大约能打平。
  - 即使没有启动开销，最好的策略（Bélády）也只值约 3% 的 GPU 时间，EMA 能拿到的不到 1%。
- **"KV 全在 GPU"这组对照偏大。** HF 的 `DynamicCache` 每步拼接整个 KV（约 3.4 GB），所以它的 GPU 工作量反而比卸载版多；墙钟时间只有 57 ms，是因为它启动的 kernel 更少。公平的对照应该用 static cache，还没做。
- **什么时候 miss 才重要。** 8-head GQA 下每个 page 每层的 KV 字节数不随模型宽度增长，所以关键是 **batch**：每条序列都带来自己的 miss。外推估算（未实测）：batch 32 时 LRU 每步传约 400 MB（12 GB/s 下约 33 ms），EMA 9–10% 的字节节省约合每步 3 ms。

GPU 忙碌时间来自每个配置在一篇文档上的一次 profile，只作参考（±0.2 ms）。

### 9.7 局限与待团队决定

**局限：**
- 只有 14 篇 dev 和 21 篇 held-out，来自 2 个任务、一种 prompt。CI 窄说明各文档效果一致，不说明覆盖面广。
- 只有一个主模型（Qwen3-0.6B）和一种选择器；1.7B 和第二种 prompt 只在 dev 上抽查。
- G 是相对 page 级 Bélády 算的；结论不能直接推到 token 级缓存。
- InfLLM baseline 用 Quest 上界的 softmax 代替真实注意力质量。
- 延迟只在一种 GPU、batch=1、3 篇文档上测过；GPU 忙碌时间只 profile 了一篇文档。

**待团队决定：**
1. 报告里把策略写成"InfLLM 式衰减规则用在 Quest 分数上"（推荐），贡献是测量、消融和分配叠加。
2. 按层分配（需要离线、和模型相关的曲线）是否算作报告里的"策略"。
3. 延迟结论怎么写：只报"字节减少、延迟没变"加上 GPU 忙碌时间的分析，还是再投入去测一个传输占比大的场景。最有希望的是加大 batch（按外推，batch 32 时 EMA 每步约省 3 ms），其次是融合 EMA kernel 和 static-cache 对照。
4. 是否需要用真实注意力质量重做 InfLLM 对比（要新的采集字段）。
5. 成员名单仍未统一（见 §7 第 5 项）。

### 9.8 复现

原始 trace 没有放进仓库（被 gitignore）。held-out trace 和各类输出备份在 Google Drive 的 `MyDrive/rank-kv-cache/` 下（`longctx_v1_heldout/`、`results_heldout/`、`results_efficient_dev/`、`results_gpu_benchmark/`）。采集用 `notebooks/long_trace_capture.py`，需要 L4。

```bash
python3 -m unittest discover -s tests
python3 efficient_dev.py DEV_TRACES --out results/efficient_dev
python3 checks.py DEV_TRACES --res results/efficient_dev --budgets 96 128 256
python3 heldout.py freeze DEV_TRACES --out frozen.json
python3 heldout.py eval HELDOUT_TRACES --frozen splits/heldout_frozen_v1.json --out results/heldout_longctx
```

`DEV_TRACES` 和 `HELDOUT_TRACES` 是放 npz trace 的目录。以上脚本都需要 numpy。`gpu_bench.py` 在 Colab 上和模型一起运行。
