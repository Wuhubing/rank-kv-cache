# 项目近况报告：Selection Rank 能否改进 KV Cache 替换？

**截至 2026-11-22 · 分支 `stage1-2` · 已完成 proposal 规定的正式检验：20 篇 dev 调参并预注册（`2f6a9bd`），40 篇 held-out 采集一次、评估一次**

**2026-11-25 补充（事后分析，非预注册）**：S2 的 β 混杂与干净 rank 增量（§4.7）、H1 `candidate` 口径的准确描述（§4.7）、逻辑回归标签的右删失（§6）。所有预注册结果和判定不变。

本文汇总从 proposal 到现在做过的全部工作：每一步做了什么、用什么实现、得到什么数据，以及每条结论是怎么推出来的。所有数字都能在 [`docs/results/`](results/) 下的 CSV/TXT 中找到原始输出，文末给出复现命令。

---

## 0. 一页摘要

**正式结论（40 篇 held-out，按预注册执行一次）：selection rank 不能实质改进 KV cache 替换。proposal 的主成功标准（S1）不满足。**

1. **缓存确实有改进空间。** held-out 上 B=128 时 LRU 的 miss rate 是 4.05%，Bélády 是 2.97%，绝对差 1.08 pp、相对差 26.7%；Bélády 比 LRU 少 65% 的 repeat miss。dev 上是 4.13% 对 3.01%，几乎一样。
2. **两条成功标准（B=128，held-out）：**
   - **S1 不满足**：完整策略（dev 调出 δ=8、β=2）只比 LRU 少 **0.39%** 的 repeat miss，CI [0.20%, 0.57%]，目标是 10%。
   - **S2 在统计上满足，但效应极小**：完整策略比 R+F 少 330 次 repeat miss，配对 CI [83, 600] 整体大于 0，只占 R+F repeat miss 的 0.22%。**注意 S2 有混杂**：完整策略是 δ=8、β=2，R+F 是 δ=0、β=4，两者同时改变了 β 和 rank 项，所以 S2 并不单独衡量 rank 项（预注册文件中"S2 isolates the rank term"的说法不准确）。S2 的预注册判定照原样保留为"满足"。
   - **事后（post-hoc，非预注册）的干净 rank 增量**：固定 β=2，只比较 δ=8 与 δ=0，held-out 上 rank 项节省 213 次 repeat miss，配对 CI [96, 341]，占基线的 0.14%；其余 117 次 [−118, 394] 来自 β 从 4 改到 2。rank 项有作用，但小到没有实际意义。
3. **H1：rank 对驱逐候选有真实但很小的增量信号。** 控制 frequency × recency 后，在 candidate 口径（t 之前被选中过、本步未选中的 page）上，top 与 bottom 四分位的复用率差为 +3.1 pp（1 步内）到 +10.8 pp（8 步内），40 个文档 cluster 的 CI 都在 0 以上。但上一次被选中时排在 top 四分位、本步未被选中的 page 非常少：在 B=128 的 LRU 缓存实际驻留、本步未选中的 page 中只占 0.74%（`candidate` 口径为 0.73%，但该口径包括已被驱逐的 page 和没有发生驱逐的步，不是"实际驱逐候选"），所以 rank 加分几乎改变不了驱逐决策。
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

GPU：pilot 和 8 篇 dev pilot 用的是 NVIDIA A100-SXM4-40GB；20 篇 dev（`dev_full`）和 40 篇 held-out 都用 NVIDIA L4。其余设置相同：bf16，torch 2.14.1+cu130，transformers 4.53.3，贪心解码，batch=1。

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

- 选出的参数：完整策略 δ=8、β=2；R+F 的 β=4；R+rank 的 δ=4。R+F 的 β 是在 δ=0 的约束下单独选的，所以它与完整策略的 β 不同（见下文 S2 的混杂）。
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
| S2：完整策略相对 R+F 节省的 repeat miss，配对 95% CI 整体大于 0 | 节省 330 次，CI [83, 600]，占 R+F 的 0.22% | **满足**（但效应极小；有 β 混杂，见下） |

**S2 的混杂与事后的干净 rank 增量（post-hoc / exploratory，非预注册）。** S2 比较的是完整策略（δ=8、β=2）和 R+F（δ=0、β=4）。两者的差别不只是加了 rank 项，β 也从 4 变成了 2，所以 S2 衡量的是"rank 项 + β 改变"的合计，[`preregistration.md`](preregistration.md) 中"S2 isolates the rank term"的说法不成立。预注册文件已冻结，不修改；S2 的判定按预注册照原样记为"满足"，这里只修正它的解读。

为了单独看 rank 项，我们在看到 held-out 结果之后另写了 [`posthoc_rank_increment.py`](../posthoc_rank_increment.py)：固定 β=2，比较 `score:8:2`（即完整策略）与 `score:0:2`（去掉 rank 项），并把 S2 的差拆成两部分：R+F − 完整策略 =（R+F − score:0:2）+（score:0:2 − 完整策略）。统计方法与 `evaluate.py` 相同（逐文档 repeat miss，配对文档 bootstrap，2000 次，seed 0）。**这是事后分析，不是预注册检验，不能用来重新判定 S2。**

| 事后分析，B=128 | held-out 原样 | held-out 截断 | dev 原样 | dev 截断 |
|---|---|---|---|---|
| 干净 rank 增量：score:0:2 − 完整策略 | **213 [96, 341]**，0.14% | 71 [6, 146]，0.07% | 121 [59, 192]，0.16% | 58 [19, 105]，0.11% |
| β 改变：R+F − score:0:2 | 117 [−118, 394]，0.08% | 210 [−3, 475]，0.22% | −17 [−161, 147]，−0.02% | 17 [−126, 181]，0.03% |
| 合计 = S2（复现预注册数字） | 330 [83, 600] | 281 [66, 551] | 104 [−33, 268] | 75 [−50, 228] |

（百分比相对各行的基线策略。B=256 时 β 改变的贡献为 0，干净 rank 增量在 held-out 上是 7 [−6, 23]，CI 跨 0。原始输出见 [`heldout/posthoc_rank_increment/`](results/heldout/posthoc_rank_increment/)、[`heldout/posthoc_rank_increment_truncated/`](results/heldout/posthoc_rank_increment_truncated/) 和 dev 下的同名目录。）

**推导**：在四条 trace 上，固定 β 后 rank 项的节省 CI 都在 0 以上，所以 rank 项确实有一点作用；但它只占基线 repeat miss 的 0.07%–0.16%，比 S2 的 0.22% 还小。held-out 原样 trace 上 S2 的 330 次里约三分之一来自 β 的改变。结论不变：rank 项的效应真实但可以忽略。

**H1**（控制 frequency × recency 后的 top 减 bottom 复用率差，单位 pp；按文档聚类，40 个 cluster）

| 口径 | 1 步内 | 2 步内 | 4 步内 | 8 步内 | top 四分位占比 |
|---|---|---|---|---|---|
| selected | +28.1 [27.2, 28.9] | +18.6 | +11.9 | +7.6 [7.1, 8.1] | 25% |
| candidate | +3.1 [2.5, 3.9] | +5.0 [4.1, 6.3] | +7.7 [6.3, 9.5] | +10.8 [9.1, 13.0] | 0.73% |
| LRU-candidate | +4.5 [3.5, 6.2] | +7.0 [5.2, 10.2] | +9.9 [7.4, 13.7] | +13.0 [10.2, 18.7] | **0.74%** |

口径说明（[`analysis.py`](../analysis.py)）：`candidate` 是"t 之前被选中过、t 步未被选中"的所有 page，与策略无关；它包括在任何有限缓存里早已被驱逐的 page，也包括 t 步根本没有发生驱逐的情况，因此是驱逐候选的一个上位集合，不是"实际驱逐候选"。`LRU-candidate` 只保留 B=128 的 LRU 缓存在 t−1 步后实际驻留、t 步未选中的 page，更接近真正参与驱逐比较的集合（但同样包括没有发生驱逐的步）。所以"rank 加分能起作用的机会有多少"应引用 LRU-candidate 的 0.74%；两个口径的数字几乎相同，结论不受影响。

按组看，candidate 口径 8 步内的效应：GovReport +8.7 [7.3, 10.5]，MFQA +16.3 [13.4, 19.5]，4–8K +12.7 [10.0, 16.0]，8–16K +10.4 [8.3, 13.3]。各组 top 四分位占比在 0.51%–0.91% 之间。

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

截断后 S1 仍不满足（只少 0.31%，CI [0.06%, 0.58%]）；S2 仍满足（节省 281 次，CI [66, 551]），但同样有 β 混杂：事后（非预注册）的干净 rank 增量只有 71 次 [6, 146]，其余 210 次来自 β 改变。逻辑回归的优势基本消失，说明它在 MFQA 上的收益主要来自重复输出的周期性。

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
| 事后（非预注册）干净 rank 增量：score:0:2 − 完整策略 | 121 [59, 192] | 213 [96, 341] |
| 完整策略 G | 0.59% | 0.59% |
| 逻辑回归 G | 9.8%（样本内） | 11.9%（样本外） |
| H1-candidate，8 步内 | +9.5 [7.1, 12.6] | +10.8 [9.1, 13.0] |
| top 四分位占比：LRU-candidate（candidate） | 0.75%（0.76%） | 0.74%（0.73%） |

dev 与 held-out 几乎一致。S2 在 dev 上不显著、在 held-out 上显著，主要是因为 held-out 的文档数是 dev 的两倍，CI 更窄，另外 β 改变的那部分在 dev 上约为 0、在 held-out 上为正；效应本身同样极小。事后的干净 rank 增量在 dev 和 held-out 上都显著且都很小。

**偏离预注册：无。** 策略、参数、系数、budget、指标、bootstrap 设置和 MFQA 规则都没有改；held-out 只采集一次、评估一次；GPU 型号与 dev 相同。唯一的操作差异是 Colab 里 held-out 的结果单元格不做回放（预注册中已写明）。事后分析（干净 rank 增量、§6 的标签删失统计）是在正式评估之后另加的，不改变任何预注册结果，输出放在新的 `posthoc_*` 目录里，不覆盖预注册输出。

---

## 5. 总结论与依据

| 问题 | 结论（held-out，40 篇） | 依据 |
|---|---|---|
| Gate A：问题有没有被回答过 | **通过**（问题需收窄为增量问题） | §4.1：ArkVale 只用当前分数，HiSparse 只用 LRU |
| 缓存有没有改进空间 | **有**：B=128 相对差距 26.7%（绝对 1.08 pp），Bélády 消掉 65% 的 repeat miss；B=256 时几乎没有空间 | §4.7 |
| H1 / Gate B：rank 在驱逐候选上有没有增量信号 | **有，但弱且罕见**：candidate 口径 +3.1 到 +10.8 pp（40 个 cluster，CI 都大于 0）；LRU 缓存中驻留且未被选中的 page 只有 0.74% 上次排在 top 四分位 | §4.7 |
| S1：完整策略比 LRU 少 ≥10% repeat miss | **不满足**：少 0.39% [0.20, 0.57] | §4.7 |
| S2：完整策略相对 R+F 的节省 CI > 0 | **满足但可忽略**：330 次 [83, 600]，占 0.22%。判定按预注册保留；但 S2 同时改变了 β（4→2），不单独衡量 rank 项 | §4.7 |
| 事后（非预注册）：固定 β=2 的干净 rank 增量 | 213 次 [96, 341]，占 0.14%；rank 项真实但可忽略 | §4.7 |
| Gate C：rank 策略能否超过强 baseline | **不能（实际意义上）**：完整策略 G = 0.59%，H2D 只省 0.2% | §4.7 |
| 差距能被简单特征挽回多少 | 固定系数的逻辑回归 G = 11.9%（主要是 MFQA）；截断 MFQA 重复后只有 2.4% 且 CI 跨 0 | §4.6、§4.7 |
| 稀疏参考运行是否可信 | **基本可信**：相对 dense，ROUGE-L 低 1.3 分，F1 低 0.8 分 | §4.7 |

**拟定的对外说法**（蓝图的负面结论版本，已由预注册的 held-out 检验支持）：

> 在固定 Quest 式选择、Qwen3-0.6B、page 级可召回缓存的设定下，B=2K 时 LRU 与 Bélády 之间有约 27% 的相对差距。sparse-attention selection rank 对驱逐候选的复用有统计上显著的增量预测力，但高排名 page 很少出现在驱逐候选中（LRU 缓存里驻留且本步未选中的 page 只有 0.7% 上次排在 top 四分位），因此把 rank 加进 recency+frequency 策略只能减少 0.4% 的 repeat miss（预注册目标 10%）。在 dev 上训练的简单特征模型能挽回约 12% 的差距，但主要来自贪心解码下的重复输出；去掉重复后只剩约 2%。

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
- **逻辑回归的训练标签有右删失**：标签是"16 步内再次被选中"。`gap_analysis.build_dataset` 把在 trace 结束前没有再被选中的候选一律记为负例，但 t+16 ≥ 127（trace 长度）的行其实没有观察完整个 16 步窗口，真实标签未知。事后统计（post-hoc，非预注册；[`posthoc_censoring.py`](../posthoc_censoring.py)，[`censoring.json`](results/dev_full/posthoc_censoring/censoring.json)）：在 dev_full 的 30 万行训练样本中，有 **25,678 行（8.6%）** 属于这种删失负例，占全部负例的 21.5%；全部 228.7 万个候选行中的比例相同（8.5%）。我们确认重建的样本能逐位复现冻结的系数。这会让模型低估临近 trace 末尾的复用概率，并可能使系数有偏（方向和对 G 的影响没有检验）；冻结的模型和预注册结果不改，后续若重新训练应丢弃或单独处理这些行。
- **ArkVale 式 baseline 缺失**：按"当前分数最低"驱逐需要记录未选中 page 的当前分数，trace v1 没有这个字段。

---

## 7. 待团队决定

已在本轮解决：MFQA 停止规则（主分析按 proposal 生成 128 token 或 EOS，截断版作为次要分析）；held-out 上比较的策略（见预注册）。

1. **按负面结论加现象刻画来写最终报告和 poster**。正式检验支持这个方向：S1 不满足，S2 显著但只有 0.22%。写 S2 时需说明它有 β 混杂，并把事后的干净 rank 增量（0.14%）明确标为 post-hoc。
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

第一条是调参（在预注册之前运行）。

事后分析（post-hoc / exploratory，非预注册；`posthoc_censoring.py` 需要 numpy）：

```bash
python3 posthoc_rank_increment.py results/heldout/trace.jsonl --out docs/results/heldout/posthoc_rank_increment
python3 posthoc_rank_increment.py results/heldout/trace_truncated.jsonl --out docs/results/heldout/posthoc_rank_increment_truncated
python3 posthoc_rank_increment.py results/dev_full/trace.jsonl --out docs/results/dev_full/posthoc_rank_increment
python3 posthoc_rank_increment.py results/dev_full/trace_truncated.jsonl --out docs/results/dev_full/posthoc_rank_increment_truncated
python3 posthoc_censoring.py results/dev_full --out docs/results/dev_full/posthoc_censoring
```
`evaluate.py`、`gap_analysis.py` 和 `preregister.py` 需要 numpy；`truncate_trace.py`、`split.py` 和 `longbench_score.py` 需要 transformers（只用 tokenizer），后者还需要 `rouge==1.0.1`。

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
