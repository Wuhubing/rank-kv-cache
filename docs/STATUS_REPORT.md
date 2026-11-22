# 项目近况报告：Selection Rank 能否改进 KV Cache 替换？

**截至 2026-11-22 · 分支 `stage1-2` · 只用了 dev 数据，held-out 未触碰**

本文汇总从 proposal 到现在做过的全部工作：每一步做了什么、用什么实现、得到什么数据，以及每条结论是怎么推出来的。所有数字都能在 [`docs/results/`](results/) 下的 CSV/TXT 中找到原始输出，文末给出复现命令。

---

## 0. 一页摘要

**当前结论（基于 8 篇 dev 文档，方向性）：selection rank 不能改进 KV cache 替换。**

1. **缓存确实有改进空间。** B=128 时 LRU 的 miss rate 是 4.34%，Bélády 是 3.09%，绝对差 1.25 pp、相对差 28.8%。只看 repeat miss，Bélády 比 LRU 少 66%。LRU 的驱逐中有 68% 是"坏驱逐"（被踢掉的 page 比某个留下的 page 更早被再次用到）。
2. **但 rank 补不上这段差距。**
   - 按 proposal 原口径（被选中的 page），H1 效应很强（+28 pp），但这批 page 本来就受保护，和驱逐决策无关。
   - 改成驱逐候选口径后，rank 的增量效应只有 +2.5 到 +11.6 pp，而且驱逐候选里 top 四分位只占 0.8%，几乎没有机会改变决策。
   - 完整 rank 策略只比 LRU 少 0.45% 的 repeat miss，填上差距的 0.7%；δ 从 1 扫到 32，结果基本不变。
3. **其他简单特征也只能填上一小部分。** 用 recency、frequency、rank、位置等特征训练的逻辑回归（按文档留一交叉验证）最多填上差距的 11.7%（截断 MFQA 重复后是 7.3%），去掉 rank 后结果不变（差 +0.04 pp，CI [−0.6, +0.8]）。最有用的特征是"近 32 步被选次数"，属于经典缓存策略（LRU-K/LRFU/ARC）的范畴。
4. **对照蓝图的决策关口：** Gate A 通过（问题没有被前人回答过）；Gate B 勉强通过（信号存在但弱且罕见）；**Gate C 不通过**（策略没有超过强 baseline）。建议转向蓝图预设的负面结论加现象刻画，**需要团队决定**。

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

所有 GPU 运行：NVIDIA A100-SXM4-40GB，bf16，torch 2.14.1+cu130，transformers 4.53.3，贪心解码，batch=1。

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
| `tests/` | 35 个单元测试：回放的 tie 规则、Bélády 与穷举最优一致、δ=β=0 与 LRU 逐步一致、H1 样本定义、合成信号能被检出、划分确定性、截断位置、notebook 与 `replay.py` 代码同步等。 |

---

## 4. 数据与结论推导

### 4.1 Phase A：文献核实 → Gate A 通过

**做法**：读 ArkVale（NeurIPS 2024）和 HiSparse（arXiv:2608.07009）全文，并读了 ArkVale 的驱逐 kernel（`select.cu`）。

**发现**：
- ArkVale 的驱逐只用**当前步**的 query 分数（`select.cu` 第 127、131 行），没有 recency、frequency 或历史排名状态，也没有把驱逐策略和 LRU 做消融。
- HiSparse 的设置和我们几乎相同（固定选择、host KV、B 个 slot、对比 FIFO/LRU/Bélády），但用的是纯 LRU，并明确把 LRU 与 Bélády 之间的差距留作"预测式替换策略能挽回的空间"（§4.3）。

**推导**：没有论文在选择固定的前提下，用"上次观测到的 top-K 排名"决定 page 是否留在缓存 → Gate A 通过，但只能声称增量问题，不能说"首个基于重要性的驱逐"。完整矩阵和"必须避免的说法"见 [`related_work_matrix.md`](related_work_matrix.md)。

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

---

## 5. 总结论与依据

| 问题 | 结论 | 依据 |
|---|---|---|
| Gate A：问题有没有被回答过 | **通过**（问题需收窄为增量问题） | §4.1：ArkVale 只用当前分数，HiSparse 只用 LRU |
| 缓存有没有改进空间 | **有**：相对差距 22–29%，Bélády 能消掉约三分之二的 repeat miss | §4.4、§4.5 |
| Gate B：rank 在驱逐候选上有没有增量信号 | **有，但弱且罕见**：+2.5 到 +11.6 pp，top 四分位只占 0.8% | §4.3、§4.4 |
| Gate C：rank 策略能否超过强 baseline | **不能**：完整策略 −0.45%，δ 扫描平坦，学习模型中 rank 贡献为 0 | §4.4、§4.6 |
| 差距能被简单特征挽回多少 | **约 7–12%**，主要靠近 32 步被选次数和 recency | §4.6 |

**拟定的对外说法**（蓝图的负面结论版本）：

> 在固定 Quest 式选择、Qwen3-0.6B、page 级可召回缓存的设定下，LRU 与 Bélády 之间有约 22–29% 的相对差距，但 sparse-attention selection rank 几乎无法帮助挽回这段差距：高排名 page 很少成为驱逐候选，加入 rank 的策略和学习模型都没有超过 recency/frequency baseline。简单特征最多挽回约 10% 的差距。

---

## 6. 局限

- **样本量小**：只有 8 篇 dev 文档，每个任务×长度段只有 2 篇，这一级的 CI 没有意义；所有结论都是方向性的。按文档聚类的 bootstrap 只有 8 个 cluster，CI 可能偏窄。
- **只有一个模型和一种选择器**：Qwen3-0.6B + Quest 式上界，K=64、16-token page。不能外推到 HiSparse 的 GLM-5.1 token 级设置。
- **trace 是固定的**：回放没有在线反馈，不影响生成；没有测延迟。
- **MFQA 重复**：贪心解码下的重复输出会夸大复用；停止规则会影响结果。
- **文献**：Scissorhands、InfLLM 没有读；部分细节标为"未核实"（见 [`related_work_matrix.md`](related_work_matrix.md)）。
- **ArkVale 式 baseline 缺失**：按"当前分数最低"驱逐需要记录未选中 page 的当前分数，trace v1 没有这个字段。

---

## 7. 待团队决定

1. **是否转向负面结论加现象刻画**。如果转向，layer-level GPU benchmark 可以缩减为可选，或直接不做。
2. **MFQA 的停止规则**：遇到换行就停，还是固定生成 128 token。必须在 held-out 采集之前定下。
3. **held-out 上要比较的策略**：建议预先固定为 LRU、FIFO、完整策略、逻辑回归（只在 dev 上训练）、Bélády，然后只跑一次 40 篇 held-out。
4. **是否在 trace 中记录未选中 page 的分数**，以便加入 ArkVale 式 baseline。
5. **成员名单**：proposal 写 Xihang Yu，README 写 Jimmy Yu。

---

## 8. 复现

结果摘要（不含原始 trace）放在 [`docs/results/`](results/)：`pilot/` 是单篇 4K pilot，`dev_pilot/` 是 8 篇 dev。原始 trace（pilot 1.2 MB、dev 10 MB）没有放进仓库，可用 notebook 重新采集，manifest 中记录了 trace 的 sha256。

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
