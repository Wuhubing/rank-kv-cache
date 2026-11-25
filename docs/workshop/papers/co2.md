# CO2 阅读笔记

> 来源：**找到并读了全文 PDF**（14 页，含 Appendix A–C）。
> - 出处：OpenReview forum `02zPmtcZa0` 的 PDF（`https://openreview.net/pdf/633903b460b6914d3e4e0414f6f17645bdb13d99.pdf`）。直接从 OpenReview 下载被 bot check 拦住，实际文件取自第三方镜像 Tiptree Lacuna（`lacuna.tiptreesystems.com/pdf/art_ca8cc82a9c5acd3ae8f54fc1d06fb577`）。
> - PDF 元数据：Author = Meguru Yamazaki, Shivaram Venkataraman；创建于 2024-07-19；脚注写 "Presented in Efficient Systems for Foundation Models workshop in the 41st ICML, PMLR 235, 2024"。和 ICML 页面（icml.cc/virtual/2024/39675）上的摘要逐字一致，可以确认是同一篇。
> - 正文、Table 1–3、Fig. 1–6、11、12 都读过（图是渲染后读的）。没找到 arXiv 版本，也**没找到公开代码**。

## 基本信息

- **标题**：CO2: Precise Attention Score Observation for improving KV Cache Replacement in Large Language Models
- **作者**：Meguru Yamazaki（Fujitsu、UW–Madison）、Shivaram Venkataraman（UW–Madison）
- **发表**：ICML 2024 Workshop ES-FoMo II（poster），非 archival
- **代码**：论文和项目页都没有给出，未找到

## 设置

| 项目 | 内容 |
|---|---|
| 模型 | OPT-6.7B、Llama2-7B（都是 MHA） |
| 选择单位 | **单个 token 的 KV**（H₂O 式），不是 page |
| 缓存语义 | **永久驱逐**（和 H₂O 一样，被丢掉的 KV 不能召回），没有 CPU 备份。GPU 上只保留 "cache size" 个 KV |
| 预算 | cache size 按 prompt 长度的比例：20%–100%（分析）；主评测 OPT 20%、Llama2 10%；吞吐表 OPT 10%、Llama2 5% |
| 缓存结构 | 预算分成两部分：按累计注意力分数管理的部分，加一个给新生成 KV 的 FIFO 部分。H₂O 默认 FIFO 占一半；CO2 用 ρ 调 FIFO 占比 |
| 上下文 | prompt 100–1500 token（分析），主评测固定 1500 token prompt + 生成 100 token |
| 数据集 | PG19、C4、Wikitext（指标都是 perplexity） |
| batch | 质量实验未说明；吞吐实验在不 OOM 的前提下取最大 batch：Full 2、H₂O 5/4、CO2 6/5 |
| 硬件 | **单张 NVIDIA V100 16GB** |

## 方法与基线

### 三个组件（§4）

1. **Long Measurement Step（N）**：prompt token 的去留不在第一步就决定，而是推迟 N 个生成步，用这 N 步累计的注意力分数再决定。动机：Fig. 3 中很多"重要但未缓存"的 prompt KV 在第一步不活跃，被 H₂O 第一步就丢掉了。论文没写这 N 步里 prompt KV 是否全部保留在 GPU（推测是全部保留，否则无法"推迟决定"）。
2. **累计注意力分数的衰减（α，EWMA）**，论文 Eq. 1：

   ```
   AAS_t = AAS_{t−1} · (1 − α) + NewAttentionScore · α
   ```

   - α 是 decay 参数。实验取 α ∈ {0, 0.2}。
   - **公式和实验设置不自洽**：按字面，α=0 时新分数永远加不进去（AAS 恒定），但 Table 2 把 α=0、N=0、ρ=0.5 当作 H₂O 基线（不衰减的纯累加）。合理的解读是 α=0 代表 H₂O 的纯累加，α>0 代表 EWMA，但论文没有说明。
3. **FIFO 大小调整（ρ）**：FIFO 部分大小 = Cache Size × ρ，ρ ∈ {0.25, 0.5}（0.5 = H₂O 默认）。

- 驱逐：在按分数管理的部分里驱逐 AAS 最低的 KV（继承 H₂O）。新 KV 先进 FIFO，离开 FIFO 后进入分数部分。
- "Oracle" 只是名字，**没有任何 oracle 或 Bélády 对比**。

### "重要 KV" 的定义（§3.2，分析用）

- 某个 KV 在某步被视为 activated：注意力分数 > LAS × γ（LAS 是该层的最大注意力分数），取 γ=0.01（只用 activated KV 时 perplexity 增加 <10%）。
- 以此把 KV 分为 activated&cached（蓝）、activated&uncached（橙，"该留没留"）、inactive&cached（绿，"不该留却留了"）。这是一种**基于未来访问的事后标签**，概念上接近我们的 Bélády/miss 分析，但它针对的是不可召回缓存。

### 基线

- Full KV cache、H₂O。没有 StreamingLLM、Scissorhands、TOVA、LRU、FIFO-only。

## 评价指标

| 指标 | 怎么测 |
|---|---|
| Perplexity | prompt 后生成 100 token 的 perplexity（Fig. 1、5、6、11，Table 2、3） |
| KV 分类计数 | 按 γ=0.01 阈值统计 activated/cached 四类（Fig. 3、12） |
| Replacement index | 被替换 KV 在分数部分中的位置（0–149，越大越新），看新 KV 是否很快被换掉（Fig. 4） |
| 吞吐 | token/s，V100 16GB，最大不 OOM 的 batch（Table 1） |
| 内存 | 只用"cache 比例"表示，没有 GB 数 |
| miss rate、延迟分解、FLOPs | **都没有** |

## Robustness / 消融

1. **H₂O 初步评测（Fig. 1，OPT-6.7B/PG19）**：cache 80–100% 时 perplexity 稳定，<60% 开始变差。20% cache 时，prompt 1000 和 1500 的 perplexity 分别升高 2.45× 和 3.34×。Llama2（App. A，Fig. 7）更稳健：1500 token、10% cache 时只升高 1.23×。
2. **Replacement index（Fig. 4）**：H₂O 20% cache（分数部分 150）下，**超过 90% 的被替换 KV index > 135**，即刚从 FIFO 进入分数部分的新 KV 几步内就被换掉。Llama2 同样超过 90%（App. A，Fig. 10，index 70 附近）。
3. **完整参数网格（App. B，Table 2 OPT 20% cache；Table 3 Llama2 10% cache）**。H₂O 基线行为 α=0、N=0、ρ=0.5，最后一行是 Full。

   OPT-6.7B / PG19（perplexity）：

   | α | N | ρ=0.5 | ρ=0.25 |
   |---|---|---|---|
   | 0 | 0 | **48.69（H₂O）** | 17.91 |
   | 0 | 20 | 41.41 | **17.59（最好）** |
   | 0.2 | 0 | 52.34 | 19.55 |
   | 0.2 | 20 | 46.38 | 19.03 |

   Full = 14.56。
   - **改善几乎全部来自 ρ：0.5→0.25 时 perplexity 48.69→17.91。**
   - N=20 单独作用只到 41.41；α=0.2 单独作用反而更差（52.34）。

   其他组合（最好 / H₂O / Full）：
   - OPT C4：10.09 / 11.81 / 9.20
   - OPT Wikitext：10.95 / 12.21 / 10.51（ρ=0.25 单独作用就最好）
   - Llama2 PG19：14.03（α=0.2、N=20、ρ=0.5）/ 14.53 / 11.81
   - Llama2 C4：7.91（α=0.2、N=20、ρ=0.5）/ 8.10 / 7.58
   - Llama2 Wikitext：22.59（α=0.2、N=0、ρ=0.5）/ 21.94 / 18.13，**CO2 最好的设置也比 H₂O 差**

   - 在 Llama2 上 ρ=0.25 全面变差（如 PG19 28.67），说明最优参数随模型和数据集变化很大。
   - Fig. 5 展示的是每个组合中**事后挑出的最优参数**，没有 held-out 选参。
4. **Cache 比例扫描（Fig. 6，OPT/PG19）**：H₂O 在 20% cache 时 perplexity 约 49，CO2 在 20%–100% 都约 15。OPT/C4（Fig. 11）：20% 时 H₂O +28%，CO2 +9%。
5. **吞吐（Table 1，V100 16GB，prompt 1500 + 生成 100；Full perplexity 9.20/7.58 对应 C4）**：

   | 方法 | 模型 | cache | batch | PPL | token/s |
   |---|---|---|---|---|---|
   | Full | OPT-6.7B | 100% | 2 | 9.20 | 6.20 |
   | H₂O | OPT-6.7B | 20% | 5 | 11.81 | 20.01 |
   | CO2 | OPT-6.7B | 10% | 6 | 11.12 | **28.98**（1.44×） |
   | Full | Llama2-7B | 100% | 2 | 7.58 | 2.46 |
   | H₂O | Llama2-7B | 10% | 4 | 8.10 | 11.39 |
   | CO2 | Llama2-7B | 5% | 5 | 9.97 | **15.02**（1.32×） |

   - 1.44× / 1.32× 是在**相近 perplexity 下用更小的 cache** 换来的更大 batch。
   - Llama2 上 CO2 的 perplexity（9.97）明显比 H₂O（8.10）差，所以"保持质量"的说法偏宽松。
6. **逐层 KV 分类（Fig. 12）**：第 8 层之前 activated KV 多，第 24 层之后少；inactive&cached 在每层都约 10⁶（OPT）。

## 关键结论

- H₂O 在长 prompt、小 cache 下退化严重（20% cache 时 perplexity 升高最多 3.34×），原因有三：第一步就丢掉以后会变重要的 prompt KV；累计分数不衰减，旧 KV 长期占位，新 KV 进来几步就被换掉（>90%）；FIFO 占比固定为一半，不是最优。
- CO2（推迟决定 N、EWMA 衰减 α、FIFO 占比 ρ）在多数组合上把 perplexity 拉回到 Full 的 20% 以内，cache 可以再减半，吞吐 1.44×（OPT）和 1.32×（Llama2）。
- 最优参数随模型和数据集变化很大，作者把在线调参留作未来工作。

## 对我们的启示

CO2 和我们的设置差别很大（token 级、不可召回、perplexity、小模型、短 prompt）。但它是**"对 KV 的注意力分数做 EWMA 来决定替换"**的较早公开工作，审稿人可能拿它来质疑我们 score EMA 的新颖性。另外它的消融结构（衰减 × 推迟 × recency 区）也是审稿人会期待的。

### 需要补的实验（按优先级）

1. **[P0] 三因素消融表，对标 CO2 的 Table 2/3。** CO2 的三个旋钮在我们的框架里都有对应：
   - α（衰减）↔ 我们的 EMA α。CO2 只试了 {0, 0.2}，我们已有 LODO 扫描，要把完整的 α 曲线放进论文。
   - ρ（FIFO/recency 区占比）↔ **在 B 中划出一个"最近 m 步被选中过的页一律保护"的 recency 区**，扫 m 或比例 ρ ∈ {0, 0.125, 0.25, 0.5}，看 score EMA + recency 区是否比单纯的 EMA 好。
   - N（推迟决定）↔ **新进入的页在 N 步内不可被驱逐**（admission protection），N ∈ {0, 2, 8, 20}。
   - 做 2×2×2（或更多）的全因子表，报告每个因素的单独贡献和交互。
   - 理由：CO2 的数据显示**它的改善几乎全部来自 recency 区（ρ），衰减单独作用反而有害**。这和我们"rank 几乎不提供 recency/frequency 之外的信息"的发现一致，可能也意味着我们的 EMA 增益里有一部分其实是"保护近期页"。审稿人会想知道 EMA 增益中有多少可以被一个简单的 recency 区复现。
2. **[P0] 参数选择要用 held-out，不用事后最优。** CO2 每个组合都挑事后最优参数，这是明显的弱点。我们已经用了 LODO 和 preregistered held-out，应该在论文里明确对比这一点，并给出"参数固定（α=0.5）跨 B、跨任务"的结果，说明不需要逐数据集调参。
3. **[P1] 不衰减的累计分数（H₂O 式，α→1 或 d=1）作为显式 baseline。** CO2 和 InfLLM 都显示不衰减最差。我们应该在同一张图上画 LFU、decayed count、H₂O 式纯累加（Quest 分数或注意力质量的纯累加）和 EMA，作为"衰减必要性"的证据，对应 CO2 Fig. 4 的"新 KV 很快被换掉"。可以报告我们这里的类似统计：被驱逐页的驻留时长分布和 admission 后多少步被驱逐。
4. **[P1] "重要但未缓存"分析的可召回版本。** CO2 的 Fig. 3/12 用事后 activated 标签分类 KV。我们可以做 Bélády 的对应分析：按层统计 LRU/EMA 中"被驱逐又在 h 步内被选中"的页（我们的 repeat miss）和"驻留但直到被驱逐都没再被选中"的页（dead residency）。这能让审稿人把我们的 G 和 CO2 的"橙点/绿点"直观对应起来。
5. **[P2] 永久驱逐场景下的质量对照（可选）。** 如果审稿人要求和 H₂O/CO2 系列比，可以说明：我们的 setting 保持选择固定，所有选中页都会被加载，attention 精确，不存在 perplexity 损失。代价体现在 PCIe 流量上，而不是质量上。不建议真的去跑 perplexity 对比，因为问题设定不同。

### 在 trace 模拟器里复现 CO2 策略

CO2 原本是不可召回的 token 级策略，在我们的 HiSparse 语义模拟器（选中页必须驻留）中只能复现它的**替换规则**：

- **状态**：每个驻留页一个 AAS（float）、一个 admission 时间。
- **每步 `begin()`**：
  - 对所有驻留页 `AAS[p] = (1−α)·AAS[p] + α·v[p]`，其中 v 是本步注意力信号。α=0 的实验行按 H₂O 解释为纯累加 `AAS[p] += v[p]`。
  - v 的选项：
    - (a) 真实 post-softmax 注意力质量（需要新字段 `attn_mass`，未选中页为 0）；
    - (b) Quest 分数在驻留页上的 softmax（现有 `scores` 字段即可，同 `DecayedSum(v="softmax")`）。
- **驱逐**：
  - 候选 = 未选中驻留页，排除 FIFO 区：最近进入缓存的 ⌊ρ·B⌋ 个页（按 admission 时间）不可驱逐。
  - 也排除 admission 后不到 N 步的页（Long Measurement Step 的对应物）。
  - 在剩余候选中驱逐 AAS 最小者，tie 按 LRU。
  - 如果候选不够（FIFO 区太大），退回到 FIFO 区内最早进入的页，这正是 H₂O 中"从 FIFO 退出"的语义。
- **参数网格**：α ∈ {0（纯累加）, 0.2, 0.5}，N ∈ {0, 20}，ρ ∈ {0, 0.25, 0.5}，与 CO2 的网格对齐，并加上我们 LODO 选出的值。
- **需要的 trace 字段**：
  - 用 Quest 分数版本时，现有 npz v2（`ranked`、`scores`）足够。
  - 真实注意力版本需要 `attn_mass [T,L,K]`，与 InfLLM 复现共用同一个字段。
- **注意**：CO2 的 AAS 在 token 级累加，所有 token 每步都有分数（全注意力）；在我们的设置里未选中页没有真实注意力，只有 Quest 估计值。这是和原文最大的语义差别，必须在论文里写明。

### 定位上的注意

- "对注意力分数做 EWMA 来决定 KV 替换"已有 CO2（ICML 2024 workshop）和 InfLLM（衰减累加），我们的 score EMA 不能声称是新规则。新的部分是：可召回 page 缓存、选择固定、对 Bélády 归一化的 G、按层分配、以及"rank-only 信息无用、未选中驻留页的当前分数才是关键"的消融。
- 可以引用 CO2 Table 2 说明："在不可召回的 token 级缓存中，衰减单独作用并不总是有益（OPT/PG19：α=0.2 时 48.69→52.34），收益主要来自 recency 区。"然后用我们的三因素消融说明可召回 page 缓存中的情况。

## 局限/没读清的地方

- Eq. 1 在 α=0 时退化为常数，和"α=0 即 H₂O"的实验行不一致；论文没解释实际实现。
- Long Measurement Step 期间 prompt KV 是否全部保留在 GPU、这部分内存是否计入 cache size，没写明。如果没计入，CO2 在前 N 步实际上用了比标称更大的 cache。
- 吞吐表用的是哪个数据集没说明（Full perplexity 9.20/7.58 和 Table 2/3 的 C4 行一致，推测是 C4）。另外 CO2 和 H₂O 用的 cache 比例不同（10% vs 20%），所以 1.44× 是"更小的 cache + 更大 batch"的综合结果。
- 只有 perplexity，没有下游任务准确率；prompt 最长 1500 token，远短于我们的 16–32K。
- 没有代码，无法核实实现细节（例如 FIFO 和分数区之间的迁移规则、GQA 的处理方式）。
- 全文是从第三方镜像拿到的 PDF。元数据和 ICML 页面的摘要完全一致，但没能从 OpenReview 原站直接下载核对版本（bot check）。
