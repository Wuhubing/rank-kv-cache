# When Fancy Eviction Fails 阅读笔记

> 来源：arXiv 2609.28870v3 PDF 全文（2026-11-07 修订版，20 页，含附录 A–D）。阅读日期 2026-11-25。Fig.2/11 主图的大多数命中率只以柱状图给出，正文写出的数值都列在下面；只有图的标为"只有图"。

## 基本信息

- **标题**：When Fancy Eviction Fails: Rethinking Cache Replacement For LLM Prefix Reuse
- **作者**：Yiyu Liu, Minlan Yu, Juncheng Yang（Harvard University）。Juncheng Yang 是 S3-FIFO、SIEVE、libCacheSim 的作者
- **Venue**：arXiv preprint（cs.DC），v1 2026-09-24，v3 2026-11-07。页面没有写会议，格式像 USENIX/ACM 系统会议投稿
- **开源**：模拟器 PrefixSim（https://github.com/cacheMon/PrefixSim，C++，基于 libCacheSim）；trace 在 HF `harvardMadsys/2026_prefixcache`
- **注意任务差异**：这篇讲的是**跨请求的 prefix cache**（miss 的代价是重新 prefill），**不是**请求内的稀疏注意力 page 选择。正文 §7 明确把 Quest/InfiniGen/H2O 列为"approximate KV management"，与本文区分。我们引用它时只能借用"LRU 难以超越、与 Bélády 有差距"这一方法论上的类比，不能说它是同一个问题。

## 设置

| 项 | 内容 |
|---|---|
| Trace | **FreeInference**（7.0 天，327.5K 请求，10.5B 处理 token，0.63B unique；34.3% 请求属于多轮会话；平均请求长度 32.0K，P99 219.3K；以 agentic 为主）；**Chutes**（130.9 天，515.8K 请求，9.7B token，2.23B unique；平均 18.7K，P99 98.3K；人类多轮对话、agentic、API 混合）。两条合计超过 20B 处理 token |
| 泛化 trace | Qwen Bailian 四条（To-B、To-C、Coder、Thinking，各 2 小时窗口）；AgentX（393 个 Claude Code 会话回放，人为设定并发 16，hash 只在会话内有效，因此是下界） |
| 对照 workload | Wikipedia 2019 CDN（207.6M 请求，21 天）、CloudPhysics w01 块 I/O（215.8M 请求，6.3 天） |
| Block | 16 token/block；Qwen3-Coder-30B tokenizer；请求截断到 256K（FreeInference 跳过 4.3%，Chutes 跳过 0.03%） |
| 容量区间 | HBM 单副本 24/48/96 GiB（24 GiB = 16,384 block，是原生上下文的下限；48 GiB ≈ 1×H200 0.8 显存比例；96 GiB ≈ 1×B200）；DRAM/SSD 全局池 0.25–12 TiB（1 TiB = 699,051 block）。附录 A 扫了 65 个容量点 |
| 模拟器 | PrefixSim 加入了"请求驻留约束"：prefill 时该请求的所有 block 必须同时在内存中，否则主动淘汰。论文说命中率与原生 vLLM "very close"，**但没有给出数值**；速度比 vLLM 的 Python 实现最高快 160× |
| 真机验证 | vLLM + Qwen3-Coder-30B，单张 NVIDIA H200，48 GiB cache，回放 FreeInference 的 10,000 条请求 |

## 方法与基线

**14 个在线策略，按设计原则分组**：Recency（LRU，作为基线）；Quick demotion（ARC、SIEVE、S3-FIFO、S4-FIFO、LIRS）；Analytic（LHD）；Frequency（LFU、W-TinyLFU）；Learned（LeCaR、LRB、3LCache）；Prefix 专用（Workload-aware [Qwen Bailian 论文]、AsymCache）。**离线上界**：Bélády（命中最优）。

**新提出的内容**：
- **compute-savings ratio**：每次命中按避免的重算 FLOPs 加权。原来的命中率等于常数代价模型。代价模型有三种：uniform、按 Qwen3-Coder-30B 实测 FLOPs、理想线性（代价 ∝ 深度，附录 C）。
- **离线最优**：ILP 精确解，加上近似算法 **BeladyCompute**：Score = ComputeIntensity × TimeUntilNextAccess，淘汰分数最高的 block。
- **在线 RandomCompute**：随机采样一组 block，淘汰 ComputeIntensity / TimeSinceLastAccess 最低的。
- **PartialNode RandomCompute**：从节点开头按连续块淘汰，减少 hole（碎片）。
- **Block 级 vs Session 级淘汰粒度**的比较。

## 评价指标

- **主指标**：block 命中率（模拟器，确定性）。差异用"百分点"报告。
- **compute-savings ratio**：按 FLOPs 加权的命中率（模拟器）。
- **真机**：平均 TTFT、中位数/P99 TTFT、prefill 吞吐（token/s），在 vLLM/H200 上测。图注说明 **TTFT 和吞吐是在两次不同的运行中分别测的**。
- **workload 特征**：对象寿命 CDF、会话时长、5 分钟窗口 Jaccard 重叠、复用间隔、频率与复用间隔/寿命的关系、P(再次复用 | 已命中次数)、会话 KV 占用的 CCDF/Lorenz 曲线、stack distance 可达比例。
- **碎片**：每请求 hole 数、hole 长度分布（附录 D）。
- **开销**：只定性讨论元数据（例如 12 TiB 池需要索引 8.4M 个 16-token block），没有测策略本身的 CPU 时间。
- **重复次数/方差**：模拟结果是确定性的，没有误差棒。真机实验**没有说明重复次数、warmup 或置信区间**，只给单次均值、中位数、P99。敏感性分析（附录 B）是把同一组 73,637 个请求按不同并发重新调度，属于确定性扫描，不是随机重复。

## Robustness / 消融（带数字）

1. **主结果（Fig.2，两条生产 trace × 4 容量）**：大多数策略（ARC、LHD、LeCaR、3LCache、LRB、Workload-aware、AsymCache）与 LRU 持平或略差；quick-demotion 类强烈依赖 trace（有的 trace 上小赢，有的输几个点）；LFU/W-TinyLFU "完全崩溃"；Bélády 明显高于所有在线策略（只有图）。附录 B：FreeInference 平均约 12 个并发会话，在整个容量扫描中，除 W-TinyLFU 外的所有策略都在 LRU **±约 1 个百分点**以内。
2. **泛化（Fig.11）**：1 TiB 时，三条 trace 上 LRU 与最好的在线策略相差都在 **0.4 个百分点**以内。但 24 GiB 时 LRU 在任何 trace 上都不是第一：Qwen To-B 上 S3-FIFO 和 ARC 为 **0.491**，LRU 为 **0.403**。LRU 追平最优在线策略所需容量：FreeInference 37 GiB，其它 trace 60–462 GiB。强制 miss 占比：Qwen 34–54%，FreeInference 6.0%，AgentX 3.75%。1 TiB 时的 OPT：To-B 0.534，Coder 0.664，AgentX 0.962。
3. **密集容量扫描（附录 A，65 个点）**：Qwen To-C 上 LIRS 从比 LRU 高 4.2 点变成在 724 GB 时低 10.1 点，之后又收敛。W-TinyLFU 在 FreeInference 上不单调：97 GB 0.866，906 GB 0.759，12 TB 0.933。论文自己的结论是："comparisons at a few capacities can hide both policy crossovers and nonmonotonic behavior."
4. **并发敏感性（附录 B，48 GiB）**：并发从 8 到 32，Bélády 0.926→0.713，**LRU 0.898→0.517**，LRU 与 Bélády 的差距从约 3 点扩大到约 20 点。48 GiB 内可达的复用比例从 92.5% 降到 50.1%。并发 32 时 S3-FIFO 0.573 > LRU 0.517。池子区间并发 1024 时只有 54.1% 可达，S3-FIFO、ARC、LIRS 反超。
5. **workload 机制**：多轮会话中位时长 55 s，P90 843 s，P99 约 4 h；会话内请求间隔中位 8.2 s，99.7% < 22 min。一次性对象占 53.0–55.7%。第二次命中后再次复用的概率从 69.4% 升到 88.7%。频率与复用间隔无关（Fig.5a 是平的），所以频率不适合排序，但可以用于准入。前 10% 会话占 76.2% KV 字节，前 1% 占 20.3%；会话 KV 中位 0.07 GiB，P99 12.2 GiB，最大 234 GiB。
6. **compute-aware（Fig.12，24 GiB，实测代价模型）**：RandomCompute 0.638，比 LRU 高 **10.4 点**，并且超过命中最优的 Bélády（线性模型下高 2.9 点）。BeladyCompute 与 ILP 最优相差 1.03/0.41/0.08 点（24/48/96 GiB）。FreeInference 上 Bélády 与 BeladyCompute 的差距为 **+12.6 点**，其余五条 trace ≤1.1 点，因此"跑两个 oracle"可以用来诊断是否值得做 compute-aware。线性代价模型（附录 C）：RandomCompute 0.646 vs LRU 0.524（+12.2），BeladyCompute 0.745 vs Bélády 0.599（+14.6）。1 TiB 时所有策略（包括两个 oracle）相差都在 0.2 点以内。
7. **碎片（附录 D）**：RandomCompute 每请求平均 9.21 个 hole，P99 143 个，最多 281 个；PartialNode 的 P99 为 2，最多 16，平均 hole 数降为 1/8.7。总重算量几乎不变（258.0M vs 257.9M block）。PartialNode 在 24 GiB 只损失 0.17 点 compute savings。
8. **真机（Fig.13，vLLM/H200/48 GiB/10K 请求）**：PartialNode RandomCompute 相比 LRU，平均 TTFT **1.04 s vs 1.29 s（−19.9%）**，prefill 吞吐 65.6K vs 55.2K tok/s（+18.8%）。但**中位 TTFT 变差（171 ms vs LRU 112 ms）**，P99 从 26.0 s 降到 16.8 s。block 粒度的 RandomCompute TTFT 高 4.0×（5.24 s），吞吐低 3.9×（14K tok/s），说明模拟器里的增益可能被真实系统的碎片开销完全抵消。
9. **深度相关代价（Fig.7–8）**：同样计算 8K token，从头算 166 ms / 71.1 TFLOP；接在 64K 已缓存前缀之后算，966 ms / 493.3 TFLOP。所以命中率不等于节省的 TTFT。
10. **粒度（Fig.14）**：24 GiB 时 session 级淘汰使 LRU 命中率损失 10.5 点、RandomCompute 的 compute savings 损失 18.7 点；48 GiB 时分别损失 3.3 和 5.5 点；池子区间几乎没有差别。

**内部不一致**：§5 开头写 LRU "reliably converging to the optimal offline ceiling on every trace we measured"，但摘要和 §3.2 强调"substantial headroom to Bélády"，附录 B 也显示并发 32 时差距约 20 点。合理的解读是：大容量下 LRU 接近 OPT，受限容量和高并发下差距很大。引用时应只引用带容量条件的说法。

## 关键结论

1. 在真实 prefix cache trace 上，LRU 很难被 13 种"fancy"策略超越；频率类策略最差；但 LRU 与 Bélády 之间仍有可观的差距，而且这个差距随容量缩小、并发增加而扩大。
2. 原因是结构性的：复用由会话内稳定节奏主导，复用间隔短且方差小，所以 recency 是很强的预测信号；频率只反映会话进行了多久。
3. 要缩小与 OPT 的差距，应在 LRU 基础上做有针对性的补充：一次性 prompt 占多数时用 quick demotion；代价随深度变化时用 compute-aware 淘汰（必须用连续块，避免碎片）；HBM 用 block 级，大池子可以用 session 级。
4. 命中率不是最终目标。同样的命中率可以对应不同的 TTFT；模拟器里的胜者在真机上可能因为碎片而大败。

## 对我们的启示

1. **我们的结果和它在叙事上同构，可以直接借用它的框架**：一个强 recency 基线（LRU），一组更复杂的策略大多打平，一个离线 Bélády 上界，再讨论剩下的差距有多少是在线可以拿到的。我们的 score-EMA 拿到了约 28% 的 LRU–Bélády 差距，在这个框架下是一个正面结果；selection rank 是预注册的负面结果，对应它的"frequency 类失败"。写作时可以引用它来说明，"LRU 难以超越"在 prefix cache 里也成立，我们是在请求内的稀疏注意力 page cache 里给出对应的证据（并说明两者问题不同）。
2. **机制解释是审稿人最看重的部分**：这篇论文的价值主要来自 §4 的"为什么"（复用间隔分布、频率与复用间隔无关、P(再复用 | 已命中次数)）。我们也应提供同类诊断：page 被选中的复用间隔分布（以 decode step 计）、"已被选中次数"与下次选中间隔的关系、selection rank/score 与下次复用间隔的关系。后者正好可以解释为什么 rank 失败、为什么 score-EMA 部分有效。用这些图回答"为什么 recency 已经很强、score 补充了什么"。
3. **扫容量要密，不能只看几个点**：附录 A 证明少数几个容量点会掩盖策略交叉和非单调。我们的 B（每层页预算）应扫得足够密（例如 8–10 个点以上），并报告 gap-closed 随 B 的曲线，检查 score-EMA 的优势是否在某些 B 下反转。per-layer budget allocation 也应在多个总预算下比较。
4. **"命中率/字节 ≠ 延迟"要正面写**：它的 Fig.7–8 和 Fig.13（block 级 RandomCompute 在模拟里赢、真机上 TTFT 4.0×）正好支持我们"PCIe 字节 −9–19% 但延迟持平"的结论。应在论文里明确区分模拟/计数指标和真机指标，并给出把字节换算成时间的模型（字节/带宽 vs 每步计算时间）。
5. **oracle 诊断法可以借用**：它用"Bélády vs BeladyCompute 的差距"来判断某个信号值不值得做。我们可以用类似方法：比较 Bélády 与"只用 recency 的最优"（或 Bélády 与 LRU）之间的差距，以及 score-EMA 拿到了其中多少，来界定"分数信号的上限价值"。
6. **统计报告是这篇论文的短板，也是我们可以超过它的地方**：真机实验只有单次运行，没有误差棒，TTFT 和吞吐还分开测。我们有 21 篇 held-out 文档和预注册，应报告每文档配对差、bootstrap CI、胜/负/平计数，真机延迟要给多次重复的中位数与 IQR，并说明 warmup。
7. **并发维度**：附录 B 显示并发是 LRU–OPT 差距的主要驱动因素（3 点 → 20 点）。我们只做了 batch 1，单请求内的 page 复用节奏可能比多请求竞争 GPU pool 时更有利于 recency。可以说明这是一个限制，或者在模拟中把多条文档的 trace 交错，加一个"多请求共享 pool"的实验。
