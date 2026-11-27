# PRR: Predict, Reuse, and Repair 阅读笔记

> 来源：arXiv 2606.30389v1 HTML 全文（含附录 A.1–A.8），阅读日期 2026-11-25。下文数字均直接取自正文/表格；作者计算错误或前后不一致的地方单独标出（见"Robustness / 消融"的"内部不一致核对"一节）。

## 基本信息

- **标题**：Predict, Reuse, and Repair: Accelerating Dynamic Sparse Attention for Long-Context LLM Decoding
- **作者**：Tianyu Wang, Gourav Rattihalli, Aditya Dhakal, Junbo Li, Zhiwei Ren, Dejan Milojicic, Longfei Shangguan（University of Pittsburgh；HPE Labs）
- **Venue**：arXiv preprint（cs.LG，2026-06-29 提交）。页面上没有写会议或接收信息；正文有 "Limitations" 节和 "page limit" 字样，像是 ACL 系列投稿格式，但没有确认。
- **代码**：https://github.com/Tianyu9748/Incremental_FlashAttention （这次没有读代码）

## 设置

| 项 | 内容 |
|---|---|
| 模型 | GLM-4-9B-1M、GLM-Z1-9B、DeepSeek-R1-8B（应为 R1-Distill-Llama-8B，原文只写 R1-8B）、Llama3-8B-1M、Qwen3-14B、Qwen3-32B，共 6 个。正文主要报告 GLM-4-9B，其余放附录 |
| Selector（DSA） | Quest、InfLLM-v2。NSA 需要从头训练，所以排除。Quest 扩展到 GQA 的方法（A.1）：每个 KV head 用组内 query 的均值 $\bar Q=\frac1G\sum_g Q_g$，score $s_p=\sum_i\max(\bar Q_i m_{p,i},\bar Q_i M_{p,i})$，每个 (KV head, page) 一个分数，组内共享 top-K |
| Block/page size、K | A.2 只说"与 Quest、InfLLM-v2 原论文相同"（block size、token budget），**没有给出具体数值**。kernel 微基准另外扫了 block size 16–256、token budget 1K–8K |
| 上下文长度 | 动机实验 16K–512K；速度预算 profiling 用 4K/8K/16K/32K/64K；主实验（Table 1）各 benchmark 的实际长度没写 |
| 数据集 | LongBench、InfiniteBench、RULER、AIME（2024，30 题；A.3 另加 MathArena）、MATH500 |
| Batch | 主实验只用 batch=1（Limitations 里明说）。A.4 的利用率 profiling 用了 BS=2/4/8/16（总长 128K） |
| 硬件 | H100，CUDA 12.8，TP=2；稀疏注意力 baseline 后端是 FlashInfer BlockSparseAttention；half precision |

## 方法

**任务定位**：PRR 不是 cache 替换，也不是 eviction。它在每一步、每一层，趁 selection 还在算的时候，先对预测集合 $P$ 做推测性注意力，等真实 top-K 集合 $A$ 出来以后，再用 online-softmax 把漏掉的块 $M=A\setminus P$ 补进来（repair）。所以 predictor 的作用是**提前一步预测下一步的 top-K 集合**。

**预测器：Holt 双指数平滑（level + trend）**，对象是 DSA 自己算出的块重要性分数 $IS_i^t$（Quest 的 upper-bound score / InfLLM-v2 的 compressed-attention score），不是真实注意力：

- 初始化：块 $i$ 在 $\tau$ 步进入 cache 时，$\ell_i^\tau=IS_i^\tau$，$v_i^\tau=0$
- 预测（Eq.2）：$\widehat{IS}_i^t=\ell_i^{t-1}+\gamma v_i^{t-1}$，$\gamma\in[0,1]$
- 更新（Eq.3）：$v_i^t=\beta(IS_i^t-\ell_i^{t-1})+(1-\beta)v_i^{t-1}$；$\ell_i^t=\alpha IS_i^t+(1-\alpha)\ell_i^{t-1}$
- **注意 α 的约定和我们相反**：PRR 的 α 是**新观测**的权重。PRR 的 α∈{0.2,…,0.8} 对应我们 $e\leftarrow a e+(1-a)z$ 里的 a∈{0.8,…,0.2}。
- 分数用的是**原始分数**，没有归一化（不像我们归一化到第 K 名的分数）。从上下文看，每一步对所有块都更新，因为 DSA 每步给所有块打分。
- $P$ = 预测分数的 top-$|P|$，$|P|\le\delta|A|$。

**在线标定（§4.2）**：每个 prompt 用 prefill 阶段的分数轨迹 $IS\in\mathbb R^{T_p\times N_p}$ 做网格搜索：α∈{0.2,0.4,0.6,0.8}，β∈{0.1,…,0.5}，γ∈{0,0.25,0.5,0.75}，共 80 组。目标是**分数加权命中率**（Eq.4）：
$H(\theta)=\frac1{T_p}\sum_\phi \frac{\sum_{b\in P_\phi\cap A_\phi}IS_{\phi,b}}{\sum_{b\in A_\phi}IS_{\phi,b}}$。
搜索放在 CPU 上，和 prefill 重叠执行。总耗时 2.4 ms（16K）/ 11.2 ms（64K），落在关键路径上的部分是 0.06/0.28 ms，单次预测 0.01 ms。从 "the fit for the final layer" 这句话推测是**每层分别标定**，但正文没有明说。

**推测预算 δ（§4.4）**：每个"模型-硬件-DSA"组合离线 profiling 一次，测 selection 延迟和推测注意力延迟在不同 (context length, δ) 下的值，建成查找表；运行时选"推测注意力能被 selection 窗口盖住"的最大 δ。不在网格里的长度用多项式回归插值。profiling 耗时不到 5 分钟。**论文没有报告任何 δ 的具体取值。**

**Repair kernel（§4.3）**：改 FlashAttention forward，额外把 $\ell$（分母）和 $m$（running max）写回 HBM；repair kernel 对 $M$ 中的每个块执行 Eq.6–8 的 log-sum-exp 合并。FLOPs 和访存只随 $|M|$ 增长。

**预测质量怎么评**："overlap rate" = 预测集合与真实 top-K 的重叠（Table 4/7）。从 Eq.1（$\min|A\setminus P|$）和 "missed 32%" 的说法看，它应该是 **recall $|P\cap A|/|A|$**，而且 $|P|$ 可以大于 $|A|$（δ>1）。论文没有给出 overlap 的公式，也没有说统计是按层/按 head 平均，还是按步平均。另一个基线 "consecutive overlap" 是上一步 top-K 与这一步 top-K 的重叠（$|P|=K$）。

## 评价指标

- **预测质量**：overlap rate（recall，见上）；连续步重叠率（Table 5）。
- **延迟**：每 token decode 延迟，以及相对串行 DSA 的 speedup（Table 1）；动机实验做了分阶段延迟拆分（selection/attention/FFN）。
- **Kernel**：相对 FlashInfer BlockSparseAttention 的 speedup（Table 3）。
- **资源利用率**：SM、L2 带宽、DRAM 带宽利用率（Fig.4、Table 6）。
- **开销**：标定搜索耗时、预测耗时、profiling 耗时（A.5/A.6）。
- **准确率**：声称和原 DSA "相同"，**但全文没有任何准确率表**。
- 没有 FLOPs/MACs，没有显存占用，没有吞吐（只有 batch=1 的延迟）。

## Robustness / 消融

1. **动机：延迟拆分（GLM-4-9B，H100）**：selection→attention 这条串行路径在 16K 时占 decode 延迟约 60%，512K 时约 71%；selection 本身从约 4 ms 涨到约 8 ms；引言说 selection 最多占每 token 延迟 41%。
2. **连续步重叠（Fig.3 + Table 5，6 模型 × 5 benchmark × 2 DSA）**：Quest 的模型平均在 0.636（DeepSeek-R1）到 0.697（GLM-Z1）之间；InfLLM-v2 在 0.655（Qwen3-32B）到 0.698（GLM-4/GLM-Z1）之间。单格最低 0.629（DeepSeek-R1/Quest/LongBench），最高 0.718（GLM-4/InfLLM-v2/AIME）。正文写"平均约 68%"。跨模型、跨任务都很平，任务之间的差异只有几个百分点。
3. **GPU 利用率**：单请求 512K 时 SM、L2、DRAM 利用率都低于 40%。BS=2→16（总长 128K）时，Quest 的 SM 利用率从 6.7% 到 8.7%，L2 BW 43–46%，DRAM BW 37–38%；InfLLM-v2 也差不多。
4. **EMA overlap（Table 7）**：Quest 的模型平均 97.75–98.24%（6 模型均值 98.02%）；InfLLM-v2 97.52–98.86%（均值 98.44%）。单格最低 96.46%（GLM-4/InfLLM-v2/RULER），最高 99.40%。跨模型、跨任务几乎没有变化。
5. **端到端 speedup（Table 1）**：Quest 的模型平均 1.28×（Qwen3-32B）到 1.42×；InfLLM-v2 1.31×（Qwen3-32B）到 1.56×（GLM-4）。最大单格 1.64×（GLM-4/InfLLM-v2/LongBench，对应"up to 40%"延迟降低）。模型越大收益越小（Qwen3-32B 最低）。
6. **分阶段消融（Table 2/8）**：S1 = 直接复用上一步 top-K：1.00–1.04×（只要漏一个块就要全量重算，所以几乎没有收益）。S2 = 加 repair kernel：Quest 平均约 1.20×，InfLLM-v2 约 1.20–1.32×（GLM-4/InfLLM-v2 平均 1.30×）。S3 = 再加 EMA predictor：Quest 1.28–1.42×，InfLLM-v2 1.32–1.56×。这里 S2→S3 的增益**同时包含**两部分：EMA 预测，以及更大的 δ（kernel 更快所以放得下）。两者没有分开。
7. **Kernel 微基准（Table 3，32 Q heads / 8 KV heads / d=128 / 128K 上下文）**：block size 16/32/64/128/256 的平均 speedup 分别为 2.19/2.22/2.27/2.31/2.11×；token budget 1K 时 1.03–1.17×，8K 时 3.64–3.71×。
8. **没有做的 sweep**：α/β/γ 的敏感性（只说网格搜索，没有报告选出了什么值，也没有报告固定参数的效果）；δ 的 sweep；K 或 block size 对 overlap 的影响；上下文长度对 overlap 的影响；逐层 overlap；"无 trend（γ=0）" vs Holt 的对比；同样 $|P|$ 下 EMA 与上一步复用的公平对比。

### 内部不一致核对（重点）

- **~91% vs ~98%**：§5.2 说 "EMA predictor achieves an average top-K overlap rate of roughly 91%"，又说更快的 kernel 让 PRR 能放进更多块，"lifting the overlap rate from 91% to 98% at no added latency cost (§5.5)"。但 §5.5、Table 4、Table 7、§5.3（"S3 attains 98%"）、A.8 全部只报告约 98%，**没有任何表格、任何配置给出 91%**，也没有给出两种情况各自对应的 δ。最合理的解读是：91% 是在 FlashInfer kernel 允许的较小 δ 下得到的，98% 是用自家 kernel 放大 δ 之后得到的。如果这样理解，那么 **98% 是在 $|P|>K$（δ>1）下测的 recall**，不能和 $|P|=K$ 时 68% 的"上一步复用"直接比；$|P|=K$ 时 EMA 的 overlap 论文没有报告，这个数字甚至可能低于 91%。结论：引用时应写成"在未报告的推测预算下约 98% recall"，不能写成"EMA 以 98% 准确率预测 top-K"。
- **平均 speedup**：摘要/引言写"1.42× over Quest and 1.56× over InfLLM-v2 across multiple LLMs"，但这两个数其实是 **GLM-4 一个模型**的平均值。§5.2 写 6 模型平均 1.35×（Quest）/1.47×（InfLLM-v2），我用 Table 1 的模型平均重算得到 1.385× / 1.467×，**Quest 的 1.35× 和表格对不上**（应约为 1.38–1.39×）。
- **Table 2 与正文**：Table 2 的 S3 平均是 1.56×，正文写 "1.55× speedup on average"。Table 1 和 Table 8 的 S3 平均有 0.01 的出入（GLM-4/Quest 1.42 vs 1.41，DeepSeek-R1/Quest 1.41 vs 1.40）。
- **A.3**：正文说"across all LLMs and benchmarks, the overlap stays below 0.7"，但 Table 5 里有 0.700、0.701、0.707、0.708、0.718 等格子不低于 0.7。A.8 又改口说"roughly 70%"。
- **"精确语义"与"保留 wasted block"矛盾**：§1 说 "Blocks in $P\setminus A$ are retained as additional low-priority context"，同时又声称"preserves the semantics of standard DSA / same accuracy"。如果 $P\setminus A$ 也进入了 softmax，输出就是 $P\cup A$ 上的注意力，和标准 DSA 不完全等价。而全文没有准确率表，这个问题没法核实。

## 关键结论

- 两种 training-free DSA（Quest、InfLLM-v2）在 6 个模型、5 个 benchmark 上，相邻 decode 步的 top-K 块平均重复约 64–72%（均值约 68%），跨模型、跨任务很稳定。
- 用 Holt 型 EMA（level + trend，按 prompt 用 prefill 标定）预测下一步的块分数，再在 δ 预算下取 top-$|P|$，对真实 top-K 的 recall 约 97.5–98.9%（6 模型均值约 98.0% / 98.4%）。
- 光有局部性不够：上一步复用（68%）加"漏一块就全量重算"几乎没有收益（1.00–1.04×）；收益来自 repair kernel（S2，约 1.2–1.3×）加 EMA 预测和更大 δ（S3，1.28–1.56×）。
- 每 token decode 延迟最多降 39%（1.64×）。batch=1，TP=2，H100。

## 对我们的启示

**1. 定位：我们不能声称的内容**

- "selector 分数的 EMA 能很好地预测下一步 top-K" 这一点 PRR 已经做过（6 模型，Quest 与 InfLLM-v2 都有）。"Quest 选择存在强时间局部性（约 68% 连续重叠）" 也已经是已知事实，而且同时有 InfiniGen、FlexiCache 等先例。我们只能把这些当作背景引用，不能当作发现。

**2. 定位：我们和 PRR 的本质区别**

- PRR 解决的是**每步重新决定 P 的推测/预取问题**：P 可以每步完全换掉，代价是带宽和计算。我们解决的是 **GPU 只能放 B 个页、换入换出有成本的 cache 替换问题**：常驻集合只会因为 miss 逐步变化，评价基准是 Bélády 的 repeat-miss 下界（G 指标）。
- PRR 的 98% recall 相当于"每步可以自由挑 δK 个块"的上界场景；我们的常驻集合在 B/K = 1.5（B=96）或 2（B=128）时，相当于 δ = 1.5 / 2 的**有状态、受换页约束**版本。论文里应把 PRR 写成"预测质量的上界参照"，把我们的贡献写成：(a) 在有容量约束、可回取（lossless offload）的设置下，用 Bélády 作参照的定量 gap closure；(b) per-layer allocation；(c) 预注册的"排名 vs 幅度"对照。
- 两者都属于**无损**范畴（miss 的页会被取回，注意力在真实 top-K 上算），这和 InertiaKV 的不可逆 eviction 不同。

**3. 需要补的实验（按优先级）**

1. **在 miss rate 之外，并列报告 PRR 式的 prediction-overlap 指标**（最高优先级，成本低）：
   - 对每层每步计算"上一步 top-K 对这一步 top-K 的 recall"（对标 PRR 的 68%）；
   - 计算 score-EMA 预测分数 top-K（δ=1）以及 top-B（δ=B/K）对真实 top-K 的 recall，对标 PRR 的 98%；
   - 计算 LRU/rank/score-EMA 常驻集合对真实 top-K 的 recall（就是 1 − 每步 miss 比例），并和 Bélády 常驻集合的 recall 放在一起。
   - 目的：让读者能把我们的 Qwen3-0.6B / page 16 / K=64 和 PRR 的数字直接对照，同时说明**高 recall（自由集合）并不能换来低 miss（有状态集合）**，从而论证为什么需要 Bélády/G 这样的评估。如果我们的连续重叠明显低于 68%（小模型、K=64），也必须报告出来。
2. **α 敏感性 sweep，并加入 PRR 式的标定**：a∈{0.2,0.4,0.6,0.8,0.9,0.95}，在 calibration split 上选、在 held-out 上报告；再加一个"按 prompt 用 prefill 轨迹选 a"的变体（PRR §4.2 的思路，在 prefill 的 Quest 分数上模拟替换，挑 G 最大的 a）。目的：审稿人会问 28%/40% 是不是调参的结果，而 PRR 认为参数需要按 prompt 自适应。
3. **Holt trend 变体（γ>0）作为替换分数**：$\hat e=\ell+\gamma v$。目的：检验"预判上升趋势"能不能在替换场景里多关一些 gap；如果没有提升，可以写成"在替换场景下 trend 项无益"。这是一个可报告的、对 PRR 设计的负面结果。成本低。
4. **Quest 分数与真实注意力的排名一致性**：每层算 Quest page score 与 page 级真实注意力质量的 Spearman ρ，以及 top-K Jaccard。目的：PRR 和我们都把 Quest 分数当作"真值"，但 Quest 只是上界估计；如果在 Qwen3-0.6B 上 ρ 偏低，我们的 G 衡量的是"对 Quest selector 的 cache 命中"，不是"对真实重要 KV 的命中"，这一点必须在论文里讲清楚（同时把它和 InertiaKV 的 ρ 边界联系起来）。
5. **跨模型**：至少再加一个模型（例如 Qwen3-1.7B/4B 或 Llama-3.2-1B/3B），报告 G 与 overlap。PRR 的结论建立在 6 个 8B–32B 模型上；我们只有 0.6B 一个模型，这是最明显的外部效度短板。
6. **K / page size 的小 sweep**（K∈{32,64,128}，page∈{16,32}，各取一两个点）：PRR 完全没做这一块，我们补上就是增量信息。
7. （可选）**和 PRR 的时延模型对接**：用 miss 数 × PCIe 单页传输时间估算 offload 场景下的关键路径延迟，说明 G 的系统含义。

## 局限 / 没读清的地方

- Quest 和 InfLLM-v2 的 block size 与 K/token budget 没有写具体值（只说同原论文）；主实验每个 benchmark 的上下文长度未知。
- overlap rate 没有给公式，也没有说明是按层/head/步的哪种平均；对应的 δ 没有报告；91% 和 98% 的来源对不上（见上）。
- prefill 阶段的 "top-K 选择轨迹" $A_\phi$ 怎么定义没写清楚（DSA 通常在 prefill 不做逐 token 选择）；标定是每层还是每 head，只能从字面推测。
- 标定选出的 (α,β,γ) 分布没有报告，所以不知道 trend 项是否真的有用，也不知道 α 大致落在哪个区间。
- 声称准确率不变，但没有准确率数据，而且"保留 $P\setminus A$"可能改变输出。
- 只测了 batch=1；TP=2 下 selection/attention 的拆分细节没写。
- 代码仓库这次没有读。
