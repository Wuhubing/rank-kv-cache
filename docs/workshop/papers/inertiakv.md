# InertiaKV 阅读笔记

> 来源：arXiv 2609.03515v1 HTML 全文（正文 + 附录 A–K），以及 abs 页面，阅读日期 2026-11-25。数字直接取自正文/表格；核对出的不一致处单独标注。

## 基本信息

- **标题**：What Matters for Aggressive Decoding-Time KV Eviction? Temporal Aggregation and Ranking Preservation（方法名 InertiaKV / InertiaKV-Lazy）
- **作者**：Bo Zeng, Yu Zhao, Yefeng Liu, Zhihong Lu, Xuanfan Ni, Xintong Wang（Ant Group；Alibaba Group；Universität Hamburg）
- **Venue**：abs 页面的 Comments 写明 "Accepted to EMNLP 2026 Main Conference"。arXiv 提交日期 2026-09-03。
- **代码**：https://github.com/BobTsang-NLP/InertiaKV （这次没有读代码。正文出现 "No Press"、"DecodingPress" 等字样，推测基于 NVIDIA kvpress 框架实现，未核实）

## 设置

| 项 | 内容 |
|---|---|
| 问题类型 | **decode 阶段、token 级、不可逆 eviction**。prefill 保留完整 cache，从第一个 decode 步开始压缩到 $B=\lfloor(1-\rho)L_0\rfloor$。和 sparse attention / offload 是不同的 regime（作者也这么说） |
| 主模型 | Llama-3.1-8B-Instruct（8 KV heads，GQA 4×）、Qwen2.5-7B-Instruct（4 KV heads，GQA 7×） |
| 扩展模型 | Llama-3.3-70B-Instruct（规模检查）、Qwen3-30B-A3B-Instruct（MoE stress check）、Mistral-7B-Instruct-v0.3 和 Qwen2.5-14B-Instruct（附录 stress test）。摘要称"six open-weight backbones" |
| 压缩比 | 主实验 ρ=0.9（只保留 10%）；sweep 用 50/75/90%（Fig.5），附录 B 用 0.75/0.80/0.90/0.95 |
| 粒度 | token 级，没有 page/block |
| Scorer | 最后一个 query 对所有 cached token 的注意力，在 head 上取均值：$s_t(i)=\frac1H\sum_h a_t^{(h)}(i)$。GQA 下先在 query-head 组内共享 |
| 上下文长度 | 效率实验 16k/32k/64k；NIAH 网格 10k–120k（12 个长度 × 9 个深度）；RULER 报告 4k 和 16k（从附录 C 的 Qwen2.5-14B 描述推断）；Mistral 的 LongBench-v2 上限 32k；短前缀检查 1k–8k |
| 数据集 | LongBench（8 个任务：hotpotqa、2wikimqa、qasper、musique、gov_report、multi_news、qmsum、passage_retrieval）、LongBench-v2（503 题，0–1 分）、RULER（13 个子任务）、NIAH |
| Baseline | TOVA、SnapKV、AdaKV+Expected Attention、KeyDiff、KVzip（masking 模式）、cumulative（H2O 式）、single-step（α=0） |
| Batch | 没有写 |
| 硬件 | H100 和 H200 |

## 方法

**聚合规则（§3.1）**：
- Single-step：$m_t=s_t$
- Cumulative：$m_t=m_{t-1}+s_t$
- **EMA**：$m_t=\alpha m_{t-1}+(1-\alpha)s_t$，**α 是旧状态的权重**（和我们的 $e\leftarrow a e+(1-a)z$ 约定相同，和 PRR 相反）。默认 α=0.8。
- 每步在 cache 超过 B 时，驱逐 $m_t$ 最小的 token。

**关键实现细节（附录 E，非常重要）**：EMA **不是**每个 decode 步更新一次，而是在每个 "eligible layer" 之后依次更新同一个共享状态：
$m_{t,\ell}=\alpha m_{t,\ell-1}+(1-\alpha)s_{t,\ell}$，$m_{t,0}=m_{t-1,K}$。
因此上一步状态的权重是 $\alpha^{K}$（K 是 eligible 层数），第 ℓ 层的权重是 $(1-\alpha)\alpha^{K-\ell}$，越靠后的层权重越大。
- 作者自己也承认 α "couples layer weighting and temporal retention"，并且 α=0 等于"只用最后一层"，所以这个消融不是纯时间维度的。
- **论文没有说明哪些层是 eligible、K 等于多少。** 如果 K 接近全部层（Llama-8B 有 32 层），那么 $0.8^{32}\approx 8\times10^{-4}$，跨步记忆几乎为零，所谓"temporal EMA"实际上主要是**层内的指数加权平均**。这是解读这篇论文时最重要的保留意见。
- $m_0=0$；EMA 在 prefill 阶段不启用。因此**第一个 decode 步的（层加权）分数决定了初始排名**，也决定了一次性从 $L_0$ 砍到 B 的那 90% 驱逐。之后每步通常只需驱逐约 1 个 token（新生成的那个）。

**Lazy refresh（Eq.6）**：只在 $t\in\mathcal R_r=\{t\ge1:(t-1)\equiv0\pmod r\}$ 这些步更新 EMA，其他步令 $m_t=m_{t-1}$（关掉 layer hook，不采集注意力），但仍然照常驱逐。新 token 在非 refresh 步的初始分数设为现有 token 的平均分。默认 r=4（Lazy4）。

**Score-Free**：只在第一个 decode 步按层依次处理分数，然后冻结排名；从第 2 步起完全不再打分。新 token 用当前平均 momentum 初始化。

**理论（Proposition 1，附录 A）**：假设每个 token 的分数在时间上 i.i.d.（均值 $\bar s(i)$，方差 ≤σ²），token 之间相互独立，$\mu_0=0$。对于差距 $\delta_{ij}>0$ 的两个 token：期望差收敛到 $\delta_{ij}$；$\mathrm{Var}[\mu_T(i)-\mu_T(j)]\le\frac{2(1-\alpha)}{1+\alpha}\sigma^2$；$\limsup\Pr[\mu_T(i)<\mu_T(j)]\le\frac{2(1-\alpha)\sigma^2}{(1+\alpha)\delta_{ij}^2}$（Chebyshev）。
**Corollary 2**：若替代 scorer 满足 $\mathbb E[\tilde s_t(i)]=f(\bar s(i))$，且 f 严格单调，则期望 EMA 的渐近排名不变；统一正缩放对排名完全不可见。作者承认这只是解释性的界，因为真实注意力在时间上相关。

**"预测质量"如何评估**：这篇论文不预测 top-K，评估的是**驱逐集合的相似度**和**任务分数**：
- scorer 之间 top-B 集合的 pairwise Jaccard（在 decode 步和层上取平均）；
- scorer 与注意力在第 15 层的 Spearman ρ，以及 10% 预算下的 top-B Jaccard；
- top-B churn（相邻步 top-B 集合的 1−Jaccard）；
- raw/EMA Jaccard，以及 raw-top-k 被驱逐的比例；
- 任务分数，配合 paired bootstrap 95% CI 和 paired t-test。

## 评价指标

- **质量**：LongBench（0–100）、LongBench-v2（0–1）、RULER（0–100）、NIAH ROUGE-L-F。Score-Free 和 Lazy4 相对 full refresh 的 Δ 都给了 95% CI。
- **集合稳定性**：Jaccard、Spearman ρ、churn、raw-top evicted rate。
- **效率**：decode 吞吐（tok/s）、Lazy4 相对 full refresh 的 speedup、prefill 时间、decode 时间构成（score update 占比）、hook 级 profiling。
- **内存**：保留的 decode-side token 数（64,029 → 6,401）、prefill 峰值显存（29.91 GB，与 full cache 相同；TOVA 为 22.88 GB）。
- 没有 FLOPs/MACs 数据，只说 "layer-scoring operations 减少 4×"。

## Robustness / 消融

1. **预算 sweep（Table 1，Llama-3.1-8B RULER，Full 94.59）**：TOVA 在 50/25/10% 预算下为 77.10/65.83/48.34；SnapKV 68.79/43.26/24.33；AdaKV+EA 89.24/71.16/28.06。InertiaKV 的 50/75/90% 结果见 Fig.5（只有图，没有数值表）。
2. **Scorer 扰动（§5.1）**：
   - Llama-3.1-8B、90% 压缩：Attention / VNorm / Entropy 三个 scorer 的驱逐集合 pairwise Jaccard 为 0.97。
   - Qwen2.5-7B LongBench 8 个任务：Entropy 加权的 Δ 恰好为 0；VNorm 平均 |Δ|≤0.33；α=0 时下降 1.33；随机驱逐只有 12.45。
   - Llama-3.1-8B：VNorm ≤0.35；α=0 时从 43.88 降到 42.11。
   - **Llama-3.3-70B（Table 2）**：EMA 平均 56.74，VNorm 56.40（最大偏差 0.67，在 qasper），α=0 为 49.37（−7.37；各任务降 4–13 分，qasper 从 45.53 降到 32.48）。
3. **排名相关性边界（第 15 层）**：VNorm 和 Entropy 的 ρ≈0.97，质量保持；KeyDiff ρ=0.49、Jaccard 0.38，LongBench 降 9–14 分；key norm ρ=0.00、recency ρ=0.12，在检索任务上崩溃；learned bilinear scorer（327,680 参数，自蒸馏 MSE+KL，λ=0.1，τ=1.0）ρ≈0.05、Jaccard≈0.08，LongBench −1.94，RULER −45.06（4k 时 −38.0，16k 时 −52.2；niah_multiquery 在 16k 从 99.2 掉到 3.4）。
4. **α sweep（Table 5，Llama-3.1-8B，90%）**：

   | α | 0 | 0.3 | 0.6 | 0.8 | 0.9 | 0.95 |
   |---|---|---|---|---|---|---|
   | LongBench | 42.11 | 43.29 | 43.87 | 43.88 | **44.05** | 43.99 |
   | LongBench-v2 | 0.296 | **0.298** | 0.290 | 0.294 | 0.294 | 0.288 |
   | RULER | 53.97 | 63.55 | 67.03 | 75.60 | 78.55 | **78.72** |

   Qwen2.5-7B：α=0.9 和 0.8 相比，LongBench 46.05 vs 46.01，LongBench-v2 0.247 vs 0.254，RULER 78.28 vs 79.43，所以保留 0.8。Mistral 上 α 从 0.8 调到 0.9，RULER 从 72.76 升到 76.79。**最优 α 随任务变化**：检索类任务偏好大 α，LongBench-v2 偏好小 α。
5. **Cumulative vs EMA（Table 12，Cum − EMA）**：Llama 上 LongBench +0.08、LongBench-v2 −0.010、RULER +2.01；Qwen 上 −0.01 / −0.010 / **−7.66**。cumulative 有竞争力，但不占优。
6. **Lazy refresh（Table 14，Lazy4 − Full，95% CI）**：Llama LongBench +0.005 [−0.044, 0.053]，RULER −0.151 [−0.223, −0.078]（唯一显著的退化），LongBench-v2 0.000；Qwen LongBench −0.026 [−0.111, 0.051]，RULER +0.005 [−0.024, 0.033]，LongBench-v2 −0.004。速度：64k 时 Llama 1.34×、Qwen 1.46×；16k–64k 范围内 1.36–1.44×（Fig.11）。score update 在 16k 时占 Full refresh decode 时间的 36%，Lazy4 降到 14%。
7. **Score-Free（Table 3，Llama LongBench）**：平均 Δ=+0.03（44.70 vs 44.67），CI 跨 0；8 个任务中 7 个不显著；multi_news −0.61 [−1.01, −0.21]，p=0.003；passage_ret +0.50（不显著）。Qwen 平均 Δ=−0.01（最大 |Δ| 0.47）。NIAH 1k/2k/4k/8k 前缀下 ROUGE-L-F 都是 0.71，和 Full、Lazy4 相同。
8. **Baseline 比较（Table 4，90%）**：Llama 上 InertiaKV / Lazy4 / TOVA / SnapKV / AdaKV / KVzip / KeyDiff 的 LongBench 为 43.88/43.88/34.50/33.97/33.78/41.59/35.26（Full 45.87），RULER 为 75.60/75.45/48.34/24.33/28.06/**86.44**/63.82（Full 94.59）。Qwen 上 LongBench 46.01 vs Full 48.67，RULER 79.43 vs Full 93.00（KVzip 81.40）。KVzip 的代价：prefill 31.54 s vs 4.86 s，decode 10.9 vs 39.1 tok/s，保留 64,029 vs 6,401 个 token。
9. **辅助机制（附录 B，Llama）**：static reserve 和 first-eviction rescue 带来负收益，压缩越激进差距越大：RULER 上 InertiaKV 相对 Aux 的优势在 CR 0.75/0.80/0.90/0.95 时为 +16.19/+17.56/+21.87/+22.80。Table 8 中 "No Mom." 的 RULER 为 53.61，InertiaKV 为 75.60。
10. **泛化 stress test（附录 C/D）**：
    - Qwen3-30B-A3B：RULER 66.26（Full 95.51，TOVA 33.56）。
    - Mistral-7B：RULER 72.76（Full 89.38，H2O 式 75.10）。
    - Qwen2.5-14B：RULER 67.70（Full 94.50，H2O 式 74.86）；multi-key/multi-query 子任务崩溃（4k 时 niah_multikey_3 从 100 掉到 0）。
    - EMA 包装在其他 scorer 上（TOVA、KeyDiff、KNorm，固定预算 1024/2048）几乎没有作用（变化 ≤0.5）："EMA 不是通用 wrapper"。
11. **机制诊断（附录 I，Table 13，Llama，100 个样本）**：EMA top-k churn（0.0005–0.0056）远低于 raw churn（0.018–0.041）。在 RULER 上 α 从 0.3 增到 0.9，trace 准确率从 0.47 升到 0.64；在 LongBench-v2 上不变（0.28–0.29）。§5.3 另报：raw 注意力 top-B churn 34%，EMA 降到 8%；75% 的 decode 步驱逐集合变化 ≤1 个 token。一条 79 步的 trace 中，top-B 集合始终和第一步完全相同。LongBench-v2 上 97.6%（Llama）/ 95.0%（Qwen）的样本对时间规则不敏感。自适应 α 的尝试失败：warmup 期噪声/漂移比 η=50–600，几乎总是选 0.8 或 0.95；ranking disagreement 和 α 敏感性的相关只有 0.14。

### 核对出的不一致 / 可疑处

- Llama LongBench 的 full-refresh InertiaKV 平均分：Table 3 是 44.67，但 Table 4、Table 5（α=0.8）、§5.1 都是 43.88。Lazy4 在 Table 3 中是 44.76，在 Table 4 中是 43.88。可能 Table 3 来自附录 K 的 "clean-root rerun"，但正文没有说明。
- Lazy4 speedup：Fig.10 / 摘要写 1.34–1.46×（Qwen 64k 时 1.46×），Fig.11 写 "1.36–1.44× across context lengths and models"，两个范围互相矛盾。
- Table 3 中 Lazy4−Full 在 Llama LongBench 上是 +0.09（44.76−44.67），Table 14 是 +0.005，量纲或数据来源可能不同。
- "temporal" 的解释混入了层加权（见方法部分）。另外 Score-Free ≈ Full 很可能只是因为 90% 的驱逐都发生在第 1 步，之后集合几乎不动，并不能说明"时间维度的惯性"本身起了作用。

## 关键结论

- 在 90% 激进压缩、TOVA 式不可逆驱逐的设置下，与注意力高度相关（ρ≈0.97）的 scorer 改动（VNorm、Entropy）经过 EMA 聚合后，驱逐集合几乎不变（Jaccard 0.97，任务 |Δ|<0.35；70B 上 ≤0.67）。显著改变排名的 scorer（KeyDiff ρ=0.49，key norm、recency、learned scorer）则大幅掉分（RULER 最多 −45）。
- EMA（α=0.8）相对 single-step：LongBench +1.8，RULER +21.6（Llama）。最优 α 依任务而定（RULER 偏好 0.9–0.95，LongBench-v2 偏好 0.3）。cumulative 有竞争力但不稳定（Qwen RULER −7.66）。
- 排名极其稳定，所以 Lazy4 几乎不损失质量，decode 快 1.34–1.46×；Score-Free（只打一次分）平均 Δ=+0.03。
- 相对 TOVA/SnapKV/AdaKV/KeyDiff 大幅领先（RULER 75.6 vs ≤63.8）；KVzip 质量更高（86.44），但慢得多、保留的 token 多得多。

## 对我们的启示

**1. 定位：问题本身不同**

- InertiaKV 是**有损、不可逆、token 级 eviction**，评价看任务质量。我们是**无损 offload 下的 page 级 cache 替换**：miss 的页会被取回，Quest 选择结果不变，评价看 repeat-miss，并以 Bélády 为参照（G）。
- 我们不应该和它比任务分数，而应把它当作"EMA 排名惯性"这一现象在 eviction 场景中的先例来引用。"selector/注意力分数的 EMA 是好的驻留优先级"这件事不能写成我们的发现。

**2. 定位：magnitude vs ordering 是我们的核心可区分点**

- InertiaKV 固定 EMA、变动 per-step scorer，结论是"只要 per-step 排名近似保持（ρ≈0.97），集合就几乎不变"。它的 Corollary 2 只保证**期望值**的单调变换不改变排名。我们正在加的 all-page 百分位排名 EMA 是**逐步的单调变换**：每步内顺序不变，但 EMA(rank) ≠ rank(EMA(score))，跨步的相对幅度被抹平了。所以我们的对照正好测试 InertiaKV 没有分开的那个问题：EMA 的收益来自"排名/顺序信息"，还是来自"幅度（与第 K 名的相对距离）信息"。
- 写作时可以这样定位：InertiaKV 发现保序扰动被吸收（同时幅度也大致保留了，ρ≈0.97）；我们的控制实验在保持顺序的同时**完全去掉幅度**。
- 我们预注册的"last selection rank 几乎不增加信息"和 InertiaKV 的 recency scorer（ρ=0.12）崩溃不矛盾：后者是把 recency 当作唯一 scorer，前者是在 recency/frequency 之上的增量。

**3. 定位：我们的设计是纯时间 EMA，比 InertiaKV 干净**

- 我们每层独立做 EMA，每步更新一次，不存在 InertiaKV 那种 "α 同时控制层权重" 的混淆。论文里可以明确指出这一点，作为方法上的优势：我们的 α 敏感性是纯时间维度的。
- 另外，我们的 z 归一化到第 K 名的分数，相当于每步一个逐层标量。InertiaKV 的 Entropy 加权正是 per-layer 标量，Δ=0。但在我们的跨步 EMA 里，这个标量会改变不同步之间的相对权重，所以不能先验地认为它无影响，需要消融（见下面第 3 项实验）。

**4. 需要补的实验（按优先级）**

1. **α（我们的 a）敏感性 sweep，包含两个端点**：a∈{0（=只看当前步）, 0.3, 0.6, 0.8, 0.9, 0.95, →1（cumulative/frequency）}。在 calibration split 上选，在 held-out 上报告 G。逐任务或逐数据子集报告最优 a 是否不同。目的：InertiaKV 显示最优 α 强烈依赖任务，cumulative 有时更好。审稿人一定会问我们的 28%/40% 对 a 是否稳健，以及和 frequency/cumulative 的关系。
2. **百分位排名 EMA 控制（已计划）+ 原始未归一化分数 EMA**：组成 {raw score, K-th 归一化 score, all-page 百分位排名, last-selection rank} 四个输入的 2×2 式对照，同一 a、同一 B、同一 held-out，报告 G 和 paired bootstrap CI。目的：直接回答 magnitude vs ordering，并检验 K-th 归一化这个逐步标量是否重要（InertiaKV 提示 per-layer 标量在单步内不改变顺序，但我们是跨步的）。
3. **Quest 分数 vs 真实注意力的排名一致性**：逐层计算 page 级 Spearman ρ 和 top-K Jaccard（对标 InertiaKV 第 15 层的 ρ 谱：0.97 保持、0.49 退化）。目的：说明 Quest 分数在 Qwen3-0.6B 上处于"保序区"还是"重排区"，也让读者知道 score-EMA 的 G 是相对 Quest 选择的，而不是相对真实注意力的。
4. **Churn / 稳定性诊断（对标 InertiaKV Table 13）**：对 raw 当前步 top-K、score-EMA top-B、LRU 常驻集合，分别报告相邻步 churn（1−Jaccard），以及"真实 top-K 中被驱逐的比例"。目的：用和 InertiaKV 相同的语言解释为什么 score-EMA 关闭了部分 gap（更低的 boundary churn），以及剩余 gap 是不是 lag（相关性转移）。
5. **Lazy update 和 "只更新被选中页" 变体**：EMA 每 r∈{2,4,8} 步更新一次；或者只更新本步被 Quest 选中的页（我们目前更新所有常驻页）。目的：InertiaKV 显示 r=4 几乎无损；如果我们的 G 也稳健，就说明 score-EMA 的额外开销可以压低。这是系统可部署性的论据。
6. **Score-Free 式冻结排名 baseline**：用 prefill 结束（或第一个 decode 步）时的 Quest 分数排名作为固定驻留优先级。目的：InertiaKV 的 Score-Free 几乎不掉分，说明在短生成任务里第一步排名已经足够。我们需要证明在我们的 decode 长度下，冻结排名明显差于 score-EMA，否则"时间聚合"这一说法站不住。顺便报告我们的生成长度分布。
7. **按 prompt 的 G 分布 / 敏感子集分析**：InertiaKV 发现 LongBench-v2 中只有 2.4–5% 的样本对时间规则敏感。我们应报告 G 的逐 prompt 分布（中位数、分位数），以及 score-EMA 不如 LRU 的 prompt 比例，避免一个均值掩盖异质性。
8. **跨模型**：至少加一个模型（Qwen3-1.7B/4B 或 Llama-3.2 系列）。InertiaKV 在 Mistral 和 Qwen2.5-14B 上出现了明显的迁移问题，提示结论可能依赖模型；我们只有 0.6B，外部效度不足。

**5. 统计报告**：学 InertiaKV 的做法，对所有关键 Δ（score-EMA vs LRU、rank-EMA vs score-EMA）给 paired bootstrap 95% CI，按 prompt 配对。

## 局限 / 没读清的地方

- **"eligible layers" 是哪些层、共有多少层没有说明**，因此无法判断 α=0.8 下跨步记忆的实际权重 $\alpha^K$ 是多少。这决定了"temporal aggregation"这一核心叙述有多少是真正的时间聚合，有多少只是层聚合。
- 生成长度、batch size、RULER 和 LongBench 的具体上下文截断都没有完整写出（RULER 的 4k/16k 是从附录推断的）。
- 大部分驱逐发生在第一个 decode 步（90% 一次性砍掉），后续只是每步驱逐新 token，所以"排名惯性 / Lazy / Score-Free 几乎无损"在长生成场景是否成立，没有被检验（作者在 Limitations 中承认）。
- Fig.1、Fig.5、Fig.6、Fig.10、Fig.11 只有图，HTML 中拿不到数值（例如 InertiaKV 在 50/75% 下的 RULER 分数）。
- 上面列出的几处表间数字不一致（44.67 vs 43.88；1.34–1.46 vs 1.36–1.44）。
- learned scorer 失败的原因（排名错位 vs 训练不足）无法从文中分开；作者也说不能排除存在更好的非注意力 scorer。
- 代码没有读。
