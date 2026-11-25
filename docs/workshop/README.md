# Workshop 投稿准备：现状、robustness 要求与补充实验

截至 2026-11-25。本分支 `docs/workshop-readiness` 基于 `efficient-policy` 的 `083f758`，**只加文档，不改任何代码和结果文件**。各论文的详细阅读笔记在 [`papers/`](papers/)。

> 本文中的论文数字来自 `papers/` 下的笔记，每篇都通读过全文，但部分数字是从图上目测读出的，正式引用前请再对照原文核一遍。FLOPs/MACs 是按模型配置估算的，没有实测。

---

## 1. 目前的状态

### 1.1 两条研究线

| | 排名研究（`stage1-2`） | 分数 EMA（`efficient-policy`） |
|---|---|---|
| 问题 | 上次被选中时的排名，能否在 recency 和 frequency 之外预测 page 复用 | 每步所有 page 的 Quest 分数做 EMA，能否缩小 LRU 与 Bélády 之间的差距 |
| 设置 | Qwen3-0.6B，Quest 式 page 选择（16 token，K=64），B=128；LongBench 的 MFQA 和 GovReport，4–16K；20 篇 dev，40 篇 held-out；L4，batch 1 | 同一模型和选择器；GovReport 和 QMSum，16–32K，开启 thinking，最多 1024 步；14 篇 dev，21 篇 held-out；B=96 |
| 结果 | **负**：完整策略只比 LRU 少 0.39% repeat miss（预注册目标 10%）；LRU 与 Bélády 相差约 27% | **正**：EMA 的 G=0.281 [0.277, 0.285]，21/21 篇胜过 LRU；加按层分配后 repeat miss 比 LRU 少 16.6% |
| 系统 | 无 | L4 实测：PCIe 字节少 9.3–19%，融合 kernel 下延迟与 LRU 打平（约 62 ms/token） |
| 修复 PR | [#6](https://github.com/Wuhubing/rank-kv-cache/pull/6)：S2 的 β 混淆。事后分析显示，单独加排名在 held-out 上少 213 次 repeat miss，占 0.14% | [#7](https://github.com/Wuhubing/rank-kv-cache/pull/7)：加入全 page 排名对照（`allrank`）和同分配 oracle，修复 F1–F9 |

### 1.2 尚未完成、投稿前必须补上的
1. **全 page 排名对照（`allrank`）的结果**：代码在 #7，需要 14 个 longctx dev NPZ（在共享 Drive 上）。这个结果决定论文的核心表述是"分数大小有用"还是"全局排序有用"。
2. **同分配 Bélády**（`validation/alloc_oracle.py`）：之前按层分配的 G 用的是均匀预算下的锚点，要换成相同分配下的 oracle。
3. **GPU 重测**：#7 修好了 warmup 和显存的测量口径，但 L4 上的数字还没有重新测。
4. 所有数据只有一个模型、一种 GPU、batch 1。

### 1.3 文献定位（详见 [`papers/`](papers/) 和仓库外的综述报告）
- **不能主张**的：EMA 打分本身（A2SF、Cascading KV、ReST-KV、InertiaKV 都用过）；"用历史重要性做可召回替换"（InfLLM 已做过）；"上一步复用弱、EMA 强"（PRR 已报告）。
- **可以主张**的（措辞用"据我们所知"）：
  - 在 Quest 式**页级**选择上测出 LRU 与 Bélády 的差距。
  - 预注册的、控制 recency 和 frequency 后的排名增量检验，结果为负。
  - 以 miss 和 PCIe 字节为目标的按层分配。
  - 刻画"字节减少不等于延迟减少"在什么条件下成立。
  - Quest 选择随 GPU 型号变化。

---

## 2. Workshop 级别需要达到什么

### 2.1 目标 workshop（详见 [`papers/workshop_cfps.md`](papers/workshop_cfps.md)）
| Workshop | 契合度 | 格式 | 现状 |
|---|---|---|---|
| ICBINB（ICLR 系列） | 最高，专收负结果，ICLR 2026 一届明确列了推理效率 | 正文 ≤4 页；必须包含假设、方法、定量的负面结果、失败原因分析 | 2027 届未公布，ICLR 2027 的 workshop 名单在 2026-12-07 公布 |
| ML for Systems（NeurIPS） | 中 | 短文 | 2027 届未公布 |
| ES-FoMo（ICML）、ENLSP（NeurIPS） | 高 | 4 页左右 | 2026 年都没办，2027 年是否恢复未知 |

**实际含义**：workshop 不要求 MLSys 主会那样的端到端加速，但要求**问题清楚、方法严谨、结论有统计支撑、失败原因有分析**。ICBINB 的评审标准是 rigor & transparency、新意、清晰度和主题契合度，"只报提升、不分析原因"会被扣分。

### 2.2 Workshop 级的 robustness 底线
从读过的论文看，同类工作最常被挑的点和对应的底线如下：

| 维度 | 底线（workshop） | 加分（接近主会） | 依据 |
|---|---|---|---|
| 预算 | B/K 从 1 扫到 6 以上，画 miss 曲线，并在每个预算上报 G | 等 miss 容量：LRU 要多大预算才能达到 EMA 的 miss | FlashOffloading 声称"容量比策略重要"；HiSparse 显示缓存翻倍的效果超过整个 LRU–Bélády 差距 |
| 基线 | hit 优先的 LRU、FIFO、Random、Bélády；ArkVale 的"当前分数"策略；InfLLM 的衰减分数（代理版） | SIEVE、S3-FIFO；CO2 式的 recency 保护区；lazy 更新 / 冻结排名 | HiSparse、ArkVale、InfLLM、CO2、InertiaKV |
| 信号消融 | raw、z、allrank、last rank、当前分数并排比较；α 从 0 扫到 1 | 加 PRR 的趋势项；幂律衰减 | PRR、InertiaKV、NoHairCache |
| 统计 | 按文档配对 bootstrap；每篇文档的 G 分布 | 噪声注入：分数变差时 G 退回 LRU，而不是更差 | 读过的系统论文大多不报方差，这一项是我们的优势 |
| 模型 | 至少 1 个主模型，加 1 个更大的模型（7–8B）做验证 | 2 个模型族以上 | PRR 用了 6 个模型，HiSparse 和 FlashOffloading 只用 1 个大模型 |
| 上下文和任务 | 两类任务以上；有一档 ≥32K | 64K–128K；RULER、SCBench 多轮 | HELMET 指出只用 NIAH 已经不够 |
| 系统 | 字节和 miss 是主指标；延迟只作为诚实的刻画；策略本身开销用 µs 和 FLOPs 报出 | batch > 1，带宽受限的场景，延迟分解 | ECHO：预取在微基准上 1.51×，端到端最多 4%；Rethinking KV（MLSys 2025）：HF 里测的加速不可靠 |
| 正确性 | 证明替换策略不改变生成结果（逐 token 一致） | | ArkVale、ECHO 都强调无损 |

### 2.3 FLOPs / MACs 应该怎么报

替换策略**不改变**模型本身的计算量：选择器和注意力对所有策略都一样。所以 FLOPs/MACs 不是比较策略优劣的指标，它的作用是**证明策略开销可以忽略**。审稿人要看的是下面这张开销账。以下按 Qwen3-0.6B 的配置估算：28 层，16 个 query head，8 个 KV head，head dim 128；上下文 L 个 token，N=L/16 个 page。

| 项目（每层每步） | 公式 | 32K 上下文时（N=2048，K=64，B=96） | 所有策略相同？ |
|---|---|---|---|
| Quest 选择器打分 | N × Hq × d × 2 MACs | ≈ 8.4 M MACs；28 层 ≈ 0.47 GFLOPs/token | 是 |
| 稀疏注意力 | (K+1) × 16 × Hq × d × 2 MACs | ≈ 4.3 M MACs | 是 |
| 模型线性层（每 token） | ≈ 2 × 非嵌入参数量 | ≈ 0.9 GFLOPs/token（另有 LM head 约 0.3） | 是 |
| **EMA 更新（策略开销）** | ≈ 5 FLOPs × B | ≈ 480 FLOPs；28 层约 1.3×10⁴ FLOPs/token，约为模型的 10⁻⁵ | 否 |
| 驱逐选择 | 在 B−K 个候选里取最小值 | O(B) 次比较 | 否 |
| 策略元数据显存 | B 个 float32 | 每层 384 B，28 层约 10.7 KB；对比 KV 缓存 96 × 64 KiB × 28 ≈ 168 MiB | 否 |
| **PCIe 传输** | miss × 64 KiB | 主指标，B=96 时少 9.3–19% | 否 |

建议论文里放一张这样的表，同时给出**实测**的策略开销（每层多少 µs，以及关掉 host 取数时的延迟，用来隔离策略本身），并给出端到端延迟分解：选择器、取数、注意力、其余计算。估算显示**选择器本身的计算量和模型线性层同一量级**，这一点也值得写：长上下文下选择器才是额外开销的大头，不是替换策略。

---

## 3. 需要补充的实验（按优先级）

标注说明：**[CPU]** 用现有 trace 重放即可；**[NPZ]** 需要 14 个 dev 或 21 个 held-out NPZ；**[GPU]** 需要重新采集 trace 或实测。所有新分析只在 dev 上做探索；held-out 只能跑**事先写好、只跑一次**的确认性分析，并标明是在预注册之后补做的。

### P0：没有这些就不投（全部是 CPU 重放，约 1–2 周）
1. **allrank 对照** [NPZ]：比较 z、allrank 和只看选中 page 的 rank，α 网格和 LODO 选参都相同。代码已在 #7。
2. **同分配 Bélády** [NPZ]：按层分配的 G 改用相同分配下的 oracle（`validation/alloc_oracle.py`）。
3. **预算扫描** [NPZ]：B/K ∈ {1, 1.25, 1.5, 2, 3, 4, 6, 8}，画 LRU、EMA、Bélády 和 compulsory 下限四条 miss 曲线，并在每个点报 G；加等 miss 容量。用来回应 FlashOffloading。
4. **基线补齐** [NPZ]：
   - ArkVale：当前分数驱逐，固定 2 个 sink page 和 2 个最近 page。
   - InfLLM：衰减注意力质量。目前只能用 softmax(Quest 分数) 做代理，要明确标注。
   - hit 优先 LRU（已有）、FIFO、Random、SIEVE、S3-FIFO。
5. **信号和 α 消融** [NPZ]：raw、z、allrank、last rank、当前分数 × α ∈ [0, 1]。α=0 等于当前分数（ArkVale 式），α→1 接近累积频率。
6. **CO2 式析因实验** [NPZ]：衰减 × recency 保护区 × 准入后最短驻留时间，回答"EMA 的收益有多少能被一个简单的 recency 区复现"。
7. **统计** [NPZ]：每篇文档的 G 分布图；配对 bootstrap；噪声注入曲线。
8. **PRR 式重叠指标** [NPZ]：报告 EMA 预测集与下一步 top-K 的重叠（集合大小固定为 K），和 miss rate 放在一起，方便与 PRR 对照。

### P1：显著增强可信度（需要采集新 trace，约 2–3 周 GPU 时间）
9. **第二个模型** [GPU]：一个 7–8B 模型（如 Qwen3-8B 或 Llama-3.1-8B），沿用同样的 schema v2 trace。
10. **更长的上下文、更多任务** [GPU]：一档 64K；加 RULER 和 SCBench 多轮。替换策略的差异最可能在 SCBench 上显现。
11. **page 大小和 K** [GPU]：page ∈ {8, 16, 32, 64}，K ∈ {32, 64, 128}。
12. **trace 加字段** [GPU]：
    - `attn_mass`：每个选中 page 的 softmax 后注意力质量，用于真正的 InfLLM 基线。
    - 真实注意力：用来算 Quest 分数与真实注意力的秩相关（回应 InertiaKV）。
13. **采样解码** [GPU]：temperature > 0、3 个 seed，检验结论不依赖贪心解码。排名研究里 MFQA 重复回答的教训说明这一项必要。
14. **跨 GPU 翻转分析** [CPU，已有 A100/L4 数据]：检验跨硬件翻转的是否集中在 top-K 边界附近的 page，从而把"选择随 GPU 变化"这个附带发现做实。

### P2：主会或系统方向才需要（workshop 可选）
15. **GPU 重测** [GPU]：用 #7 修好的 harness 在 L4 上重测；报告策略开销 µs、延迟分解，并证明输出逐 token 一致。
16. **带宽受限的场景** [GPU]：batch > 1，更小的 B，更长的上下文，更大的模型，或限速的链路。目标是找到字节减少能变成延迟减少的区间，或者证明这样的区间在常见硬件上不存在，这本身也是一个结论。
17. 对照 ECHO 或 HiSparse 这类真实系统。

---

## 4. 建议的论文形态

**方案 A（ICBINB，4 页）**
- 标题方向："Selection rank does not help KV page replacement, but score history does"。
- 主体：预注册的排名负结果和失败原因（排名靠前的 page 几乎不离开选择集，占候选 0.7%），对照 EMA 的正结果（必须有 allrank 对照和预算扫描），以及"字节减少但延迟不变"的刻画。
- 需要：P0 全部，P1 至少第 9、13 项。

**方案 B（ML for Systems 或 2027 届的效率类 workshop）**
- 主体：EMA 加按层分配；以 miss 和字节为主指标，配合系统开销账。
- 需要：P0 全部，P1 的第 9–12 项，以及 P2 的第 15 项。

两种方案都建议先挂 arXiv 占位：2026 年 8–11 月这个方向连续出了 HiSparse、AVSG、FlashOffloading 等多篇预印本。

---

## 5. 阅读笔记索引

| 笔记 | 论文 | 对我们最重要的一点 |
|---|---|---|
| [hisparse.md](papers/hisparse.md) | HiSparse（arXiv 2608.07009） | 设置几乎相同但是 token 级；B=2K 时 LRU 13.4%、Bélády 8.2%；只有一条 trace；需要对齐它的 hit 优先 LRU |
| [flashoffloading.md](papers/flashoffloading.md) | FlashOffloading（arXiv 2611.07542） | "容量比策略重要"是拿不同容量比较得出的；Score 在每个预算上都比 LRU 好 4–8%；没有 oracle |
| [infllm.md](papers/infllm.md) | InfLLM（NeurIPS 2024） | 公开代码默认是普通 LRU，论文里的衰减分数是另一个变体；miss 率曲线只用了一篇 GovReport |
| [arkvale.md](papers/arkvale.md) | ArkVale（NeurIPS 2024） | 驱逐规则就是我们的"当前分数"策略，Quest 上界正是它的 digest；没有 miss 率、消融或 oracle |
| [co2.md](papers/co2.md) | CO2（ICML 2024 ES-FoMo） | 收益几乎全部来自 FIFO 保护区的大小，单用衰减反而更差；需要做析因实验。全文来自第三方镜像 |
| [prr.md](papers/prr.md) | PRR（arXiv 2606.30389） | 98% 是在预测集大于 K 时的召回率，和 68% 不可比；文中 91% 与表格不一致 |
| [inertiakv.md](papers/inertiakv.md) | InertiaKV（EMNLP 2026） | 它的 EMA 主要跨层而不是跨步；大部分驱逐发生在第一步解码 |
| [echo.md](papers/echo.md) | ECHO（OSDI 2026） | EMA 用于预取而不是驱逐；GPU 池命中率已经 0.97–0.99，所以预取端到端只多 4%，和我们的"字节少但延迟不变"同理 |
| [fancy_eviction_fails.md](papers/fancy_eviction_fails.md) | When Fancy Eviction Fails（arXiv 2609.28870） | 跨请求的前缀缓存，只能类比；LRU 稳健，但与 Bélády 的差距随并发增大 |
| [rethinking_kv_compression.md](papers/rethinking_kv_compression.md) | Rethinking KV Cache Compression（MLSys 2025） | 系统主会接收"实际不加速"型结论的先例；HF 里测的加速不可靠 |
| [workshop_cfps.md](papers/workshop_cfps.md) | 各 workshop 的 CFP | ICBINB 最契合；2027 届都未公布 |
