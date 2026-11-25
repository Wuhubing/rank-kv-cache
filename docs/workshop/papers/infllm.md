# InfLLM 阅读笔记

> 来源：arXiv:2402.04617v2 全文 PDF（17 页，含 Appendix A–C）。逐段读过正文和附录，Fig. 4 是按 200 dpi 渲染后读的数。另外读了官方代码 `thunlp/InfLLM` 的 `inf_llm/attention/context_manager.py` 和 `config/*.yaml`（main 分支，2026-11-25 抓取）。

## 基本信息

- **标题**：InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory
- **作者**：Chaojun Xiao\*, Pengle Zhang\*, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, Maosong Sun（清华、MIT、人大）
- **发表**：NeurIPS 2024（arXiv v2，2024-05-28）
- **代码**：https://github.com/thunlp/InfLLM

## 设置

| 项目 | 内容 |
|---|---|
| 模型 | Mistral-7B-Instruct-v0.2（原生 32K）、Llama-3-8B-Instruct（原生 8K）；附录有 Vicuna（4K）；§4.4 有 Llama-3-8B-Instruct-Gradient-1048k |
| 选择单位 | **memory unit = 连续 128 token 的块**（l_bs=128）。每块取 r_k=4 个 representative token：按块内 local window 中查询对它的 dot product 均值 r_m 打分，取最高的 4 个 |
| 选择 K | Mistral 每步加载 96 个 unit，Llama-3 加载 32 个（§4.2）。另有 local window 4K、initial tokens 128 |
| GPU 缓存大小 | App. C.1："GPU cache 大小为 32，是每步加载 unit 数的两倍"。**这和 §4.2 的 96/32 对不上**（按 96 算，cache 应为 192）。公开的 yaml 里 `topk: 16`、`max_cached_block: 32`，与 C.1 的"两倍"一致 |
| 编码方式 | 输入按 chunk 编码，chunk 512（1024K 实验用 2048），解码逐 token。**查找和缓存更新在每个 chunk（编码）和每个解码步都发生** |
| 上下文长度 | ∞-Bench 平均 145.1K，95% 分位 214K；LongBench 95% 分位 31K；passkey 32K–1024K |
| 数据集 | ∞-Bench（R.PK、R.Num、R.KV、Choice、QA、Sum、Math.F）；LongBench（16 个子任务）；**miss rate 实验只用 "a sample of data from GovReport"** |
| batch | 未明说，代码按 batch 建 unit（`num_units = batch_size`），评测应是 batch 1 |
| 硬件 | NVIDIA A100 或 A800，half precision |
| 粒度细节（代码） | `num_units = batch_size`、`unit_size = num_heads`：**同一层所有 head 共享一套块选择和一个 GPU 缓存**，块分数在 head 维求和（`global_score.sum(dim=-1).sum(dim=1)`）。这和我们按层共享、head 平均的设计一致 |

## 方法与基线

### 块选择（不是替换策略，但决定访问流）

- relevance：sim(X, B) = Σ_{i∈当前 token} Σ_{j=1..r_k} q_i · k̂_j^B，取最高的 k_m 个块。
- 位置编码：memory 里所有 token 和当前 token 的距离都设为 l_L（local window 大小）。

### GPU 缓存替换（App. A，Eq. 1）

论文正文称"least recently used strategy"，但附录给出的实际规则是**带衰减的注意力质量分数**：

```
s_b ← s_b · d + Σ_{j=1..l_X} Σ_{i=1..l_bs} attention_score(q_{j+l_P}, k_i)
```

- attention_score 是 softmax 之后的注意力概率（0 到 1），求和覆盖当前所有 query token（编码时是整个 chunk）和块内所有 key。
- 每次 attention 计算之后更新分数，按 s_b 排序，把最低的块 offload 回 CPU。
- d 是衰减系数，正文实验用 **d = 0.1**。
- 加载流程：先算 relevance 决定要哪些块，命中就直接用，未命中就从 CPU 拷到 GPU。

### 代码里的实际行为（与论文不完全一致，重要）

`ContextManager` 有两种 `cache_strategy`：

1. **`"lru"`（代码默认，公开 yaml 都没覆盖它）**：真正的 LRU。每步 `load_count += 1`，被选中的块记为当前 `load_count`，驱逐时间戳最小的块。**也就是说，InfLLM 公开发布的默认配置是纯 LRU，不是论文 Eq. 1。**
2. **`"lru-s"`（对应论文 Eq. 1）**：
   - 每步先把**本步被选中的块的分数置 0**（`self.cached_blocks[u][bidx] = 0`）。
   - attention 结束后，所有驻留块分数乘 `score_decay`，再给本步被选中的块加上它们的注意力质量（head 和 query 维求和）。
   - 结果：被选中块的分数**只等于本步的注意力质量**，历史被清零。没被选中的驻留块保留 d^age × (最后一次被选中时的质量)。这不是 Eq. 1 字面上的累计。
   - 驱逐时，本步选中的块被 `ignore_blocks` 保护，victim 只从未选中的驻留块里挑。
   - 推论：d=0.1 时，每晚一步分数就缩小 10 倍，所以排序**主要由 recency 决定**，注意力质量只在质量差超过 10 倍时才会翻转顺序。
- 驱逐粒度：每次查找前，根据新加载块数腾出空间（`num_remove = |cached| − max_cached_block + #misses`）。

### 对比策略

- Random（从 GPU 缓存随机挑块 offload）、FIFO（最早加载的块先出）、他们的"LRU"（即 Eq. 1，扫 d）。
- **没有 oracle（Bélády）**，没有纯 recency LRU 的单独一条线（d=0 那个点最接近"只看最后一步质量"，不是 LRU），也没有 LFU 或 ARC。

## 评价指标

| 指标 | 怎么测 |
|---|---|
| 任务准确率 | ∞-Bench、LongBench 各任务官方指标（Table 1、5、6） |
| **GPU cache miss rate** | App. A Fig. 4：GovReport 的一个样本，横轴 score decay d ∈ {0, 0.1, …, 1.0}，Random 和 FIFO 画成水平线。没说明是按块访问次数算、是否包括编码阶段、样本数，也没给表格 |
| VRAM | Table 2：InfLLM 26.3G，Llama-1M 全注意力 76.6G；"100K token 只需 26G VRAM"（§3.2） |
| 时间 | Table 2：InfLLM 26.7s，Llama-1M 40.4s（−34%）；没说明是 per-sample 还是总时长、输入多长 |
| 召回质量 | 没有单独的 recall/precision 指标，只用下游准确率 |
| FLOPs、PCIe 字节、kernel 开销 | **都没有报告** |

## Robustness / 消融

1. **Representative token 数（Fig. 2a）**：r_k ∈ {1, 2, 4, 8}，R.KV / Math.F / QA 随 r_k 增大而提升，8 时略降。
2. **选中 unit 数（Fig. 2b）**：{2, 4, 8, 16, 32, 64, 96, 128}，1→32 提升明显，之后趋平（R.KV 从约 0 升到约 95%+）。
3. **Unit 大小（Fig. 2c）**：{32, 64, 128, 256}，总上下文固定 12K。最优值因任务而异（R.KV 和 Math.F 的趋势相反）。
4. **消融（Table 4）**：

   | 设置 | R.KV | Math.F | QA |
   |---|---|---|---|
   | InfLLM | 96.8 | 25.7 | 15.7 |
   | Decoding-Only（只在解码查找） | 85.2 | 26.3 | 12.0 |
   | w/o Lookup | 0.4 | 16.3 | 11.4 |
   | Mean Repr（块 key 均值代替代表 token） | 84.6 | 25.1 | 14.9 |

5. **缓存策略（App. A，Fig. 4，GovReport 一个样本，按图读数）**：

   | d | 0 | 0.1 | 0.2–0.8 | 0.9 | 1.0 |
   |---|---|---|---|---|---|
   | "LRU"（Eq. 1）miss rate | ≈3.4% | ≈2.3% | ≈2.25–2.35% | ≈2.75% | ≈5.6% |

   Random ≈4.25%，FIFO ≈4.15%（水平线）。
   - **d=1（不衰减的累计质量，接近 H₂O/LFU）最差，比 Random 还差。**
   - d=0（只看最后一次质量）比中间值差约 1 pp。
   - 0.1–0.8 基本是平台，对 d 不敏感。
6. **长度扩展（Fig. 3）**：passkey 32K→1024K，InfLLM 保持 100%，LM-Infinite 迅速下降。
7. **多 passkey 顺序（App. B）**：64K、50 条，顺序正确率 100%。
8. **主要准确率数字**：∞-Bench 平均，Mistral 上 InfLLM 57.7 vs NTK 44.3、SelfExtend 44.8、Streaming 21.5、H₂O 18.7；Llama-3 上 45.0 vs SelfExtend 38.0。RAG 对比（Table 3）：R.KV 96.8 vs 13.2。

## 关键结论

- 训练无关的块级 context memory 能让 8K/32K 模型处理 100K–1024K 输入，∞-Bench 上明显好于 sliding window 和 H₂O（57.7 vs 21.5/18.7）。
- 把 KV 放 CPU、GPU 只留少量块的缓存是可行的。他们的 decayed attention-mass 分数 miss rate 约 2.3%，FIFO 约 4.15%，Random 约 4.25%（只有一个 GovReport 样本）。
- 衰减很关键：不衰减（d=1）的累计分数最差（约 5.6%），只看最后一次（d=0）也比中间 d 差。
- 和全注意力的 Llama-1M 相比，VRAM 只用 34%，时间少 34%。

## 对我们的启示

### 需要补的实验（按优先级）

1. **[P0] 真实注意力质量版本的 InfLLM baseline。** 我们现在的 "InfLLM-style dsum" 用 Quest 上界的 softmax 当质量代理（held-out B=96 时 G 0.270）。审稿人对比时第一个问题就会是"这不是 InfLLM 的信号"。
   - 需要在 capture 时记录**每步每层每个被选中 page 的 post-softmax 注意力质量**（GQA 下对所有 query head 求和，和 InfLLM 一样在 head 维求和），然后跑 Eq. 1 的精确版本。
   - 如果 attention kernel 能顺便给出所有驻留 page 的质量更好；InfLLM 只算被选中块，因为只有它们参与 attention。
   - 报告"代理 vs 真实质量"的 G 差。如果差很小，可以说代理足够；如果真实质量更好，就要改 headline 的定位。
2. **[P0] 同时实现论文 Eq. 1 和代码 `lru-s` 两个变体，并单列 InfLLM 的默认 `lru`。**
   - Eq. 1：s ← d·s + mass，所有驻留 page 每步衰减。
   - `lru-s`：选中时分数清零，再写入本步质量；未选中只衰减。
   - `lru`：就是我们的 LRU。我们可以写"InfLLM 发布代码的默认策略即 LRU，我们的 LRU 行就是它"。这一句很能防住审稿人。
3. **[P1] 复现他们的 Fig. 4：在我们的 trace 上画 miss rate vs d（0, 0.1, …, 0.9, 1.0）**，同时画 FIFO、Random、LRU、Bélády 的水平线，B=96 和 128 各一张。
   - 这能直接回答"你们的 EMA 和 InfLLM 的 d 扫描有什么不同"。我们的 α 和他们的 d 是同一个量（EMA 的 α ≈ d，差别只在新值的权重是 1−α 还是 1）。
   - 预期能复现"d=1 最差、中间平台"的形状。如果复现出来，就是独立验证；如果没复现出来，要解释（page 16 vs 块 128、解码-only vs 含编码）。
4. **[P1] 单位大小扫描。** InfLLM 的块是 128 token，我们是 16。至少跑 page=32、64（选择也要相应改成按块的 Quest bound 重新 capture，或者把 16-token page 的分数取 max 聚合成 32/64 的块做近似）。审稿人会问"在 InfLLM 的粒度下你们的结论还成立吗"。
5. **[P1] 包含 prefill/chunk 编码阶段的访问流。** InfLLM 在编码 chunk 时也查找和换块，他们的 miss rate 很可能大部分来自编码阶段。我们的 trace 只有 decode。至少要在论文里写明这个差别；如果能 capture chunked prefill（例如 chunk=512）阶段的 page 选择，就补一组结果。
6. **[P2] 更长上下文。** 我们的 held-out 文档是 16–32K，InfLLM 是 100K+。如果 Qwen3-0.6B 支持（原生 32K，YaRN 可到 128K），补几条 64K–128K 文档的 trace，看 G 是否随长度变化。
7. **[P2] 第二个模型。** InfLLM 用 7–8B 模型；我们只有 0.6B（dev 上有 1.7B 的检查）。至少给一个 7–8B GQA 模型（Llama-3-8B 或 Mistral-7B，正好和 InfLLM 一样）的 trace 级 miss/G 结果。

### 在 trace 模拟器里复现 InfLLM 策略

- **需要的 trace 字段**（npz v2 已有 `ranked [T,L,K]` 和 `scores [T,L,P]`）：
  - `attn_mass [T, L, K]`（float16）：本步被选中的每个 page 的 post-softmax 注意力质量，按 query head 求和（head 平均也可以，只差一个常数）。softmax 的分母要包括 sink/local 部分，这样才和 InfLLM 一致。
  - 可选 `attn_mass_local [T, L]`：local window 和 sink 得到的质量，用于检查归一化。
  - 如果要做 InfLLM 的"块质量"：在 page=16 上记录，块质量 = 组内 page 质量之和，不需要重新 capture。
- **实现**（`kvpolicy.py` 里已有 `InfLLM` 和 `DecayedSum` 类，只需换输入）：
  - `begin()`：对所有驻留 p，`f[p] = d·f[p] + mass.get(p, 0)`（Eq. 1）；`lru-s` 变体先把 `p ∈ ranked` 的 f 置 0。
  - `admit()`：未命中页 `f[p] = mass[p]`。
  - `evict()`：未选中驻留页中 f 最小者，tie 按 LRU。
  - 参数：d ∈ {0, 0.1, 0.5, 0.9, 1.0}，主结果用 d=0.1（论文值）和 LODO 选出的 d。
- **注意的不一致点**：InfLLM 的质量是在 attention **之后**才知道的，所以它更新的是下一步的驱逐依据；我们的 Quest 分数在 attention 之前就有。两者时序相同（本步的驱逐在 attention 前发生，用的是截至上一步的分数），复现时要保证 mass 在第 t 步只影响 t+1 及以后的驱逐。

### 定位上的注意

- 我们的 score EMA 本质上是"InfLLM 的衰减规则作用于所有驻留 page 的 Quest 分数"，这一点 STATUS_REPORT 已经承认，必须在论文里引 InfLLM App. A。
- InfLLM 只比了 FIFO 和 Random，没有 Bélády，也没有归一化的 gap。我们的 G 指标和"LRU–Bélády gap"是 InfLLM 没做的部分。
- InfLLM 的数字（约 2.3% miss）不能和我们直接比：128-token 块、包含编码阶段、单个 GovReport 样本、不同模型。

## 局限/没读清的地方

- **Fig. 4 的测量细节没交代**：样本数（"a sample"）、用哪个模型（推测 Mistral）、miss rate 的分母（块访问次数？）、是否包含编码阶段、cache 大小用的是 32 还是 2×96。
- §4.2 说 Mistral 每步加载 96 个 unit，App. C.1 说 cache 是 32、"twice the number of loaded units"，自相矛盾；公开 yaml 是 topk 16 / cache 32。无法确定论文主表到底用哪个配置。
- Eq. 1 下标写的是 l_X，正文却称 l_u；"LRU" 的名字和实际规则不符。代码 `lru-s` 的清零行为和 Eq. 1 字面不同，论文的 Fig. 4 用的是哪种实现不清楚。
- 没有报告 PCIe 传输量、换页的时间开销、缓存命中对延迟的影响；Table 2 的时间对比也没说明输入长度。
- Fig. 2 各点的精确数值只能从图上读，误差约 ±1 个点。
