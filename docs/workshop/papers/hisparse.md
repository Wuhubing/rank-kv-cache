# HiSparse 阅读笔记

来源：arXiv HTML 全文 v1（https://arxiv.org/html/2608.07009v1），包括正文、Appendix A 和图注。HTML 版的图是矢量图，没有可读的 PNG，所以下面的数字只取自正文和图注。

## 基本信息

- **标题**：HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management
- **作者**：Zhiqiang Xie（通讯作者，Stanford & Meta）、Zhangheng Huang（Alibaba）、Tingwei Huang（Ant Group）、Ziyi Xu（SJTU）、Ruiyang Ma（PKU）、Christos Kozyrakis（Stanford & NVIDIA Research）
- **状态**：arXiv preprint，cs.DC，2026-08-07（v1），CC BY 4.0。文中没有写投稿 venue，体例像系统会议投稿。
- **代码**：已经合入上游 SGLang（v0.5.11 之后），用 `--enable-hisparse` 开启，配置项 `--hisparse-config` 里有 `top_k`、`device_buffer_size`(B)、`host_to_device_ratio` 和 transfer block size。
  - 文档：https://docs.sglang.io/docs/advanced_features/hisparse_guide
  - 博客：https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/
  - 主要代码文件：`managers/hisparse_coordinator.py`、`jit_kernel/hisparse.py` + `csrc/hisparse.cuh`、`mem_cache/allocator/hisparse.py`、`hisparse_memory_pool.py`、`pool_host/hisparse.py`
  - 另有 HIP 版本可以跑 AMD GPU。

## 设置

| 项 | 内容 |
|---|---|
| 模型 | DeepSeek-V4-Flash（NSA 风格 Compressed Sparse Attention）；GLM-5.1-FP8（DSA，78 个 sparse layer）；Qwen3-30B-A3B-Thinking-2507 + Quest（training-free，每层都做选择）；GLM-5.2-FP8（DSA + IndexShare，21 个 anchor 层 + 57 个 shared 层，只用于 prefetch 实验）；DeepSeek-V4-Pro（token-level，k=1024，只出现在 Fig.7 kernel 微基准）。全部用 BF16 KV，FP8 只指权重。 |
| Selector | DSA lightning indexer（token-level）、NSA/CSA（block/compressed entry）、Quest（page min/max 上界）。HiSparse 本身与 indexer 无关，只消费 selector 输出的 logical positions。 |
| 选择/缓存单元 | **管理单元是 KV record（token，或者 V4-Flash 的 4-token compressed entry）**。Quest 虽然按 page 选，但 HiSparse 的 cache slot 按 token 记账：Quest 输出的仍是 token positions，k=2048 token。文中没有写 Qwen3+Quest 的 page size。 |
| K | 所有实验都选 k=2048 token（V4-Flash 是 top-512 × 4-token entry，覆盖 2048 token）；V4-Pro 的 k=1024。 |
| Cache budget | B = 每个 request、每层的 slot 数，要求 B ≥ k。trace 实验用 B=2048/4096/8192（1×/2×/4× k）；kernel 实验扫 B/k；端到端实验用 B=4096（2k）。论文推荐 B ∈ [2k, 4k]，默认 2k。 |
| 上下文 | 端到端实验：32K 输入 / 8K 输出（并发扫描）；峰值吞吐扫输入长度 4K→200K（Qwen+Quest）和 4K→160K（GLM-5.1）。trace：一个 100,384-token 的 LongBenchV2 prompt，decode 1,799 步，统计前 1,000 步。 |
| 数据/任务 | 端到端用 SGLang `bench_serving`（合成定长、闭环并发），不测准确率。policy trace 只有 LongBenchV2 的**一条请求**。 |
| Batch | 闭环并发 8–256；kernel 实验 batch 16（扫 B/k），另外在 B=2k 下扫 batch。 |
| 硬件 | 8×H200（PCIe Gen5 x16 host 链路，约 64 GB/s 单向；2 TB host DRAM，最大点约 1 TB pinned）；2×B200；GH200（NVLink-C2C）。 |

## 方法与基线

- **系统**：完整 KV 放在 pinned host DRAM，每个 (request, layer) 在 GPU 上有一个 B-slot 的 cache。一个 fused CUDA kernel `Resolve`（每层 launch 一次，被 capture 进 decode CUDA graph）依次完成：shared-memory hash 建立 selected set → probe 所有 B 个 slot → 并行 scan 选 victim、更新 LRU → GPU-assisted IO（`ld.global.nc.v2.b64` 直接读 pinned host，做法来自 Strata）→ 输出 physical slots。新 token 的 KV 写 GPU 并 write-through 到 host。
- **替换策略**：LRU 的一个改良版。同一步内，hit 的条目排在新 fetch 的 miss 之前（MRU 端），所以被多步重复选中的条目比只出现一次的条目更不容易被逐出。每层独立管理，不做跨层 budget 分配，并且明确说跨层协调"adds little"。
- **trace 中比较的策略**：Swap-vanilla（B=k，只保留当前 top-k）、FIFO、Random、LRU（B=4096 和 8192）、Bélády（离线 oracle，B=4096 和 8192）。
- **系统级基线**：未修改的 SGLang v0.5.11（full KV 全放 HBM）。**不和任何 offloading 系统比较**，ESS 和 ECHO 被视为并发工作。
- **Oracle**：(1) Bélády（miss 率下界）；(2) no-IO oracle（跳过 host IO，输出无效，只用作 TPOT/吞吐上界）。
- **Learned predictor**：没有。试过 speculative cross-layer prefetch（用第 ℓ 层的选择去预取第 ℓ+1 层），结果是负面的。exact prefetch 只用在 IndexShare 模型上。

## 评价指标

- **Miss rate**：per-step top-k miss rate，在所有层上取平均并做了平滑，即 miss 的 selected entry 数 ÷ k。这是 trace replay 的结果，定义上是 demand miss（selected 但不在 cache）。
- **Hit rate**：87% = 1 − 13.4%（B=2k，LRU）。
- **吞吐**：generation throughput（tok/s），分 prefill+decode 和 decode-only（作为 PD 分离的代理），以峰值吞吐和随并发的曲线报告。
- **延迟**：mean TTFT、mean TPOT，都按 achieved throughput 画曲线。
- **Kernel**：Resolve 每次调用的耗时，拆成 IO / probe & scan / residual（residual 1–4 µs）。
- **内存**：每个请求的 decode KV HBM，例如 GLM-5.1 128K：13.09 GB → 0.4 GB（B=4096）。
- **没有报告**：bytes per token（只有一个估算：不缓存的话每 token 约 200 MB）、PCIe 实测带宽利用率、准确率（声称 exact，因此不测）、FLOPs、policy overhead 和 policy 本身的耗时（只有 Resolve 总耗时）。

## Robustness / 消融

1. **替换策略（Fig.6）**：GLM-5.1，k=2048，单条 LongBenchV2 trace（100,384 token，78 层，前 1,000 步）。

   | 配置 | mean miss |
   |---|---|
   | Swap-vanilla B=2048 | 30% |
   | FIFO B=4096 | 17.2% |
   | Random B=4096 | 16.1% |
   | **LRU B=4096** | **13.4%** |
   | Bélády B=4096 | 8.2% |
   | LRU B=8192 | 6.7% |
   | Bélády B=8192 | 图中有，数字未给 |

   - 按我们的 G 定义，以 Random/FIFO 为参考：LRU 比 Random 低 2.7pp，但 LRU 离 Bélády 还差 5.2pp，也就是 LRU 相对 Bélády 多出约 63% 的 miss。
   - **这里的 B 翻倍（4096→8192）对 miss 的降幅（−6.7pp）超过 LRU→Bélády 的差距（−5.2pp）。** 这条"容量 > 策略"的证据与 FlashOffloading 的结论一致。
   - 作者自己说 Bélády 和 LRU 之间的差距"bounding what a smarter, predictive replacement policy could still recover"，即承认这部分空间存在，但没有去做。
2. **Cache 大小 × 模型 × 平台（Fig.7 上排）**：batch 16，扫 B/k，四个模型（V4-Flash、GLM-5.1、Qwen3+Quest、V4-Pro），H200(PCIe) 对 GH200(C2C)。B 增大时 IO 下降，probe&scan 上升。probe&scan 和平台基本无关（两个平台相差约 5%）。结论是 2k–4k 最优。正文里没有逐点数值。
3. **Batch 扫描（Fig.7 下排）**：B=2k 时，IO 随 batch 快速增长，probe&scan 基本稳定。
4. **带宽敏感性（§4.5）**：GLM-5.1，B=2k，batch 16，IO 从 112 µs（H200 PCIe）降到 29 µs（GH200 C2C）。快链路让最优点向 B=2k 移动。作者称最优 B 主要由链路带宽决定，而不是由 workload 决定。
5. **上下文长度（Fig.5）**：
   - Qwen3+Quest/GH200：4K 时 2430→2668；32K 时 511→1824（3.6×）；200K 时 111→520（4.7×）。
   - GLM-5.1/H200：4K 时 2288→2280；32K 时 624→1919（3.1×）；160K 时 232→680（2.9×）。
6. **端到端（Fig.4，V4-Flash/2×B200，32K/8K）**：
   - 并发 64：600→1257 tok/s（2.1×），decode-only 1511→4308（2.9×）。
   - TTFT@64：829 s → 171 s。
   - TPOT@16：15.9 对 16.0 ms。
7. **Prefetch（Fig.8，GLM-5.2/8×H200，B=4096，并发 8–256）**：
   - 峰值吞吐：baseline 618，sync 1515，exact prefetch 1727（2.8×），no-IO oracle 2034。prefetch 达到 oracle 的 85%，不开 prefetch 是 74%。
   - decode-only：oracle 4671，prefetch 3410（73%）。
   - TPOT：prefetch 降低 13–15%，吞吐提高 14–17%。
   - 暴露的 IO：并发 8 时 sync 7.7 ms/token，prefetch 3.0 ms/token；并发 256 时分别是 22.0 和 11.2 ms/token。
   - oracle 和 baseline 在并发 8 时 TPOT 分别为 24.1 和 24.8 ms，说明 resolve 机制本身几乎零成本。
   - 21/78 个 anchor 层的 IO（约 27%）无法被 prefetch 隐藏。
   - prefill write-through 不增加 TTFT（10.7 对 10.7–10.8 s）。
8. **Speculative cross-layer prefetch**：hit rate 只是略有提升，端到端没有可测收益。
9. **没有做**：多条 trace / 多个数据集的 policy replay、seeds/方差、不同 k 下的 policy 比较、按层的 miss 分布、learned/score-based 策略、按层分配 budget、准确率。

## 关键结论（带数字）

- 容量墙：GLM-5.1 128K 请求需要 13.09 GB KV；32K 时 8×H200 上的 full-KV baseline 在约 60 并发饱和（约 240 GB KV）。
- B=2k 的 LRU 把 87% 的 selection 变成 GPU hit；只保留当前 top-k 会 miss 30%。
- LRU（13.4%）优于 FIFO（17.2%）和 Random（16.1%），并且"tracks the trend of Bélády"（8.2%）。
- 峰值吞吐最多提升 4.7×（Qwen3+Quest，200K）。收益全部来自更大的 batch，单步 decode 并没有变快。
- TPOT 的代价全部是 host IO：同步时约 7–8 ms/token，exact prefetch 后约 3 ms。
- 推荐 B=2k–4k，快链路时偏向 2k。

## 对我们的启示

**定位**：HiSparse 是我们最直接的 baseline。它是 Quest/DSA 之上的 host offload + GPU LRU cache，工业级实现，已合入 SGLang，并且明确把 Bélády 与 LRU 的差距列为"可回收的空间"。我们的工作正是在回收这部分差距。审稿人会自然地问："你们的 score-EMA 在 HiSparse 的设置下（token 级、k=2048、B=2k、大模型、大 batch）还成立吗？"

按优先级排列的补充实验：

1. **在 HiSparse 的度量口径下重报我们的结果（必须做）。** 他们报 per-step miss rate（miss/k，按层平均）、B/k 比例，以及 Swap-vanilla/FIFO/Random/LRU/Bélády 五条线。我们应该：
   - 给出同样五条线 + score-EMA 的 miss rate 表，并把 B 换算成 B/K（96/64=1.5×，128/64=2×）；
   - 在 page 粒度上补跑 B=K（Swap-vanilla）和 B=4K。

   原因：G=28% 的说服力取决于绝对 miss rate 和 HiSparse 的数字能否放在一起比。另外他们用的 LRU 是"hit 排在新 miss 之前"的改良版，我们的 LRU baseline 必须实现这个细节，或者证明它无关紧要。否则审稿人会认为我们的 LRU 是 strawman。
2. **容量与策略的对照（必须做）。** HiSparse 自己的数据显示 B 翻倍（−6.7pp）比 LRU→Bélády（−5.2pp）更有用。我们要报告"score-EMA 在 B 下的 miss 率 ≈ LRU 在多大 B 下的 miss 率"，即等效容量增益（例如"相当于 LRU 多 X% 的 budget"），并讨论这部分省下的 HBM 换成 batch 时值多少。这同时回应 FlashOffloading 的"capacity dominates"。
3. **Token 级 selector 的复现（强烈建议）。** HiSparse 的 policy trace 是 DSA token 级、k=2048。我们是 Quest page 级，16-token page，K=64，相当于 1024 token。至少要：
   - 在 Qwen3-0.6B 上做 page size 消融（例如 page=1/4/16，或 token 级的 Quest 变体）；
   - 或者用开源 DSA 模型或公开 trace 跑 replay。

   目的是说明 score-EMA 的优势不是 page 粒度造成的。
4. **K 和上下文长度扫描。** HiSparse 的 k=2048、上下文 100K；我们 K=64 page，上下文明显更短。需要扫 K（32/64/128）和上下文长度（尤其要推到 64K–128K token），并报告 G 随长度怎么变。长上下文下 drift 的模式可能不同。
5. **Batch>1 和带宽敏感性。** HiSparse 的核心论点是：batch 大时 IO 占主导（Fig.7 下排），所以 miss 越少越值钱。我们 batch=1 时延迟和 LRU 打平，正好落在"IO 不是瓶颈"的区间。应补 batch 4/8/16 的 TPOT 和 PCIe bytes，至少用模型或 microbench 说明在多大 batch 或多低带宽下，−9–19% 的 bytes 能变成延迟收益。不然审稿人会说"bytes 降了但没用"。
6. **Policy overhead 分解。** HiSparse 用 no-IO oracle 证明 resolve 机制零成本，FlashOffloading 则批评 HiSparse 的 O(|C|) scan。我们需要分别给出 score-EMA 更新、排序/选 victim 的 kernel 耗时，并做 no-IO 对照，说明 score-EMA 比 LRU 多出的开销。
7. **多 trace + 置信区间。** HiSparse 只有一条 trace，这是它的弱点。我们已经有 21 篇 held-out 文档，应该把逐文档的 G 分布或 bootstrap CI 作为相对他们的方法学优势突出出来。
8. **Per-layer budget。** HiSparse 声称跨层协调"adds little"，依据的是 §4.6 的 speculative prefetch 结论，而不是 budget 分配实验。我们的 per-layer allocation 把 G 从 28% 提到 40%，可以直接反驳这个说法，但必须控制总 budget 相同（uniform 对 allocated 的 HBM 总量一致）。

**需要 align 的结论**：
- recency 是很好的 online proxy；
- Bélády 差距客观存在；
- B∈[2k, 4k] 是合理区间（我们的 1.5×/2× 在区间内或略低）。

**需要 rebut 或细化的结论**：
- "LRU tracks Bélády"。实际上差距有 5.2pp，相当于多 63% 的 miss。
- "跨层协调 adds little"。我们有 per-layer budget 的反例。

## 局限 / 没读清的地方

- HTML 里的图是矢量或嵌入格式，没有拿到 Fig.6 各条曲线随步数的变化、Fig.7 各点的 µs 数值，以及 Bélády 在 B=8192 时的数字（文中只说"lower still"）。
- 文中没有写 Qwen3+Quest 的 page size，也没有写 Quest 选出的 page 是否按 token 计入 B。
- policy trace 只有 GLM-5.1 的一条 LongBenchV2 请求，没有给 Quest trace 的 policy 比较，也没有方差。
- miss rate 是否把新 token 的 compulsory miss 计入：新 token 直接写进 GPU cache，所以大概率不算 miss，但文中没有明说。
- 没有准确率评估（声称 exact）。也没有和 ESS、ECHO、InfiniGen、ShadowKV 等系统做定量比较。
