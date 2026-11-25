# ECHO 阅读笔记

> 来源：USENIX OSDI '26 正式版 PDF（https://www.usenix.org/system/files/osdi26-liu-guangda.pdf，正文 pp.17–30，另含 Artifact Appendix pp.36–37）。阅读日期 2026-11-25，已读全文。下文数字均取自正文和图注。图中只以柱状/曲线形式给出、正文没有写出数值的，标为"只有图"。

## 基本信息

- **标题**：ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs
- **作者**：Guangda Liu, Wenhao Chen, Chengwei Li, Zhenyu Ning, Jing Lin, Yiwu Yao, Quan Chen, Shixuan Sun, Jieru Zhao（通讯作者）, Minyi Guo（上海交通大学；华为；贵州大学）
- **Venue**：OSDI 2026（第 20 届，2026-07-13~15，Seattle）
- **代码**：https://github.com/sjtu-zhao-lab/ECHO；Artifact 在 Zenodo（records/20376436），提供 Docker 镜像和脚本，可复现 Fig.7–9、12–15、18–19
- **一句话**：面向 DeepSeek-V3.2（DSA 原生稀疏注意力）的 serving 系统。MLA KV 全量备份在 host，GPU 上按层维护一个 token 级 cache pool。cache 管理全部做成能被 CUDA Graph 捕获的 GPU 并行操作。另外用 EMA 预测第 k 名 index score 作为阈值，在 indexer kernel 里提前 prefetch（无损）。

## 设置

| 项 | 内容 |
|---|---|
| 模型 | DeepSeek-V3.2-Exp 4-bit AWQ 量化版（indexer 以及第 0/1/2/60 层不量化）。原始 FP8 版放不进 8×H20。动机实验（Fig.1）另外用了 8×H200 上的 FP8 版；Fig.3(a) 用了 DeepSeek-V3.1（dense attention） |
| 硬件 | 单节点 8×NVIDIA H20（96 GB），224 核 Xeon Platinum 8480+，1.5 TB DRAM |
| 互连 | GPU–host 走 **PCIe Gen5，64 GB/s**。讨论部分提到 NVLink-C2C 900 GB/s（Grace）/1.8 TB/s（Vera）时 recall 开销会更小，但没有做实验 |
| 并行 | ECHO 和 SGLang：attention DP8 + MoE TP8；vLLM：TP8（不支持 DP+TP；EP 会 OOM） |
| 软件 | 基于 SGLang v0.5.4 实现；对比 vLLM v0.11.1；CUDA 12.9，PyTorch 2.8；cache 管理用 Triton，prefetch 融合进 DeepGEMM 的 indexer kernel |
| 池子大小 | host pool 1.8M token（约 1000 GB 内存）；GPU pool 在受限实验里设为 200K 和 110K token。元数据 4N_H+13N_G 字节，N_H=2M、N_G=200K 时每层约 10 MB，全模型 610 MB |
| DSA 参数 | top-k=2048（token 级，不是 page 级）；indexer H=64、d=128，FP8 |
| 上下文/负载 | 长上下文：从 InfiniteBench 采 318 条 80K–100K token 的请求（共约 26M 输入 token）。短上下文：ShareGPT 采 100 条请求。prefill chunk = 2048 |
| Batch | serving 场景，按请求速率扫描（0.1–0.9 和 Inf）。论文报告"有效 batch size"：SGLang 约 30（全部 DP worker 合计），vLLM 约 1.5，ECHO 更大（只在图上标注） |
| 部署 | 吞吐实验：PD 分离，8×H20 作为 decode 实例，KV 预先算好再加载；延迟实验：PD 混合 |

## 方法与基线

**Cache 结构**：indexer 的 K cache 常驻 GPU（因为每步都要对全部 L 个 token 打分）；MLA latent KV 在 host 和 GPU 各存一份，GPU pool 只作为"已选 token 的 cache"。**GPU pool 按层管理**（每层选的 token 不同），host pool 按模型管理。

**淘汰策略（与我们最相关）**：每个 GPU slot 有一个整型 `GPUTokenPriority`。allocate 时给新 token 加 priority；free 时先给本步 indexer 选中的 token（protected）加 priority，再用并行 argtopk 取 priority 最低的 slot 淘汰。论文称之为 **"LRU-like eviction policy"**。淘汰只改元数据，不需要写回 host，因为所有 KV 生成时已经备份到 host。正文还写了"token eviction priority is then adjusted based on the index scores"，但没有给出具体公式，也没有和其它淘汰策略（FIFO、score-based、Bélády）比较。**淘汰策略不是这篇论文的研究对象**。

**EMA 的用途和我们不同**：ECHO 的 EMA 预测的是**下一步第 k(=2048) 名 index score**，$\hat s_{t+1}=\alpha\hat s_t+(1-\alpha)s_t$，α=0.5。这个预测值作为 top-p 式阈值：indexer kernel 每算出一个 token 的分数，超过阈值就立刻 prefetch。prefetch 之后仍然会对真实 top-k 做一次 "guaranteed recall"，所以结果是**无损**的。它不对 token/page 的分数做 EMA 来决定淘汰顺序。prefill 阶段则用 EMA 预测值平移分数，再做 256-bin radix 近似 top-k，在 Q block 之间做 inter-query prefetch。

**Graph-friendly 管理**：所有元数据都是定长 GPU 张量（free bitmap、priority、GPU↔host 双向映射、indices buffer）。allocate 用 atomicAdd 并行分配，free 用 argtopk 加 scatter，recall 通过 UVM 让 GPU kernel 直接读 host 内存。整个 decode 路径可以被一张完整的 CUDA Graph 捕获。

**Fused prefetch kernel**：在 DeepGEMM indexer kernel 里做 warp specialization，形成 TMA→GEMM→prefetch 三级流水。prefetch 数量用全局计数器设上限，上限与上下文长度成正比。

**基线**：SGLang v0.5.4（不 offload）、vLLM v0.11.1（TP8）。Table 1 定性比较了 Strata、ArkVale、InfiniGen、FreeKV、CLO、SparseServe，但**没有在实验中跑这些 offloading 系统**（理由是它们不是 serving 系统，或者不支持 graph、多 GPU、MoE）。

## 评价指标

- **吞吐**：generation throughput（output tokens/s），在不同请求速率下测；输出长度固定为 256（"to reduce variance from long-tail generations"），另外有不限长度的版本。
- **延迟**：ShareGPT 100 条请求上的平均 TTFT、ITL、端到端延迟（只报均值，**没有 P50/P99**）。
- **延迟拆分**（Fig.16，ShareGPT、Inf 速率）：每层 prefill 延迟；全层 decode 延迟拆成 graph replay、MoE、其它 kernel；offloading 专属开销拆成 allocate/free/recall/offload 四项。
- **命中率**：每层 GPU pool 命中率，按 decode 步平均（Fig.17）。
- **微基准**：indexer+recall 延迟在不同上下文长度、batch、命中率（0.5/0.9/0.97）下的变化（Fig.4、18、20）；分配器 alloc+free 延迟（Fig.3b）。
- **字节/带宽**：没有直接报 PCIe 字节数；只给了"每步平均 recall 37 token、每个 DP worker 平均 offload 13 token"。
- **FLOPs**：只在动机部分报了 FlashMLA sparse 与 dense kernel 的 TFLOPS 利用率（sparse 在 64K、8 请求时利用率 53%）。
- **显存**：Fig.1 给出 HBM 占用分解（参数占 60% 以上；每个 DP worker 可放 511K（AWQ）/655K（FP8）token，SGLang 实际只有 380K）。
- **准确率**：**没有准确率实验**。论文以"lossless prefetch + guaranteed recall"为理由，认为不需要。
- **重复次数/方差/warmup**：正文**没有说明**重复次数、warmup、误差棒或置信区间。所有图都是单值。Artifact 脚本每个速率各跑一次 baseline 和 ECHO，每个速率写一份 log。

## Robustness / 消融（带数字）

1. **最大吞吐（Fig.12a，全部 HBM，输出 256）**：低速率时三者相当；Inf 速率下 ECHO 是 SGLang 的 **2.15×**、vLLM 的 **4.1×**。SGLang GPU KV pool 为 325K token，每 GPU 只能放 3–4 个 InfiniteBench 请求；vLLM TP 部署因为 MLA KV 重复，8 卡合计只有 180K token。
2. **不限输出长度（Fig.12b）**：高速率下长尾生成造成 DP 负载不均，ECHO 的 ITL 更高，优势缩小（只有图）。
3. **受限 HBM（Fig.13）**：GPU pool 设为 200K 时 ECHO/SGLang 为 **3.10×**；110K 时为 **4.12×**。池子更小时 ECHO 本身还略有提升，因为管理开销更低。
4. **分任务（Fig.14，不限长度）**：Code.Debug +27.07%，En.MC +2.83%，En.QA +7.11%，Code.Run −1.74%。请求数分别只有 75/160/31/42，并发不足，加上长尾输出，增益不稳定。**这是论文自己承认的负面或中性结果**。
5. **延迟（Fig.15，ShareGPT，PD 混合）**：TTFT 最多 +7.9%（prefill 时要把 KV 备份到 host）。ITL +2.7% 到 **+27.8%**，低负载时最高，因为固定的管理和 recall 成本摊不开；速率 ≥0.3 时降到 2.7–7.7%。端到端延迟在 0.1–0.2 速率时 **+15.9–19.2%**，≥0.3 时最多 +7.2%，0.5 到 Inf 时低于 4.6%。论文结论是 ECHO "most beneficial for throughput-oriented long-context serving"。
6. **延迟拆分（Fig.16）**：prefill 每层 6.82→7.42 ms（+8.8%，主要在 attention 部分）。decode 全层延迟 +2.8%，MoE 约占 70%。graph replay 多出 5.9 ms（CPU 侧 launch 和元数据）。offloading 专属开销：allocate 0.167 + free 0.572 + recall 0.145 + offload 0.261 = **1.15 ms，占全层 decode 延迟的 0.28%**。另有 6.8 ms 来自 DP worker 之间的 ReduceScatter 同步抖动。
7. **命中率（Fig.17）**：大多数层 0.97–0.99；第 12 层 0.95，第 17 层 0.88。论文用"垂直线"注意力模式和相邻 query 选择相似来解释。
8. **intra-query prefetch（Fig.18–19）**：微基准中命中率 0.5 时总延迟最多降 **1.29×**，0.9 时最多 **1.51×**（recall 几乎完全被隐藏），0.97 时提升很小。**端到端吞吐只提升最多 4%**，因为真实命中率很高，需要 recall 的 token 很少，而且 MoE 占主导。
9. **inter-query prefetch（Fig.20）**：最多 1.1×；**没有做端到端消融**（PD 分离实验用的是预先算好的 KV）。
10. **分配器（Fig.3b）**：300K pool 上一次 alloc+free，SGLang 约 12 µs，ECHO 6–8 µs（1.6–1.9×）。Graph 本身的价值（Fig.3a）：V3.1 1.2×，V3.2 1.5×。
11. **EMA 预测质量（Fig.7–8）**：在一条 20K token 的 ShareGPT 请求上，第 3 层和第 50 层的候选 token 数大多接近 2048，偶尔超出，超出部分由上限截断。**只用一条请求、两层来展示**，没有统计汇总。

## 关键结论

1. 对原生稀疏注意力模型，瓶颈从计算转到了 HBM 容量，限制了并发。offload 到 host 可以换来更大的有效 batch，在长上下文、高负载下吞吐最多 2.1×（受限 HBM 时 4.12×）。
2. 动态 offload 要在生产 serving 中可用，cache 管理必须能放进 CUDA Graph。graph 化之后，管理开销只占 decode 延迟的 0.28%。
3. GPU 每层 token pool 加上 LRU-like 淘汰，在 DSA 上命中率已达 0.97–0.99。因此 recall 字节很少，prefetch 的端到端收益（≤4%）远小于微基准（1.51×）。
4. 低负载下延迟有代价（ITL 最多 +27.8%，E2E 最多 +19.2%）。offload 是"吞吐优先"的技术。

## 对我们的启示

1. **我们的"字节减少但延迟持平"与 ECHO 的观察一致，可以直接引用来支撑**：ECHO 在命中率 0.97+ 的工作点上，prefetch 微基准 1.51× 只换来端到端 ≤4%。原因是剩下的 miss 很少，而且其它算子（MoE）占主导。我们在 L4 上 batch=1、PCIe 字节 −9–19%、延迟与 LRU 打平，是同一机制。写作时应主动用"recall 只占 decode 关键路径的 x%"来解释，而不是回避。
2. **审稿人期待的延迟拆分粒度**：ECHO 把 offload 专属开销拆到 allocate/free/recall/offload 四项（ms 和占比），并且把 graph replay、MoE、通信分开。我们至少要给出：selection（Quest 打分 + top-K）、策略记账（rank/score-EMA 的更新与排序）、H2D 拷贝、attention、其它，并说明哪些在关键路径上、哪些和计算重叠。score-EMA 比 LRU 多出的元数据更新成本必须单独列出（ECHO 的元数据是 4N_H+13N_G 字节/层，我们也应给出每层字节数）。
3. **"在哪个区间 transfer 才重要"要画出来**：ECHO 的 Fig.4 用"indexer 延迟 vs recall 延迟随命中率、上下文长度变化"界定了 prefetch 有效的区间（约 100K 上下文且命中率 90% 时两者交叉）。我们可以仿照，用 PCIe 带宽、B（页预算）、上下文长度、batch 做一个 regime 图：miss 字节 × 1/带宽 与 per-step 计算时间的比值。说明 L4 batch=1 处在"transfer 不是瓶颈"的区间，以及在什么带宽/批量下 −9–19% 字节会变成延迟收益。可以用 roofline 或微基准外推，并明确标注为外推。
4. **ECHO 的弱点就是我们能做得更严谨的地方**：没有重复次数和误差棒；EMA 预测质量只展示 1 条请求 × 2 层；淘汰策略没有和任何替代方案或 Bélády 比较；没有字节数。我们有预注册、21 篇 held-out 文档、Bélády 上界、gap-closed 指标，应在 workshop 稿中明确报告：重复次数、warmup 步数、CUDA event 计时方式、按文档的配对差和置信区间（bootstrap 或配对 t），以及每层命中率分布（ECHO Fig.17 那样的按层图，我们有 per-layer budget allocation，正好可以画）。
5. **定位差异要写清楚**：ECHO 的 EMA 是阈值预测，用于 prefetch；我们的 score-EMA 是淘汰排序。ECHO 的淘汰是"selected 就提升 priority"的 LRU-like。这正好说明，即使在 OSDI 级系统里，淘汰仍默认用 LRU-like，没有人系统比较过它离 Bélády 有多远。这是我们 characterization 的切入点。另外 ECHO 是 token 级（k=2048），我们是 16-token page（K=64），粒度不同，引用时要注明。
6. **负载/批量维度**：ECHO 的收益完全来自更大的有效 batch。我们只测了 batch 1，审稿人会问多请求并发下 per-layer pool 的竞争和命中率。至少要在 limitation 中说明，最好补一个 batch>1 的模拟（命中率和字节可以离线回放，不必真跑端到端）。
