# Rethinking KV Cache Compression for LLM Serving 阅读笔记

> 来源：MLSys 2025 版本（arXiv 2503.24000v1 PDF，首页有 "Proceedings of the 8th MLSys Conference, Santa Clara, 2025" 版权行），含附录 A–G。阅读日期 2026-11-25，已读全文。MLSys 虚拟页面（https://mlsys.org/virtual/2025/3341）只有摘要和代码链接。

## 基本信息

- **标题**：Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving
- **作者**：Wei Gao*, Xinyu Zhou*, Peng Sun, Tianwei Zhang, Yonggang Wen（NTU、NTU S-Lab、上海 AI Lab、商汤）
- **Venue**：MLSys 2025
- **代码**：https://github.com/LLMkvsys/rethink-kv-compression （这次没有读代码）
- **类型**：综述加部署层面的实证研究，属于"负面结果"或"重新审视"类论文。核心论点是：现有 KV 压缩论文的评测方式（HF transformers、固定输出长度、只报平均准确率）高估了它们在生产环境中的价值。

## 设置

| 项 | 内容 |
|---|---|
| 模型 | 吞吐：LLaMA-2-7B（正文，文中写 LLaMA-7B）、LLaMA-2-13B、LLaMA-2-70B（H800）、Mistral-7B（附录）。长度与负样本分析：LLaMA-3.1-8B-Instruct、Mistral-7B-v0.1 |
| 硬件 | 主要是 4×NVIDIA A6000（NVLink 互连），Intel Xeon Gold 6326 2.9 GHz；70B 实验在 H800 上（Fig.2） |
| 互连 | NVLink（只影响 TP 实验）。**不涉及 offload，也不涉及 PCIe** |
| 框架 | HF Transformers（TRL）4.43.1 + Torch 2.1.2；TRL + FlashAttention 2.5.6；**LMDeploy v0.6.0.1**（有 PagedAttention 和 FlashAttention，压缩算法在它上面实现）。附录 A.4 解释了为什么选 LMDeploy 而不是 vLLM：量化 kernel 更快，KIVI 难以集成进 vLLM |
| 压缩算法 | KIVI-4/2（G=32，R=128），GEAR-4/2（s=2%，r=2%），StreamingLLM（64 个初始 token + 448 个最近 token = 512），H2O（64 heavy + 448 recent = 512）；附录另有 SnapKV |
| Batch/长度 | 吞吐用合成输入，扫 batch 1–16、prompt/KV 长度 256–8192；TP=1/2/4。端到端延迟：ShareGPT 每条样本，**batch=1**。路由实验：4×A6000，ShareGPT 1000 条，Poisson 10 req/s |
| 数据 | ShareGPT（1000 条，最多生成 1024 token）；LongBench（负样本分析，使用官方指标和设置） |

## 方法与基线

这篇论文不提出新的压缩算法。它的工作分三部分：
1. **综述**（Table 1 列出 13 个量化方法和 28 个稀疏方法的特性、最重评测设置、内存/prefill/decode 加速、所用框架；Table 2 列出现有 benchmark）。从中提炼出三个"缺失环节"：(1) 吞吐多在 TRL 上测，忽略了 FlashAttention/PagedAttention 和多 GPU；(2) 用固定输出长度测效率，忽略了压缩导致输出变长；(3) 只报告平均准确率，忽略了单个样本上的失败。
2. **实证**：在三个框架上测 prefill/decode 吞吐和 attention 层时间；测输出长度分布偏移（D=(L_un−L_cs)/L_un），并用温度 0.9/1.1 作为"自然波动"对照；测 batch=1 的端到端延迟 CDF；用 Algorithm 1 收集负样本（在 baseline 表现良好的样本中，压缩后相对准确率下降超过阈值 θ 的样本）。
3. **工具**：吞吐预测器（离线 profile attention 层，接入 Vidur 运行时预测器）；长度预测器（BERT 分类器）；负样本 benchmark（θ=10%）；以及基于这两个预测器的请求路由器。

**基线**：FP16 不压缩，在同一框架下。

## 评价指标

- **吞吐**：prefill 和 decode 吞吐（tokens/s），以及相对 FP16 的加速比。附录 B 明确写了"**exclude the initialization overhead and average the throughput performance over three times**"，即 3 次取平均、剔除初始化，相当于 warmup。但没有报告方差或误差棒。
- **算子时间**：attention 层执行时间，按 prefill 和 decode 分开（Fig.3），用来定位压缩本身的额外开销。
- **端到端延迟**：ShareGPT 每条样本 batch=1 的延迟，画 CDF（Fig.5），可以看到尾部。
- **长度**：输出长度差分布（对数密度直方图 + KDE），|D|≥50% 的样本比例。
- **准确率**：LongBench 各任务指标；负样本数随阈值变化；按任务类型的负样本占比。
- **语义质量**：与 ShareGPT 中 ChatGPT 回答的语义相似度分数（Table 4，用来判断"变长"是否是"啰嗦"）。
- **工具准确率**：预测器准确率（Table 6）。
- 没有 FLOPs/MACs；显存只在 Table 1 文献汇总中出现；没有测压缩策略本身的 CPU 开销。

## Robustness / 消融（带数字）

1. **TRL 测出的加速不可靠（Obs.1，Fig.1a–d）**：batch > 4 且长度达到 1024 时，StreamingLLM 在 TRL 上的加速比在 PagedAttention/FlashAttention 基线面前不再明显。
2. **prefill 吞吐（Fig.1e–h）**：KIVI 和 StreamingLLM 接近甚至略高于 FP16；GEAR 和 H2O 一直更低，并且 prompt 越长差距越大（GEAR 要做误差修正，H2O 要多遍 attention 来得到注意力分数）。
3. **decode 吞吐（Fig.1i–l）**：batch 和 KV 都小时几乎没有差别；batch 大、KV 长时稀疏方法保持优势，量化方法的优势消失；KV 长度 8192 时量化方法 OOM。
4. **TP 削弱压缩收益（Table 3）**：prefill 相对 FP16（TP=1/2/4）：KIVI-4 1.06/1.09/1.03×，GEAR-4 0.86/0.80/0.90×，H2O 0.58/0.58/0.51×，Stream 0.95/0.96/0.92×。decode：KIVI-4 0.98/0.88/0.90×，GEAR-4 1.02/0.97/0.97×，H2O 1.34/0.69/0.85×，Stream 1.34/1.01/0.97×。FP16 decode 吞吐 129.72/194.83/195.02 T/s。解释：TP 缓解了每卡的显存带宽压力，压缩"少读数据"的好处随之变小。（Table 3 对应的 batch/长度在正文里没有写明。）
5. **输出变长（Obs.3，Table 5，LLaMA-3.1-8B，1000 条）**：输出变长 ≥50%（D≤−50%）的样本比例：KIVI 24.5%，GEAR 27.1%，H2O 21.3%，Stream 26.4%；对照组温度 0.9 为 27.5%，1.1 为 31.4%。输出变短 ≥50% 的比例：温度对照 20.8%/21.3%，压缩方法只有 6.8–16.5%。也就是说温度变化让输出变长和变短大致对称，压缩则偏向变长。压缩比越高，分布越平，变长越多（Fig.4）。Mistral-7B（附录 Table 9）趋势相同。平均长度增加 1.55–1.76×，语义分数 46.2–50.7，FP16 为 49.6（Table 4）。
6. **端到端延迟（Obs.4，Fig.5）**：batch=1 时各方法的延迟 CDF 都集中在约 20–25 s，压缩方法没有明显收益，GEAR 的尾延迟更长。
7. **负样本（Obs.5–6，Fig.6–7）**：LongBench 平均分 FP16 41.2，KIVI 41.3，GEAR 40.9，H2O 39.1，Stream 38.9。平均分差别很小，但 θ=10% 时仍有数百个负样本（Fig.6 纵轴到 600–800）。把多种算法组合起来可以减少负样本，但无法消除。负样本集中在 summarization 和 QA。在负样本集上（Table 7），QA：baseline 52.0，KIVI 28.8，GEAR 28.7，H2O 33.8，Stream 30.4；Code：baseline 97.0，KIVI/GEAR 30.0，H2O 57.2，Stream 61.3。（注意这个集合是按"压缩后失败"挑出来的，存在选择偏差，数字只能说明失败有多严重。）
8. **工具**：吞吐预测器准确率 85.8–88.5%，长度预测器 87.8–95.7%（Table 6）。路由（Table 8，平均 E2E）：FP16 11.4 s；KIVI baseline 9.1 → 两个预测器都用 6.3 s；GEAR 13.4 → 7.4；H2O 10.6 → 6.9；Stream 10.3 → 6.6。只用吞吐预测器带来 1.18–1.48× 提速，只用长度预测器 0.83–1.03×（可能更慢），两者结合 1.45–1.80×。

## 关键结论

1. 在 HF transformers 上测的加速不能代表生产效果。必须在有 FlashAttention/PagedAttention 的 serving 框架里测，而且收益只在特定的 (batch, 长度, TP) 区间才存在，有些区间是负收益。
2. 额外的计算步骤（量化误差修正、注意力分数统计）会抵消访存节省，尤其在 prefill 阶段。
3. 有损压缩会让输出变长。只看固定长度下的吞吐会高估端到端收益。
4. 平均准确率掩盖了单个样本的严重失败，失败集中在 summarization 和 QA。
5. 推荐把评测拆成三个维度：吞吐区间、长度分布、负样本。

## 对我们的启示

1. **这是审稿人心目中"系统评测"的标准清单，我们必须逐项回应**：
   - *真实 kernel 上测*：我们已经用 fused Triton kernel 在 L4 上测了。要在论文中写明，基线（LRU）和策略用的是同一套 kernel，计时包含策略记账的开销，而不是在 HF eager 下测。
   - *区间（regime）报告*：它最核心的发现是"收益依赖 (batch, 长度, TP)"。我们只测了 batch 1 和 L4 一种 PCIe 带宽。至少要扫上下文长度和 B，并说明在哪些区间延迟持平、哪些区间字节减少有可能转化为延迟收益（可以用带宽模型外推，并标注为外推）。
   - *额外开销计入*：H2O/GEAR 的教训是统计分数的额外遍历会吃掉收益。我们的 score-EMA 需要每步每页更新 EMA 并排序，必须单独测量这部分开销（每步 µs、占 decode 的百分比），并与 LRU 的开销对比。
2. **输出长度/准确率维度**：Quest 风格的 page 选择是有损的，但我们比较的各种淘汰策略都在同一个选择结果上运作。如果 GPU pool miss 时会同步 recall，那么不同淘汰策略的输出应完全相同。这一点要写明并验证（例如检查输出 token 完全一致，或 logits 差为 0）。这样"长度偏移"问题不存在，可以抢先回答审稿人。如果有任何近似（例如 miss 时跳过某页），就必须报告长度分布和准确率。
3. **按样本报告，而不是只报平均**：它的负样本分析对应到我们，就是按文档报告 gap-closed 和字节差的分布：胜/负/平文档数、最差的那篇文档、每篇的配对差。21 篇 held-out 文档完全可以画一个按文档排序的条形图或 CDF。
4. **重复与 warmup 的最低标准**：它只做了"3 次平均、剔除初始化"，也被 MLSys 接收了。我们的延迟测量至少要达到：明确的 warmup 步数、≥3–5 次重复、报告中位数和离散程度（IQR 或 std），并用 CUDA events 或同步计时，说明计时边界。
5. **"负面结果也能被主会接收"的先例**：这篇是 MLSys 主会论文，它的贡献主要是"现有方法在部署层面不如宣称的那样好"，加上可复现的工具。这说明我们"字节减少但延迟持平"加上"预注册的 rank 负结果"本身就可以成为贡献，前提是测量严谨、给出机制解释，并附上可复用的工具（例如我们的回放模拟器、Bélády 计算、预注册流程）。
6. **TP 发现的类比**：它发现增加带宽供给（TP）会削弱"少读数据"的收益。对我们来说，L4 batch=1 时 PCIe 不饱和，所以字节减少不转化为延迟收益，是同一类现象。可以引用它作为支撑：传输/访存节省只有在该资源是瓶颈的区间才有价值。
