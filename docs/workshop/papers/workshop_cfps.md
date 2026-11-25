# 候选 Workshop CFP 调研

> 调研日期 2026-11-25。来源是各 workshop 官网、会议官网的 workshop 列表和 CFP 页面（链接见各行）。OpenReview API 这次返回 "Challenge verification required"，**无法直接拉取录用论文列表**。录用论文只能通过会议虚拟页面的标题列表和少量 arXiv 摘要来了解，所以下面"证据水平"一栏多数是根据标题和摘要做的判断，已经标明。凡是标"未公布"的，意思是在官网和搜索中都没有找到，不代表一定不办。

## 总表

| Workshop | 最近一届 / URL | 与我们的契合度 | 格式 / 页数 | 是否欢迎负面结果或 characterization | 截稿信息 | 录用论文的证据水平（抽样） |
|---|---|---|---|---|---|---|
| **ICBINB**（I Can't Believe It's Not Better） | ICLR 2026："Where LLMs Need to Improve"，https://sites.google.com/view/icbinb-2026/home （提交页 /submit，审稿指南 /reviewer-guidelines）。NeurIPS 2026 一届是 **ICBINB-BIO**（只收 AI for biology），https://icbinb-bio.github.io/ | **高**（如果 ICLR 2027 继续办 LLM 主题）。ICLR 2026 版把 "Efficiency and scaling: training, inference … under real-world compute constraints" 列为明确主题。我们的预注册 rank 负结果加上"字节减少但延迟持平"正好对口。NeurIPS 2026 BIO 版**不对口** | ICLR 2026：full paper ≤4 页，tiny paper ≤2 页（不含参考文献；附录不限，但审稿人不必看）；ICLR 模板；双盲，OpenReview；要求 LLM 使用声明；archival 状态写"待定"，可选 non-archival。BIO 版：full ≤8 页，tiny ≤4 页，non-archival | **最友好**，是这个 workshop 的宗旨。full paper 必须包含四个要素：问题（含假设与指标）、来自文献的方法、**有定量证据的负面或零结果**、失败原因分析。审稿标准：rigor & transparency、novelty/significance of insights、clarity、alignment；"只报 SOTA 提升而没有分析失败的论文"在 alignment 一项给低分。设有 Entropic Award（最意外的负面结果）和 Didactic Award | ICLR 2026：截稿 2026-01-31，通知 03-01，会议 04-27（已结束）。BIO：截稿 2026-09-02，会议 2026-12-11（已截稿）。**2027 未公布**：ICLR 2027 workshop 录用名单要到 2026-12-07 才公布 | ICLR 2026 共 56 篇。与效率相关的有："More Than a Quick Glance: Overcoming the Greedy Bias in KV-Cache Compression"（arXiv 2602.02199，提出 LASER-KV；Babilong 长上下文准确率，称已有方法退化 15–30%，128K 下高出约 10%）、"EoRA"（压缩补偿）、"Challenges in Inference-Time Scaling with Uncertainty-Aware Tree Search"。这说明收的不一定是纯负面结果，"指出已有方法的失败模式 + 修正"也能进。证据水平以单 GPU、开源模型、准确率为主，**没有看到系统级延迟测量的要求** |
| **ES-FoMo**（Efficient Systems for Foundation Models，ICML） | 最近确认的是 ES-FoMo III @ ICML 2025，https://es-fomo.com/ ，CFP https://es-fomo.com/call/ | **很高（主题）**：训练与推理系统、受限资源（带宽/计算/内存）、benchmarks，并明确欢迎"engineering-focused work with open codebases, implementation details and performance benchmarks" | 建议正文 ≤4 页（明确说不是硬上限），参考文献和附录不限；ICML 模板；双盲（代码也要匿名）；non-archival，允许一稿多投（包括已被 ICML 接收或在投 NeurIPS 的工作） | 没有专门提负面结果。欢迎 benchmark 和 characterization。审稿说明是 "focus on correctness and relevance" | 2025：截稿 05-27，通知 06-09。**ICML 2026 的 44 个 workshop 列表里没有 ES-FoMo**（https://icml.cc/virtual/2026/events/workshop），es-fomo.com 也没有 2026/2027 信息。**2027 未公布**。ICML 2027 workshop 征集也还没发布（2026 年的 workshop 申请截止是 02-13，按惯例 2027 年 workshop 论文截稿大约在 5 月，**这是推测**） | 2025 oral：FPTQuant、Cartridges、zip2zip、Hardware-Efficient Attention for Fast Decoding 等，多数是之后投主会的完整工作，带完整 kernel/吞吐实验。抽样的一篇 KV 相关论文 Batch-Max（arXiv 2412.05693，ES-FoMo III）：报告吞吐和准确率保持，摘要里看不到重复次数或方差。**门槛偏"完整工作的短版"** |
| **ENLSP**（Efficient Natural Language and Speech Processing，NeurIPS） | 最近确认的是 ENLSP-IV @ NeurIPS 2024，https://neurips2024-enlsp.github.io | 主题**高**（inference 方向明确列出 "Improved KV-caching solutions"） | long ≤8 页，short ≤4 页（不含参考文献和补充材料），NeurIPS 模板；双盲，每篇 ≥3 个审稿人；non-archival（可选 PMLR 特刊） | 没有专门提负面结果；evaluation/benchmarking 在主题之内 | 2024：截稿 2024-09-15，会议 12-14。**NeurIPS 2025 没有找到 ENLSP**；**NeurIPS 2026 的 102 个 workshop 列表里没有**（https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/）。可以认为已经停办或中断，**2027 未公布** | 2024 spotlight：GEAR（KV 压缩）、RetrievalAttention、MixAttention、Evolved Universal Transformer Memory（学习式 KV 淘汰）。这些都是后来发表在主会或 arXiv 上的完整工作 |
| **ML for Systems**（NeurIPS） | NeurIPS 2026（第 10 届，Sydney，12-11/12），https://mlforsystems.org/call_for_papers.html | **中**。这个 workshop 的重心是"用 ML 优化系统"，但 2026 CFP 明确加入了 "ML for large-scale AI infrastructure (training, serving, scheduling of LLMs)"、"Systems for LLMs"、"benchmarks, datasets, simulators, evaluation methodologies"。我们的工作是"系统 for LLM"加上 cache 策略评测，可以算进去，但 score-EMA 不是 ML 方法，需要把重点放在 trace/simulator/evaluation methodology 上 | ≤4 页（严格），不含参考文献和附录；NeurIPS 2026 模板；**不要求匿名**；non-archival，网站可选挂链接；OpenReview（NeurIPS.cc/2026/Workshop/MLForSys） | 没有明确说欢迎负面结果。欢迎 benchmark、simulator、evaluation methodology | 2026：截稿 2026-08-29 AoE（页面上还有一行冲突的"8 月 31 日"），通知 09-29（已截稿）。**2027 未公布**。按惯例 NeurIPS 2027 的截稿约在 2027 年 8 月下旬 | 2025 录用（https://neurips.cc/virtual/2025/workshop/109537）中与 serving 相关的有：LLMVisor（多租户延迟归因）、PORT（多 LLM 路由）、Learning to Shard（分布式推理并行度）、Deep-Learning-Driven Prefetching for Far Memory、A Joint Learning Approach to Hardware Caching and Prefetching。**只看了标题**，从题目看多是 4 页的原型加模拟或小规模真机实验 |
| **Long-Context Foundation Models (LCFM)**（NeurIPS 2026 第 3 届） | https://longcontextfm.github.io ；https://neurips.cc/virtual/2026/workshop/137572 | **中高**：主题包括 "Efficiency techniques for foundation models" 和长上下文推理；我们的场景就是长文档 decode | short ≤4 页 / long ≤8 页（不含参考文献和附录）；双盲；non-archival，允许 preprint 和一稿多投 | 没有专门提负面结果 | 2026：截稿 09-10，延到 09-13，通知 09-29（已截稿），会议在 Atlanta。**2027 未公布**（之前在 ICML 2024/2025 办过，下一届可能回到 ICML） | 2026 录用列表这次没有读到。（未核实） |
| **AXIOM: Foundations of Efficient Deep Learning**（NeurIPS 2026，Paris） | https://neurips.cc/virtual/2026/workshop/137620 | 中：sparsity & compression、hardware-aware learning 在主题内，但偏理论（"theory-guided efficient AI"） | 未查到 | 未查到 | 未查到截稿日期（应该已截稿）。会议 2026-12-12/13。**2027 未公布** | 未查到 |
| **AdaptFM: Resource-Adaptive Foundation Model Inference**（ICML 2026） | ICML 2026 workshop 列表 #33 | 中高：资源预算自适应推理 | 未细查 | 未细查 | 已结束（ICML 2026 workshop 在 07-10/11）。**2027 未公布** | 未查到 |
| **Efficient Reasoning**（NeurIPS 2025 第 1 届） | https://efficient-reasoning.github.io | 中：CFP 的 System Deployment 部分提到 "dynamic KV-cache placement" | ≤4 页（第二轮可到 9 页）；non-archival | 没有专门提 | 2025 已结束；NeurIPS 2026 列表里**没有**；**2027 未公布** | best paper 是 M1（Mamba reasoning），oral 都是推理算法，没有找到 KV/serving 论文 |
| **EuroMLSys**（与 EuroSys 同址） | 第 6 届 @ EuroSys '26（Edinburgh，2026-04-27），https://euromlsys.eu | **高**：ML 与系统交叉，LLM 优化是明确主题；系统社区审稿 | ≤6 页（含图表）；单盲。ACM 出版，往届是 archival 的 workshop proceedings（**这次没有核实 2026 年的 archival 政策**） | 没有专门提 | 2026：截稿 2026-02-15（另一份 CFP 写 02-24）。**2027 未公布**；按惯例应在 2027 年 2 月左右截稿（推测） | 未抽样 |
| **MLSys 2027 workshops** | MLSys 2027：Bellevue，2027-06-21~25（https://mlsys.org） | 未知 | 未知 | 未知 | **workshop 征集和名单都未公布**（主会论文截稿 2026-10-30，已过）。MLSys 2026 有哪些 workshop 这次也没有查到。往届 workshop 提案约在会前 6 个月截止，论文截稿多在 4–5 月（推测） | — |
| **ICLR 2027 workshops** | ICLR 2027（San Francisco；workshop 在 4 月 29/30 日，官方页面把年份误写成 2026） | 取决于录用名单；**关键是看 ICBINB 是否继续办** | ICLR 没有统一页数，由各 workshop 自定（ICBINB 2026 的 tiny paper 页数自定为 2 页） | — | 官方时间表：workshop 录用名单 **2026-12-07** 公布；**建议的 workshop 论文截稿 2027-02-01**；必须在 2027-02-26 前通知作者。（https://iclr.cc/Conferences/2027/CallForWorkshops） | — |
| **ICML 2027 workshops** | ICML 2027（据非官方来源在南美，城市未定） | 未知 | — | — | **workshop 征集未发布**。2026 年的 workshop 申请截止是 02-13，通知截止 05-14；以此类推，2027 年 workshop 论文截稿大概在 2027 年 5 月（推测） | — |

## 各 workshop 补充说明

### ICBINB（首选，前提是 2027 年有 LLM 主题的一届）
- ICBINB 每届主题会变：ICLR 2026 是 LLM，NeurIPS 2026 是 biology。**能不能投取决于 ICLR 2027 或 ICML 2027 那一届的主题**。2026-12-07 ICLR 2027 名单公布后要立刻查看。
- 写法要对齐 full paper 的四要素模板：(1) 问题：GPU 只放 B 页时，跨步复用的淘汰策略能否逼近 Bélády，指标是 miss 字节、gap-closed、延迟。(2) 来自文献的方法：Quest 页选择 + LRU/FIFO/score，"用 selection rank 做淘汰"是一个合理的直觉。(3) 有定量证据的负面结果：预注册的 rank 假设被否定；score-EMA 只拿到约 28% 的差距；字节 −9–19% 但 L4 延迟持平。(4) 失败原因分析：复用间隔分布、rank 与下次复用的相关性、传输不在关键路径上的区间分析。
- 审稿人看重 rigor & transparency：预注册、held-out、配对统计是我们的强项，正好对应 Didactic/Entropic 奖的评选口味。

### ES-FoMo（如果 2027 年恢复）
- 偏好"完整系统工作的短版"，以及可复现的代码和 benchmark。我们的 fused Triton kernel、回放模拟器、Bélády 计算可以作为卖点。
- 审稿只看 correctness & relevance，门槛相对宽松。但录用论文普遍有 kernel 级测量，所以延迟拆分（见 echo.md 启示 2）是必需的。

### ML for Systems
- 不要求匿名，严格 4 页。要把重点放在 evaluation methodology（预注册、held-out、oracle gap 指标）和 trace/simulator 上，而不是 score-EMA 本身。

### EuroMLSys / MLSys workshops
- 系统社区审稿，会更看重真机测量、多种硬件或带宽区间、开销计入。L4 + batch 1 单一配置会是主要弱点。投这类 venue 之前，最好补一个带宽或 batch 的 regime 分析。

## 时间线建议（以 2026-11-25 为基准）

1. **2026-12-07**：查 ICLR 2027 workshop 录用名单，重点找 ICBINB、效率/KV/sparsity/长上下文类 workshop。论文截稿大多会在 **2027-02-01** 前后。
2. **2027 年 2 月**：EuroMLSys 2027（待确认）。
3. **2027 年 4–5 月**：ICML 2027 workshop（待确认是否有 ES-FoMo IV、LCFM、ICBINB），以及 MLSys 2027 workshop（待公布）。
4. **2027 年 8 月下旬**：NeurIPS 2027 workshop（ML for Systems 大概率继续办，ICBINB 主题待定）。

## 来源

- ICBINB ICLR 2026：https://sites.google.com/view/icbinb-2026/home ，/submit ，/reviewer-guidelines ；录用列表 https://iclr.cc/virtual/2026/workshop/10000772
- ICBINB-BIO NeurIPS 2026：https://icbinb-bio.github.io/submit/
- ES-FoMo：https://es-fomo.com/ ，https://es-fomo.com/call/ ；ICML 2026 workshop 列表 https://icml.cc/virtual/2026/events/workshop
- ENLSP-IV：https://neurips2024-enlsp.github.io
- NeurIPS 2026 workshop 列表：https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/
- ML for Systems 2026：https://mlforsystems.org/call_for_papers.html ；2025 录用 https://neurips.cc/virtual/2025/workshop/109537
- LCFM：https://longcontextfm.github.io
- AXIOM：https://neurips.cc/virtual/2026/workshop/137620
- Efficient Reasoning：https://efficient-reasoning.github.io
- EuroMLSys：https://euromlsys.eu
- ICLR 2027 workshop 时间表：https://iclr.cc/Conferences/2027/CallForWorkshops
- MLSys 2027：https://mlsys.org
- ICLR 2026 workshop 列表：https://iclr.cc/virtual/2026/events/workshop
