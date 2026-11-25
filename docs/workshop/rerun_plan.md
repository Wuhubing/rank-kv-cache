# 在 wulab2 上从头重跑：执行计划

草拟于 2026-11-25，**待审阅，尚未开始执行**。配套文档：[README.md](README.md)（投稿准备与补充实验清单）。

## 0. 目标和不能违反的规则

**目标**：在 wulab2 上把两条研究线的数据从头采集一遍，补齐 workshop 需要的实验，并把"换一种 GPU 结论是否还成立"做成一条 robustness 证据。

**规则**
1. **冻结的东西不改**：`splits/preregistration.json`、`splits/heldout_frozen_v1.json`、两份 split 文件、已提交的结果文件都保持原样。新结果只写进新目录。
2. **原有 held-out 只能再评估一次**：用冻结参数在新 GPU 上重新采集，按事先写好的复现预注册评估一次。不在新的 held-out 上调任何参数。
3. **探索只在 dev 上做**：全 page 排名对照、预算扫描、各种消融都只用 dev。
4. **新模型或新任务要用新的 held-out**：从 LongBench 另抽文档，先写预注册再采集。
5. **一个分支只让一台机器写**：wulab2 上的工作分支只从 wulab2 推送。
6. **不备份到 Drive**：原始 trace 只放在 wulab2 的 `/data/weibing`，仓库里只提交小的结果文件和 manifest（含 sha256）。
7. **只用一张卡**：开跑前用 `nvidia-smi` 挑一张空闲的卡，用 `CUDA_VISIBLE_DEVICES` 固定，全部采集和 benchmark 都在这张卡上做，卡号和 GPU UUID 写进 manifest。长任务放在 tmux 里跑。

## 1. 机器情况（2026-11-25 实测）

| 项目 | 情况 | 影响 |
|---|---|---|
| GPU | 4 张 RTX 6000 Ada，每张 48 GB，当时空闲 | **只用其中一张**：单卡显存够 8B 模型加 64K 上下文；全部 trace 出自同一张物理卡，对比更干净，也不占满共享资源 |
| 驱动 | 550.90，最高 CUDA 12.4 | 之前用的 torch 2.14.1+cu130 不能用，要换 cu124 的 torch |
| 根分区 | 只剩 127 GB（已用 99%） | 家目录不放任何大文件 |
| `/data` | 剩 9.9 TB，已有 `/data/weibing` | 模型、HF 缓存、trace、结果都放这里 |
| CPU / 内存 | 48 核，503 GB | CPU 回放可以开多进程 |
| 网络 | 能访问 Hugging Face 和 GitHub | 模型和数据直接下载 |
| 访问 | MacBook 和 mini 都能免密 `ssh wulab2` | — |

## 2. 第 0 步：开跑前要决定的事（需要你确认）

| # | 决定 | 建议 |
|---|---|---|
| D1 | 原 held-out 怎么用 | 冻结参数做**跨硬件复现**，评估一次；不在新 GPU 上重新调参 |
| D2 | 代码基线 | 先合并 #6（合到 `stage1-2`）和 #7（合到 `efficient-policy`），再从合并后的 `efficient-policy` 开 `rerun/wulab2` 分支 |
| D3 | 谁来执行 | mini 上的 Claude 通过 SSH 操作 wulab2，按本计划分阶段推进；每个关卡由 MacBook 这边审查 |
| D4 | 第二个模型 | Qwen3-8B（与 0.6B 同一模型族，chat 模板和 thinking 开关一致）；备选 Llama-3.1-8B |

## 3. 阶段 1：环境（约半天）

1. 在 `/data/weibing/rank-kv-cache` 克隆仓库，切到 `rerun/wulab2`。
2. 设置 `HF_HOME=/data/weibing/hf`，避免写进家目录。
3. 用 uv 建环境：`transformers==4.53.3`（与原实验一致），torch 选一个 cu124 的版本并在 manifest 里记录。numpy、pytest、rouge 1.0.1 照旧。
4. 按 split 文件里固定的 revision 下载 Qwen3-0.6B 和 LongBench。
5. 跑全部单元测试（PR #7 上是 155 个通过、1 个跳过），以及 GPU 相关测试。
6. 跑选择器自检：notebook 里已有和暴力计算比对的检查。
7. **同卡确定性**：同一篇文档在选定的那张卡上跑两次，逐行比较 trace，应完全一致。因为全程只用这一张卡，不需要再检查卡与卡之间是否一致。

**关卡 1**：测试全过，自检通过，同卡重跑完全一致。

## 4. 阶段 2：把 Colab 采集改成服务器脚本（约 1–2 天）

现在的采集代码在 `notebooks/first_trace.ipynb`（短文档，trace v1）和 `notebooks/long_trace_capture.py`（长文档，schema v2 NPZ），都按 Colab 写的。

1. 新建 `collect/short.py` 和 `collect/long.py`，选择器、prompt 模板、停止规则、精度逻辑**原样搬过来**，不改算法。
2. 命令行参数：`--split`、`--subset dev|heldout`、`--docs`（只跑指定文档，方便试跑和补跑）、`--out`、`--dense-reference`。GPU 由外部的 `CUDA_VISIBLE_DEVICES` 指定。
3. manifest 记录：GPU 型号和 UUID、驱动、torch、transformers、模型和数据的 revision、prompt token 的哈希、生成的 token、trace 的 sha256。
4. 支持断点续跑（按文档判断是否完成），写文件时先写临时文件再改名。
5. 单进程、单卡、按文档顺序跑；放在一个 tmux 会话里，进程号和进度写进状态文件。
6. 加入采集结果的格式校验，直接复用 #7 里更严格的 NPZ 读取校验。

**关卡 2**：短、长各跑 1 篇文档，整条链路"采集 → 校验 → 回放 → 出表"跑通；记录每篇耗时，用来估算后面的总时长。

## 5. 阶段 3：重采 dev（单卡 GPU）

单卡耗时粗估（关卡 2 实测后更新）：长文档每篇一两分钟（1024 步加 32K prefill），14 篇加 dense 参考约半小时到一小时；短文档 20 篇几分钟。

1. 排名研究：20 篇 dev（`splits/longbench_v1.json`），128 token，附带 dense 参考生成。
2. EMA 研究：14 篇 dev（`splits/longctx_v1.json`），最多 1024 步，thinking 开启。
3. **和 L4 的 dev 对比**：
   - 选择一致率：同一 prompt 在 L4 和 Ada 上，(层, 步) 选出同一组 page 的比例。L4 的 trace 只在共享 Drive 上，本机没有，如果拿不到就只比较汇总指标。
   - 汇总指标：LRU 与 Bélády 的差距、EMA 的 G、H1 效应，看是否在原来的置信区间附近。
   - 跨硬件翻转的 page 是否集中在 top-K 边界附近，这是"选择随 GPU 变化"这个附带发现的检验。

**关卡 3**：dev 采集完整、校验通过；与 L4 的对比写成一页小结。

## 6. 阶段 4：dev 上的 P0 实验（CPU 回放，约 1 周）

全部用阶段 3 的新 dev trace，48 核并行回放。每项都写成脚本，输出放在 `results_wulab2/dev/<实验名>/`。

| # | 实验 | 现有代码 | 要回答的问题 |
|---|---|---|---|
| 1 | 全 page 排名对照：z、allrank、只看选中的 rank | `validation/rank_control_dev.py`（#7） | 有用的是分数大小，还是全局排序 |
| 2 | 同分配 Bélády | `validation/alloc_oracle.py`（#7） | 按层分配到底补上了多少真实差距 |
| 3 | 预算扫描：B/K ∈ {1, 1.25, 1.5, 2, 3, 4, 6, 8}，加等 miss 容量 | 扩展 `efficient_dev.py` | 回应"容量比策略重要" |
| 4 | 基线补齐：ArkVale 当前分数（固定 sink 和 recent page）、InfLLM 衰减分数（代理版）、hit 优先 LRU、FIFO、Random、SIEVE、S3-FIFO | 在 `kvpolicy.py` 里加策略 | 和同类工作正面比较 |
| 5 | 信号 × α：raw、z、allrank、last rank、当前分数，α 从 0 到 1 | 扩展 `ScoreEMA` | EMA 的作用到底来自哪里 |
| 6 | CO2 式析因：衰减 × recency 保护区 × 最短驻留 | 新策略参数 | 简单的 recency 区能复现多少收益 |
| 7 | 统计：每篇的 G 分布、配对 bootstrap、噪声注入 | 新脚本 | 分数不可靠时 G 是否退回 LRU 而不是更差 |
| 8 | PRR 式重叠：EMA 预测集（大小 K）与下一步 top-K 的重叠 | 新脚本 | 和 PRR 的数字可比 |
| 9 | 排名研究的事后对照：`score:8:2` 对 `score:0:2` | `posthoc_rank_increment.py`（#6） | 在新硬件上复查 |

**关卡 4**：每项给出一页结论。根据第 1 项的结果定下论文的核心表述。

## 7. 阶段 5：跨硬件复现（先写预注册，再采 held-out，只评估一次）

1. **先写** `docs/replication_wulab2.md` 并提交：
   - 所有参数原样沿用两份冻结配置（δ、β、逻辑回归系数、EMA 的 α、按层分配方案），不允许用阶段 3–4 的结果改动。
   - 列出评估命令和判定标准：排名研究的 S1、S2 是否复现；EMA 的 G 及其 CI、21 篇中胜过 LRU 的篇数、加分配后的降幅。
   - 写明"偏离"的定义，以及复现失败时报告什么。
2. 提交之后，再采集 40 篇短 held-out 和 21 篇长 held-out。
3. 按预注册的命令各跑一次，结果写进 `results_wulab2/heldout/`，不覆盖原来 L4 的结果。

**关卡 5**：复现报告，包括与 L4 结果的并排对比和所有偏离。

## 8. 阶段 6：P1 扩展（GPU，按需）

每项都需要新文档时，先在 dev 上探索；要做确认性评估，就另抽 held-out 并单独写预注册。

1. **第二个模型 Qwen3-8B**：同样的 schema v2 trace。64K 上下文下权重约 16 GB、KV 约 9.4 GB（36 层 × 8 KV head × 128 维 × K/V × bf16），单张 48 GB 的卡放得下。速度约为 0.6B 的 1/3 到 1/5，单卡几个小时量级。
2. **64K 上下文**，以及 RULER 和 SCBench 多轮。
3. **page 大小和 K**：page ∈ {8, 16, 32, 64}，K ∈ {32, 64, 128}。
4. **trace 加字段**：每个选中 page 的真实注意力质量（用于真正的 InfLLM 基线），以及 Quest 分数与真实注意力的秩相关。
5. **采样解码**：temperature > 0，3 个 seed。

## 9. 阶段 7：系统重测（GPU）

1. 用 #7 修好的 `gpu_bench.py`（warmup 生效，显存按阶段报告）在采集用的同一张 RTX 6000 Ada 上重测，测时确认这张卡上没有别人的任务。
2. 报告：每层策略开销（µs）、延迟分解（选择器 / 取数 / 注意力 / 其余）、PCIe 字节、显存；关掉 host 取数的对照；证明各策略的输出逐 token 一致。
3. 补上 FLOPs/MACs 开销表的实测部分（估算见 README 第 2.3 节）。
4. 如果要冲主会，再做 batch > 1 和带宽受限的场景。

## 10. 时间线（粗估，关卡 2 实测后更新）

| 周 | 内容 |
|---|---|
| 第 1 周 | 阶段 0–2：决定、环境、采集脚本、单篇试跑 |
| 第 2 周 | 阶段 3：重采 dev；开始阶段 4 |
| 第 3 周 | 阶段 4 完成；写复现预注册 |
| 第 4 周 | 阶段 5：采 held-out、评估一次；开始阶段 6 的 8B 模型 |
| 第 5–6 周 | 阶段 6–7，写 workshop 论文 |

## 11. 风险

| 风险 | 应对 |
|---|---|
| 根分区满了导致任务失败 | 所有路径指向 `/data/weibing`；启动前检查 `HF_HOME` 和临时目录 |
| 选定的那张卡被别人占用 | 每次开跑前检查；被打断可按文档续跑。如果必须换卡，记录新卡的 UUID，并在新卡上重跑一篇已有文档，确认 trace 与原卡一致后再继续 |
| cu124 的 torch 与原来的数值行为不同 | 本来就是新硬件；把版本写进 manifest，并在复现报告里说明 |
| 分数存为 float16，平局更多 | 照原样保留，以保证和原实验一致；在复现报告里分析平局比例 |
| 长上下文显存不够或太慢 | 关卡 2 先测 32K；8B 和 64K 放到最后 |
| 原始 trace 只有一份 | 不备份到 Drive（按你的要求）；manifest 里记录 sha256，丢失后可以按同样配置重采 |

## 12. 交付物

- `rerun/wulab2` 分支：采集脚本、新实验脚本、`results_wulab2/` 下的小结果文件。
- 每个关卡一页小结，放在 `docs/rerun/`。
- 复现预注册和复现报告。
- 更新 workshop README 里的数字和图。
