# 结论总结（分支 `xiaoyang/score-ema-final`）

每条结论后面的 **[证据]** 是本分支里可以直接重算的文件；复核步骤见 [`REVIEW.md`](REVIEW.md)。

## 一句话

Proposal 问的是：sparse-attention 的 selection rank 能否在 recency 和 frequency 之外改进 HiSparse 式 KV page cache 的替换。**答案是否定的**（预注册检验，rank 策略只比 LRU 少 0.39% 的 repeat miss，目标 10%）。驱逐真正需要的是**未被选中的驻留 page 的当前 Quest 分数**：每个 page 只存一个浮点数的 score EMA 填上 LRU 到 Bélády 差距的约 28%，再加按层分配约 40%。这一结果在 L4 和 Apple M1 Max 两种硬件上各自通过了预注册检验，在新文档和新任务上也成立。系统上，省下的 miss 按每个 page 约 5.3 µs 线性变成 GPU 时间，但 page 传输只占每步 GPU 时间的 9–14%，所以每步提速是 EMA +1.4–1.5%、分配 + EMA +2.0–2.1%，Bélády 上限 +5.2–5.4%（B=96，batch ≥ 16）。

## 设定

Qwen3-0.6B，bf16，Quest 式选择器（16-token page，K=64，各 query head 的上下界取平均，每层一个排名、一个缓存）。每步 top-K page 必须驻留，策略只决定驱逐哪些"驻留但本步未选中"的 page。G = (LRU miss − 策略 miss) / (LRU miss − Bélády miss)，CI 为按文档有放回抽样 2000 次（seed 0）。

## 1. Proposal 原题：rank（短上下文，L4，团队 Wuhubing 完成）

4–16K token，生成 128 token，20 篇 dev 调参并预注册，40 篇 held-out 只评估一次，B=128。

| 问题 | 结果 | 结论 |
|---|---|---|
| 改进空间 | LRU 4.05%，Bélády 2.97%（绝对差 1.08 pp，相对 26.7%） | 相对空间大，绝对空间小 |
| H1：rank 在 recency × frequency 之外是否有信号 | 广义候选（之前被选过、本步未选中，不一定驻留）上 top 与 bottom 四分位复用率差 +3.1 pp（1 步内）到 +10.8 pp（8 步内），CI > 0；只看 LRU 实际驻留候选，8 步内 +13.0 pp | 支持，但 top 四分位只占广义候选的 0.73% |
| H2：加 rank 是否减少 miss | 完整策略（δ=8, β=2）比 R+F（δ=0, β=4）少 330 次 [83, 600]，占 0.22% | 统计上支持，实际可忽略。330 是两组 dev 最优参数之差，不是 rank 单独的贡献 |
| 主成功标准 S1：比 LRU 少 ≥ 10% repeat miss | 少 0.39% [0.20%, 0.57%] | **不满足** |

**[证据]** `docs/STATUS_REPORT.md` §4.7，`docs/results/heldout/eval/success.json`，`docs/results/heldout/eval/h1_effects.csv`。

不在本分支、但与此相关的个人版短实验（Xiaoyang 的另一套 60 篇划分，与上表的 held-out 有 7 篇重合，不能合并成 80 篇）：完整 rank 策略省 0.405%；固定 β 时 rank 单独增量 138 次 [65, 215]；在同一设定下记录分数后，当前分数省 15.45% [13.76%, 16.88%]、score EMA 省 18.13% [16.76%, 19.32%]（未预注册，α 取自长上下文 dev）。这部分的数据和复核在个人交接包 `KV_Cache_Agent_Handoff`，不在本分支。

## 2. 为什么 rank 不行：信息消融（长上下文 dev）

16–32K token，带思考的长摘要，338–1023 步，B=96（B/K=1.5），14 篇 dev，按文档留一选参。

| 策略 | 用到的信息 | G（M1 Max） | G（L4） |
|---|---|---|---|
| Score EMA + 按层分配 | 每步所有驻留 page 的当前分数 | +0.400 | +0.409 |
| 线性下次使用回归（6 特征，DAgger） | 同上 + recency、频率 | +0.297 | +0.300 |
| Score EMA（每 page 1 个浮点数） | 同上 | **+0.278** | +0.279 |
| InfLLM 式衰减分数和 | 同上 | +0.264 | +0.269 |
| 只用当前分数 | 同上，不平均 | +0.240 | +0.244 |
| 衰减选中计数 | 只有被选中时看到的信息 | +0.040 | +0.044 |
| Rank 策略（proposal） | 同上 | +0.030 | +0.027 |
| EMA，用 rank 代替分数 | 同上 | +0.016 | +0.016 |
| Score EMA，只在被选中时更新 | 同上（分数会过时） | −1.595 | −1.596 |

关键是信息而不是规则：同一个 EMA，每步更新所有驻留 page 是 +0.278，只在被选中时更新是 −1.6，换成 rank 是 +0.016。EMA 的输入 z = (s − s_K)/(s_1 − s_K) 衡量的就是 page 离本步入选门槛（第 K 名分数）有多近。Score EMA 本质上是 InfLLM 的衰减分数规则用在 Quest 上界分数上，不是新规则。

**[证据]** `docs/results/mps_dev/checks_report.txt`、`docs/results/mps_dev/REPLICATION.md`、`docs/results/longctx_dev/checks_report.txt`。

## 3. Score EMA + 按层分配：预注册 held-out、两种硬件、新数据（B=96）

预注册标准：(i) EMA 的 G 的 CI 下界 > 0.15；(ii) EMA 优于 LRU 的文档 ≥ 18/21；(iii) 分配 + EMA 相对 EMA 的配对 CI > 0。

| 数据 | 采集硬件 | 文档 | EMA 的 G [CI] | 优于 LRU | ΔG（分配） | 分配 + EMA |
|---|---|---|---|---|---|---|
| longctx_v1 held-out（预注册） | L4 | 21 | 0.281 [0.277, 0.285] | 21/21 | +0.120 [0.113, 0.128] | 0.402 |
| longctx_v1 held-out（预注册） | M1 Max | 21 | **0.280 [0.276, 0.283]** | **21/21** | **+0.120 [0.111, 0.128]** | 0.399 |
| 新 GovReport（>32K，截中间） | M1 Max | 3 | 0.284 | 3/3 | +0.096 | 0.381 |
| NarrativeQA（新任务） | M1 Max | 8 | 0.283 | 8/8 | +0.129 | 0.411 |

- 两台机器生成的 token 不同，同一篇的 LRU miss 相差 0.80–1.37 倍；但同一篇的 EMA G 两边最多差 0.015，分配 + EMA 最多差 0.030。
- M1 Max 的冻结配置只来自 M1 Max dev；预注册（commit `1b6fe70`）早于任何 held-out 采集和结果（`29e59ee`）。
- M1 Max 主检验用的 21 篇团队在 L4 上评估过，所以它检验的是"换硬件、重新从 dev 冻结后是否仍成立"；新文档的两组单独报告，不判定成功。LongBench 里够长的 GovReport/QMSum 只剩 3 篇（QMSum 0 篇）。

**[证据]** `docs/heldout_results_mps.md`、`docs/preregistration_mps.md`、`docs/results/{mps_heldout,v2_fresh,v2_ood,longctx_heldout}/misses.csv`、`recompute_heldout.py`。

## 4. 系统：少掉的 miss 能换成多少时间

- **Batch 1，整模型（团队，L4）**：真实两级卸载 cache；融合 Triton kernel 后 EMA 的策略开销与 LRU 相同，PCIe 流量少 9–19%，但延迟持平（LRU 62.3、EMA 62.3、分配 + EMA 61.6、Bélády 62.1 ms/token），瓶颈是 CPU 派发。**[证据]** `docs/gpu_benchmark.md`、`results_gpu/`。
- **Batch 1–64，层级 GPU 时间模型（本分支，L4，两个会话共 4 次测量）**：按 21 篇 M1 Max held-out 的逐步 miss 数做零拷贝 gather，加上同 batch 的解码和 Quest 打分 GPU 时间；不含 CPU 启动间隙、策略 kernel 本身的开销和计算重叠。

| B | batch | LRU 传输占比 | EMA 提速 | 分配 + EMA | Bélády |
|---|---|---|---|---|---|
| 96 | 1 | 6.8–6.9% | +0.67% | +0.85–0.86% | +2.40–2.42% |
| 96 | 16 | 13.6–13.8% | +1.43–1.45% | +2.03–2.07% | +5.22–5.32% |
| 96 | 64 | 13.8–14.2% | +1.41–1.46% | +2.02–2.08% | +5.23–5.40% |
| 128 | 64 | 8.7–8.9% | +0.93–0.96% | +1.81–1.87% | +3.52–3.64% |

每个 page 5.3–5.9 µs（batch ≥ 16 时 5.3–5.5 µs，约 12 GB/s）。传输时间两次会话相差 < 0.1%，提速最多差 0.15 个百分点。若打分不花时间，batch 64、B=96 的传输占比约 22%，提速为 EMA +2.3%、分配 + EMA +3.3%、Bélády +8.8%；若传输与计算完全重叠，收益会消失（未测）。

**[证据]** `docs/batch_benchmark.md`、`docs/results/batch_bench_l4/`、`docs/results/batch_bench_l4_run2/`、`docs/results/mps_heldout/miss_schedule.npz`。

## 可以说与不可以说

可以说：rank 有统计显著的增量信号，但策略层面只少 0.4%；驱逐需要未选中驻留 page 的当前分数；score EMA 填上约 28%、加分配约 40% 的差距，两种硬件各自通过预注册检验，在新文档和新任务上成立；省下的 miss 按比例变成时间，但在此配置下提速上限约 5%。

不可以说：score EMA 是新的替换算法（它是 InfLLM 式规则用在 Quest 分数上）；"首个基于重要性的驱逐"或"首个用过去重要性做可召回替换"（InfLLM 已做过）；不带条件的"变快了 X%"（整模型 batch 1 持平；提速数字来自分开计时的层级模型）；结论适用于 token 级缓存、其他模型或其他选择器（未测）；M1 Max 主检验是新文档上的检验；把 330 次当作 rank 单独的贡献；把两套短实验的 held-out 合并成 80 篇。

局限：一个模型（Qwen3-0.6B；1.7B 与第二种 prompt 只在 dev 上抽查过）、一种选择器、长摘要类解码；held-out 21 篇加 11 篇新文档；InfLLM baseline 用 Quest 上界的 softmax 代替真实注意力；trace 依赖硬件（bf16 近似平局），各硬件只在自己的 trace 内比较；GPU 时间只在 L4 上测过。

## 本分支相对团队分支 `efficient-policy`（经 `xiaoyang/validate-wuhubing`）新增了什么

- 修复复核问题：rank baseline 的 tie-break、AUC 平局处理（`7bc0f2c`），held-out 驱动按存储的 rid 检查并读取冻结 α（`heldout.py`）。
- `local/capture.py`：MPS/CUDA/CPU 通用的 trace 采集（分块 prefill、自检、看门狗）。
- M1 Max 上独立重采集 14 篇 dev、21 篇 held-out、11 篇新文档，以及预注册和结果。
- 层级 batch 扩展 benchmark（`notebooks/batch_transfer_bench.ipynb`）和两次 L4 运行结果。
- Poster 图（`poster/make_figs.py`，只读已提交的 CSV）。
