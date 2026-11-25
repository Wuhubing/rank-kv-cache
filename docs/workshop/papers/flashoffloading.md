# FlashOffloading 阅读笔记

来源：arXiv HTML 全文 v1（https://arxiv.org/html/2611.07542v1），包括正文、Table 1 和 Fig.9/10/11 的 PNG 原图。Fig.9 的数字是我从图上目测读出的，误差约 ±10 token。

## 基本信息

- **标题**：FlashOffloading: Low-Latency KV Cache Offloading for Long-Context LLM Serving
- **作者**：Xuan Wang、Zhuanhao Wu、Tingzhu Fang、Xushi Li、Sicheng Zhang、Zhenyu Yang、Feng Liu、Zhaohui Fan、Qiang Xie、Huayi Zou、Yuncong Liu、Zeyu Song、Ke Zhu、Yanjia Zeng、Jingchun Gao、Yan Shi、Kui Luo、Gezi Li、Liangjun Feng、Cheng Li（通讯作者），来自 Huawei / Huawei Ascend / SJTU。
- **状态**：arXiv preprint，cs.AI，2026-11-05（v1）。用的是 MLSys 模板，抬头写着"8th MLSys Conference 2025"，看起来是模板残留，HTML 还提示"page layout violates the ICML style"。正式 venue 未知。
- **代码**：https://github.com/vllm-project/vllm-ascend/tree/releases/v0.30.0（vLLM 0.30 + vLLM-Ascend 0.30 的修改版）

## 设置

| 项 | 内容 |
|---|---|
| 模型 | 只有一个：GLM-5.2-w4a8c16（MLA，78 个 KV 层，D_KV=576；DSA indexer 21 层，D_index=128；IndexShare 每 4 层共享一次选择），decode 时开 MTP3 speculative decoding。每 token KV 为 95.23 kB（含 indexer）。 |
| Selector | DSA Lightning Indexer，top-2048。MTP3 下每步有 4 路 query（j=0..3），各自做 top-k，miss 取并集。 |
| 选择单元 | **token 级**（单个 KV entry 约 1 KB，fine-grained）。 |
| K | 2048 token。 |
| Cache budget | 每请求每层 hot cache 大小 \|C_b\|：trace 扫 8K–32K token（步长 2K）；kernel 微基准固定 12288；端到端 8192–30720，随 batch 调整，64K 时用 16K；"典型配置" 14336。约为序列长度的 1/8–1/4，即 4×k–15×k，**远大于 HiSparse 的 2k–4k**。 |
| 上下文 | trace：Open-SWE 20 条请求，10 条短（66K–92K）+ 10 条长（102K–131K）。端到端输入 64K/128K/256K/512K/1M，输出 1 或 8192。prefill 消融用 256K 输入、50% prefix hit。 |
| 数据/任务 | Open-SWE 长上下文任务（用于收集 trace）；端到端是 burst 并发的合成请求，90% prefix hit。不测准确率。 |
| Batch | kernel 微基准 bs 1–32；端到端每个 DP rank 的 decode batch 为 1–15（生产配置 12）。 |
| 硬件 | Ascend 910C NPU（每卡 2 die，每 die 64 GB HBM）。kernel 实验用单 die；端到端用 4 台服务器 32 卡 64 die，1P1D：P 节点 DP2/TP16/EP32，D 节点 DP16/TP2/EP32。host 链路写作"PCIe/HCCS"，没有给带宽数字。P→D 用 H2H 写入 D 节点的 host DRAM。 |

## 方法与基线

- **Prefill**：4 个 full-layer buffer，B0 常驻第 0 层，B1–B3 轮转，H2D 预取、计算、D2H 写回三者流水。这部分与我们的工作关系不大。
- **Decode**：
  - `cache_slots` 是反向索引（token→slot，int32，长度 \|S_b\|），把 miss-lookup 变成 O(1)，管理复杂度变为 O(k)+O(\|Miss\|)，与 \|C\| 解耦。
  - **LIM kernel**：在 top-k 之前把 index 和 slot 编进同一个 int32，top-k 结束时 miss 结果自然得出，同时完成 eviction 选择和 index 更新。
  - **CopySFA**：在 SFA 的 KVgather 里按 hit/miss 分支，miss 直接从 pinned host 读，与 hit tile 的 Q×K 计算重叠。
- **Trace 中比较的策略（§6.3.1，Fig.9）**：只改 step③ 的 eviction 选择，其余流程相同（MTP3 下 4 路 miss 取并集，逐出数等于 miss 数）。
  - Random：随机选择；
  - LRU：least-recently selected；
  - Score：逐出 indexer score 低的 token，具体是瞬时 score 还是累积 score，文中没有说清；
  - **LIM（ours）**：score threshold + chunk scan，是为了 kernel 低延迟设计的近似策略。
- **Oracle**：**没有 Bélády，也没有 no-IO oracle。**
- **Learned predictor**：没有。在相关工作部分批评 ECHO 依赖 EMA 预测 top-k score 来做 overlap，自己采用 exact、非预测的 intra-kernel 重叠。
- **系统基线**：无 offload（none）、P-only、P+D。**不和 HiSparse/ECHO/ESS 做系统级对比**，只在 trace 上拿"HiSparse 典型配置"（LRU，\|C\|=8192）作比较。

## 评价指标

- **Miss**：每请求每层的平均 miss token 数，按 MTP3 4 路 query 取并集，是绝对数，不是比率。按 k=2048 换算，约 150–500 token，相当于 7%–24%。注意并集的分母不是 2048，而是 4 路去重后的选择数，可能大于 2048。
- **管理开销**：LIM−LI 的 kernel 耗时差（µs），单 die。
- **重叠效果**：CopySFA 对比 Copy+SFA 串行；OverlapRatio = (T_copy+T_SFA−T_fused)/T_copy。
- **端到端**：P99 TE2E、P99 TTFT、平均 TPOT、平均 ITL、平均 input/output TPS；server 端记录 HBM 占用（权重/KV/激活）、max-num-batched-tokens、每 DP rank 的实际 batch，以及 kernel 级 ITL 分解（Fig.13）。
- **SLO**：吞吐都是在 30 ms TPOT 约束下报告。
- **没有报告**：bytes/PCIe 流量、准确率、FLOPs、方差/seeds。

## Robustness / 消融

1. **Fig.9：策略 × budget（核心）**。纵轴是每请求每层的平均 miss token 数，4 路 MTP 取并集。下面是目测读数：

   **短组（66K–92K）**

   | \|C_b\| | Random | LRU | Score | LIM |
   |---|---|---|---|---|
   | 8K | ~448 | ~382 | ~355 | ~460 |
   | 10K | ~368 | ~305 | ~290 | ~372 |
   | 12K | ~310 | ~255 | ~243 | ~313 |
   | 14K | ~265 | ~222 | ~213 | ~268 |
   | 16K | ~235 | ~200 | ~192 | ~237 |
   | 20K | ~192 | ~172 | ~168 | ~193 |
   | 24K | ~168 | ~155 | ~153 | ~168 |
   | 32K | ~150 | ~150 | ~148 | ~150 |

   **长组（102K–131K）**

   | \|C_b\| | Random | LRU | Score | LIM |
   |---|---|---|---|---|
   | 8K | ~475 | ~407 | ~375 | ~497 |
   | 10K | ~395 | ~330 | ~310 | ~412 |
   | 12K | ~338 | ~282 | ~265 | ~350 |
   | 14K | ~300 | ~247 | ~235 | ~305 |
   | 16K | ~268 | ~225 | ~213 | ~270 |
   | 20K | ~222 | ~197 | ~188 | ~222 |
   | 24K | ~195 | ~180 | ~175 | ~195 |
   | 32K | ~172 | ~168 | ~165 | ~170 |

   从图上可以读出的事实：
   - **Score < LRU < Random ≈ LIM，在所有 budget 下都成立。** 他们自己的 LIM 和 Random 几乎重合。
   - Score 相对 LRU 只好 4–8%：8K 时约 7%/8%，16K 时约 4%/5%。
   - Score 相对 LIM 好 18–20%（正文给的数字），这实际上是 Score 对 Random 的差距。
   - 曲线在约 150 和约 165–170 处趋平，可以视为 compulsory miss（新进入 top-k 的 token，以及 4 路 MTP 的新选择），任何策略都消除不了。
   - 正文称"\|C\| 从 8192 翻到 16384，miss 降 43%–49%"，这是对 LIM/Random 曲线说的；对 LRU 来说约降 45%（382→200，407→225）。
   - 正文称"LIM@14336 有 268–350 miss，低于 HiSparse 的 LRU@8192（380–406）"，**这是不同容量之间的比较**。同容量下 LRU（约 222–247）明显优于 LIM（约 268–305）。
2. **Fig.10：LIM 的管理开销**（单 die，MTP3，\|C\|=12288，miss union=300）

   | 序列长度 | bs=1 | bs=16 | bs=24 |
   |---|---|---|---|
   | 64K（LI → LIM） | 101→207 µs（+106） | 371→516（+144） | 517→677（+160） |
   | 128K（LI → LIM） | 142→257（+115） | 628→794（+166） | 911→1142（+232） |

   按 21 个 LI 层 × 约 160 µs 计，每步约 3.4 ms，占约 100 ms ITL 的一小部分。
3. **Fig.11：CopySFA 重叠**（序列 65536，miss union=300）

   | bs | SFA + Copy | CopySFA |
   |---|---|---|
   | 1 | 120 + 17 = 137 µs | 132 |
   | 8 | 231 + 39 = 270 | 236 |
   | 12 | 240 + 54 = 295 | 252 |
   | 16 | 366 + 71 = 436 | 369 |
   | 24 | 499 + 100 = 598 | 500 |
   | 32 | 701 + 130 = 831 | 714 |

   加速 1.03–1.27×；bs ≥ 16 时 OverlapRatio 为 90–98%。
4. **Prefill chunk 消融（Fig.8）**：开 offload 后，同 chunk 大小下 TTFT 变差（每次 forward 都要重新加载 prefix）；但能用更大的 chunk，整体反而更快。只给了定性结论。
5. **端到端（Table 1，90% prefix hit）**：
   - **Prefill（P-only 对 none，输出 1）**：P99 TTFT 降低 7.9%/4.7%/27.4%（64K/128K/256K），input TPS 提高 8.6%/5.0%/37.8%。KV 预分配 36.75→9.04 GB，max-batched-tokens 16384→131072，最大可服务序列约 368K→约 1.6M。
   - **Decode（P+D 对 P-only，输出 8192）**：

     | 输入 | 每 NPU batch | TPOT（ms） | output TPS |
     |---|---|---|---|
     | 64K | 4→12 | 16.9→30.2 | 3787→6358（+67.8%） |
     | 128K | 2→6 | 16.6→26.0 | 1928→3692（+91.5%） |
     | 256K | 1→2 | 16.1→22.9 | 994→1397（+40.5%） |

   - 512K/1M 只有 P+D 能跑：TPOT 24.2/34.1 ms，output TPS 661/469。
6. **Fig.13：ITL 分解（64K）**
   - 同为 BS=4 时，offload 使 ITL 增加 15.2 ms（+23.7%）。其中 LIM +3.0 ms，CopySFA +0.1 ms，"其他 kernel" +8.8 ms。后者来自 MoE dispatch 同步：offload 让各 DP rank 的 miss 分布不同，先完成的 rank 在同步点空等。
   - 生产配置 BS=12：ITL +50.6 ms（+79%）。LIM 11.7 ms，CopySFA 18.5 ms，同步等待 42.5 ms。
7. **没有做**：Bélády/oracle、多模型、多硬件、seeds/方差、按层 miss、不同 k、准确率，以及 policy 在端到端层面的差异（端到端只跑 LIM）。

## 关键结论（带数字）

- "**Capacity is the primary determinant of miss count, eviction policy secondary**"：\|C\| 8K→16K 使 miss 降 43–49%；同为 16K 时，Score 只比 LIM 再低 18–20%，而且随容量增大，策略之间的差距收敛。
- 他们选择用 LIM 这种更弱的策略换取 O(k)+O(\|Miss\|) 的低延迟管理，再靠大 cache 把 miss 压下来。管理开销为 106–232 µs/LI 层，每步约 3 ms。
- 端到端 decode 吞吐最多提升 1.91×（128K），prefill 最多 1.38×。512K/1M 从不可服务变为可服务。所有数字都在约 30 ms TPOT 下给出。
- 隐含的事实：在他们自己的 trace 上，Score 稳定优于 LRU（4–8%），LRU 稳定优于 LIM。

## 对我们的启示

**定位**：这篇论文的核心 claim（容量 >> 策略）直接威胁我们"策略能回收 LRU–Bélády 差距"的叙事，必须正面回应。好在他们的证据有明显漏洞：没有 oracle，比较时容量不同，自己的策略约等于 Random。同时，他们的 Score 策略在自己的 trace 上稳定优于 LRU，可以作为我们方向的第三方佐证。

按优先级排列的补充实验：

1. **用 oracle 把"策略差距"归一化（必须做，也是最强的回应）。** FlashOffloading 只看绝对 miss 数，没有 Bélády，所以"差距收敛"无法区分两种情况：策略真的不重要，还是所有策略都在逼近 compulsory floor。我们要在多个 B 下画出 LRU、score-EMA、Bélády 以及 compulsory floor（只计第一次出现的 page），并报告 G(B) 曲线。如果大 B 时 G 仍然稳定，或者 LRU–Bélády 差距在相对 floor 的意义上不收敛，就是有力的反驳；如果确实收敛，就如实承认，把贡献限定在"HBM 紧张的区间"。
2. **等效容量（必须做）。** 报告"score-EMA@B 的 miss 等于 LRU 在多大 B 下的 miss"（iso-miss 容量节省），以及"LRU 要多少额外 budget 才能达到 score-EMA 的 G"。这能把"策略"翻译成他们认可的"容量"语言。比如 score-EMA@96 ≈ LRU@1xx，相当于 HBM 省了 X%。
3. **把 budget 扫到大 B/K（必须做）。** 他们的 \|C\|/k 是 4–15，我们只有 1.5–2。需要扫 B=64/96/128/192/256/384 等（B/K 从 1 到 6+），说明在他们的工作区间里 score-EMA 的优势还剩多少。如果优势只在小 B 存在，就要论证为什么小 B 是现实区间（端侧/小显存、batch 大时每请求 budget 被压小、HiSparse 推荐 2k）。
4. **复现他们的策略集合。** 补一个"Score（瞬时 score 逐出）"基线，以及近似 LIM 的 threshold + chunk scan 版本，和我们的 score-EMA 区分开。需要说明 EMA 相对瞬时 score 带来多少额外增益，不然审稿人会说"score 逐出 FlashOffloading 已经做过了，只比 LRU 好 4–8%"。我们约 28% 的 G 必须换算成同口径的"相对 LRU 的 miss 降幅"才能比较。
5. **管理开销（建议做）。** 他们把 O(\|C\|) 管理开销作为批评 HiSparse 的主要理由，自己 LIM 的开销是 106–232 µs/层。我们要报告 score-EMA 的更新、选 victim 在 L4 上每层每步的 µs（fused Triton），说明在 K=64 page、B≤128 时开销可以忽略。还要说明如果规模到 \|C\|=16K token，开销会怎样增长（可以用 page 粒度的优势来论证）。
6. **把 bytes 转换成吞吐（建议做）。** 他们的端到端证据链是 miss→H2D→ITL→在 TPOT SLO 下的 batch。我们 batch=1 时延迟打平，需要补 batch>1 或者带宽受限情况下（例如限速 PCIe，或用解析模型）的 TPOT/吞吐。或者明确说明：−9–19% 的 bytes 在 overlap 已经接近 90%+ 时（他们的 CopySFA 数据）主要换来的是功耗和链路余量，而不是延迟。
7. **MTP/多 query 和 token 级粒度（可选）。** 他们的 miss 是 4 路 MTP 的并集，是 token 级。我们是单 query、page 级。讨论里需要说明粒度差异，最好做一个 page size 消融。
8. **方差。** 他们只有 20 条请求的均值，没有 CI。我们 21 篇 held-out 文档 + 预注册是方法学优势，应该显式给出 CI 和逐文档的胜率。

**需要 align 的结论**：
- 在 budget 很大时容量的边际收益更高，这一点要承认，例如引用他们的 8K→16K −45%；
- recency/score 都明显优于 random；
- 管理开销必须和 \|C\| 解耦。

**需要 rebut 的结论**：
- "capacity dominates policy" 来自没有 oracle、容量不对齐的比较。他们的 LIM 等于 Random，自然看不出策略的价值。
- 他们自己的 Score 曲线在所有 budget 下都低于 LRU，说明基于 score 的策略本来就更好。
- 在 HBM 受限的区间（小 B/K、高并发导致每请求 budget 被压小），策略的价值会被放大。

## 局限 / 没读清的地方

- Fig.9 的数字是目测，误差约 ±10。"Score" 是瞬时 score 还是累积 score、LRU 的 recency 是按步还是按 MTP 路计、LIM 的阈值怎么定，正文都没有说清。
- miss 的统计口径：是 4 路去重后的并集，并且按层平均。21 个 indexer 层和 57 个共享层是否都计入、是否包含新 token，都不明确。
- 没有 host 链路带宽数字（PCIe/HCCS），没有 bytes 流量，单模型、单硬件，无准确率，无方差。
- 端到端只比较 offload 开/关，不和 HiSparse 做系统级比较。"HiSparse 典型配置 LRU@8192" 是作者自己的 trace 模拟，不是 HiSparse 的实现。
- venue 标注自相矛盾（MLSys 2025 模板 + 2026 arXiv），正式发表状态未知。
- Fig.12/13 的图上数字没有逐一读取，只用了正文给出的分解数字。
