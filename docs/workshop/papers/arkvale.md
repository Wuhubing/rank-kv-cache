# ArkVale 阅读笔记

> 来源：NeurIPS 2024 proceedings PDF（22 页，含 checklist）。正文、Fig. 3–9、Table 1 都逐段读过，Fig. 4b、8、9 是渲染后读的数。另外读了代码 `pku-liang/ArkVale` 的 `source/arkvale_cpp/src/select.cu`、`source/arkvale/infer_state.py`、`source/arkvale/adapter/modeling.py`（main 分支，2026-11-25 抓取）。OpenReview 讨论页（forum `4oAt5L4lYe`）仍被 bot check 拦住，没读到。

## 基本信息

- **标题**：ArkVale: Efficient Generative LLM Inference with Recallable Key-Value Eviction
- **作者**：Renze Chen, Zhuofeng Wang, Beiquan Cao, Tong Wu, Size Zheng, Xiuhong Li, Xuechao Wei, Shengen Yan, Meng Li, Yun Liang（北大、Infinigence-AI）
- **发表**：NeurIPS 2024
- **代码**：https://github.com/pku-liang/ArkVale（checklist 写的是"暂不开源"，但摘要和现在的仓库都已公开）

## 设置

| 项目 | 内容 |
|---|---|
| 模型 | **只有 LongChat-7b-v1.5-32k**（Llama-2 架构，32 层，h=4096，MHA） |
| 选择单位 | page，大小 p=32（主设置），LongBench 另测了 p=16（ArkVale-16） |
| 预算 | cache budget c ∈ {512, 1024, 2048, 4096} token，即 p=32 时 16/32/64/128 个 page |
| 选择 K | **k = min(C, c/2) / p**，C=1280，即 c=4096 时 k=40 page，c=2048 时 k=32，c≤2560 时 k = c/2p。**B/K = 2（c≤2560）或 3.2（c=4096）** |
| 固定驻留（代码） | `n_sink_pages=2`、`n_win_pages=2`：开头 2 页和最近 2 页始终驻留，不参与 victim 选择；topk 再减去 sink 和 window 页数 |
| 层 | 前两层不压缩（Fig. 3 显示第 0 层稀疏度低） |
| head 粒度（代码） | 默认 `n_groups=1`：**每层所有 head 共享一套排序和一个缓存**（可选分组） |
| 上下文 | LongBench 各数据集（≤32K）；passkey 10K/20K/30K；性能测试用约 10K/20K/30K 的 GovReport 样本 |
| 数据集 | LongBench 6 个：HotpotQA、NarrativeQA、Qasper、GovReport、TriviaQA、PassageRetrieval；passkey retrieval |
| batch | 准确率实验未说明（推测 1）；Fig. 2 用 b=8；latency 用 **batch 4**；throughput 用 40 GB KV 内存上限下能放下的最大 batch |
| 硬件 | Intel Xeon Gold 6348 + **NVIDIA A100 80GB PCIe**；CUDA 12.3、PyTorch 2.3.0、HF Transformers 4.40.0；kernel 用 CUTLASS、FlashInfer、RAFT |

## 方法与基线

### 流程（§5.1，Fig. 5）

1. page 写满后，异步把 K/V 复制到 CPU 作为备份，并在 GPU 上保留 key 的 digest。
2. 每个解码步 attention **之前**：用当前 query 和所有 page 的 digest 估计重要性（无论是否驻留）。
3. 排序，取 top-k 参与 attention。
4. top-k 中不在 GPU 的 page 从 CPU recall，**和排在最后的驻留 page 交换**。

### 替换规则（精确）

- 评分 I(q, K) = max_{k∈K} q·k 的估计值。
- **victim = 本步估计分数最低的 n_recalls 个驻留 page**（不含 sink/window 页）。没有 recency、frequency 或历史状态。
  - 代码 `select.cu`：第一次 radix top-k 在全部 page 上取 top-k；然后 `new_in[i] = in[pos_ids[i]]` 把驻留槽位的**本步分数**拷出来，在 `cap − ns − nw` 个槽位上取 n_recalls 个最小值作为 victim。
  - 因为命中的 page 都在本步 top-k 里，分数一定高于任何未选中的驻留页，所以命中页天然不会被换出。
- 选择和驱逐用**同一个分数**。

### Digest（§5.2）

- **Bounding-sphere**：c = AABB 中心，r ∈ {r_max, r_center, r_mean}，I ≈ q·c + r‖q‖（Eq. 1、2）。
- **Bounding-cuboid**：I ≈ Σ_i max(q_i b_i^max, q_i b_i^min)（Eq. 3）。其中 b = c ± r，r ∈ {max, center, mean}（逐维，Eq. 4）。
  - **cuboid-max 正好就是 Quest 的上界**（b^max = 逐维 max key，b^min = 逐维 min key）。
  - cuboid-mean 的半径是 mean|c − k|，比 max 小，会低估，但召回率最好。
- baseline digest：centroid（key 均值点积）。

### 对比

- 准确率：Origin（全 KV）、StreamingLLM、H₂O、TOVA（都是永久驱逐、不可召回）。为了避免 H₂O/TOVA OOM，用了"两阶段 prefill"：context 用 FlashAttention，question 部分用显式注意力矩阵。
- **没有和 Quest 比**（related work 说 Quest 需要全部 KV 驻留）。
- **没有任何驱逐策略的消融**（没有 LRU、FIFO、历史分数），**没有 oracle**，**没有 miss rate**。

## 评价指标

| 指标 | 怎么测 |
|---|---|
| Recall accuracy（Fig. 7） | 估计 top-k 和真实 top-k（按 page 内最大 softmax）的重叠 \|E_k ∩ R_k\| / \|R_k\|，k=1..40。**衡量 digest 估计质量，不是缓存命中率** |
| 每步 recall 数（Fig. 4b） | 离线模拟：LongBench 每个数据集均匀抽 4 个样本，p=32，扫 cache 容量 c（page）和 k，报告平均每步需要 recall 的 page 数 |
| 任务准确率 | LongBench F1 / Rouge-L / Accuracy（Fig. 8）；passkey 准确率（Table 1，每个长度 20 例，深度 0%–95%） |
| 解码延迟 | Fig. 9a，batch 4，p=32，分解为 Attn / Recall / Select / Estimate / Others |
| 吞吐 | Fig. 9b，40 GB KV 内存上限下的最大 batch 吞吐，相对值 |
| 内存 | Fig. 2 只做背景分析（KV 4blsh 字节） |
| FLOPs、PCIe 字节 | 没有单独报告；只在 §5.1 估算"<10 次 recall ≈ 5 MB，几百微秒" |

## Robustness / 消融

1. **稀疏度（Fig. 3）**：LongBench 上，除第 0 层外，每层不到 10 个 page（320 token）就贡献 99% 以上的注意力分数（稀疏度 >95%）。
2. **重要性漂移（Fig. 4a）**：一个 GovReport 样本（prompt 12470），7 个 page 的排名轨迹。例如 page 256 开始不重要，约 4500 token 之后变得关键。**只是一个样本的示例，没有聚合统计。**
3. **Recall 数网格（Fig. 4b，p=32，平均每步 recall 的 page 数）**，摘录：

   | k \ c | 64 | 96 | 128 |
   |---|---|---|---|
   | 64 | 230.1 | 107.6 | 99.5 |
   | 40 | 20.9 | 7.4 | 5.4 |
   | 32 | 10.2 | 3.4 | 2.2 |
   | 16 | 2.7 | 1.4 | 1.2 |

   其他：k=c 的对角线很大（4→17.2，8→33.8，16→60.0，32→106.0）；k=4、c≥96 时为 1.0。结论是只要 **k < min(40, c/2)，每步 recall < 10**。
   - 注意：k=c=4 时 recall 17.2 > 4，所以这个数字**应该是对层（或 head）求和**，论文没说明。
4. **Digest 估计（Fig. 7）**：centroid 对 top-1 不到 5%，top-4 不到 50%；六种 bounding-volume 都 ≥60%；cuboid-mean 最好，top-1 95%，其余 k 都 >80%。cuboid 好于 sphere，mean 好于 max/center。
5. **Page 大小（Fig. 8）**：ArkVale-16 vs 32 在大多数情况下差别很小，预算紧时 16 更好。
6. **LongBench（Fig. 8）**：c≥2048 时接近 Origin。达到 Origin 水平的预算：HotpotQA 1024、NarrativeQA 2048、Qasper 1024、PassageRetrieval 2048、TriviaQA 512。GovReport 在 4096 时约 29 vs Origin 30.6。
7. **Passkey（Table 1）**：ArkVale 在所有长度和预算下 95–100%；StreamingLLM、H₂O、TOVA 在 512–4096 预算下 0–40%。
8. **延迟（Fig. 9a）**：最多 2.2×，平均 1.7×。开销主要来自 Estimate（随 seq_len/p 线性增长），Recall 很小。按图读：10K 时约 22–25 ms vs Full 约 31 ms；30K 时约 23–27 ms vs Full 约 55 ms。
9. **吞吐（Fig. 9b）**：最多 4.6×，平均 3.5×，原因是延迟降低加上单样本 KV 内存变小、batch 可以更大。

## 关键结论

- token/page 重要性会随解码步漂移，永久驱逐（H₂O、TOVA、StreamingLLM）会丢掉以后需要的信息。CPU 备份加按 digest 召回能解决这个问题：passkey 95–100% vs 基线 ≤40%。
- 2K–4K 预算下 LongBench 准确率接近全 KV；解码延迟 1.7×（最多 2.2×），吞吐 3.5×（最多 4.6×）。
- 每步 recall 的开销（k < c/2 时 <10 页）远小于解码步的总时间。
- cuboid-mean digest 估计 top-k 最准。

## 对我们的启示

### 需要补的实验（按优先级）

1. **[P0] 把我们的 "current score" 行明确标成 ArkVale 规则，并按 ArkVale 的配置跑一次。**
   - 我们的 `ScoreCurrent`（held-out B=96 时 G 0.246，B=128 时 0.224）用的就是本步 Quest 上界分数，等价于 **ArkVale + cuboid-max digest**。
   - 要和 ArkVale 完全对齐，还需要：
     - (a) 固定 2 个 sink 页和 2 个最近页，不参与驱逐（按 32-token 页换算成我们的 16-token 页，即 4+4 页，两种都跑）。
     - (b) 前两层不受预算限制（或者报告去掉前两层后的结果）。
     - (c) **B = 2K 的配置**：我们的 B=128、K=64 正好是 ArkVale 的 c/2 规则，应该作为主对比点。
   - 审稿人看到 "current score" 会问"这就是 ArkVale 吗"，这里要能直接回答"是的，在 ArkVale 的配置下我们的 EMA 比它多关闭 X 的 gap"。
2. **[P0] 报告 ArkVale 的 Fig. 4b 式指标：每步每层平均 recall 页数和字节数**，在 (K, B) 网格上画热力图，并加上 LRU、EMA、Bélády 的对比。
   - ArkVale 用"每步 recall 数"论证开销可忽略；我们用 miss rate 和 G。两种都给，审稿人才能把我们的数字放进 ArkVale 的框架。
   - 字节数要按我们的配置换算：Qwen3-0.6B 每层 8 KV head × 128 dim × 16 token × 2(K,V) × 2 B = 64 KB/页。
3. **[P1] cuboid-mean digest 作为替代分数。** ArkVale 表明 cuboid-mean 估计比 max（= Quest）准得多（top-1 95%）。
   - 问题："如果用更准的分数，score EMA 相对 current score 的优势还在吗？"
   - 需要 capture 每页逐维的 center 和 mean|c−k|（或者直接存 key，离线算），再生成 `scores_cuboid_mean [T,L,P]`。
   - 注意：选择仍然用 Quest bound（保持选择固定），只把驱逐分数换成 cuboid-mean。另做一组"选择和驱逐都用 cuboid-mean"。
4. **[P1] 吞吐和 batch 扫描。** ArkVale 的系统卖点是吞吐 3.5×，来自更小的 GPU KV 占用和更大的 batch。我们只有 batch 1 延迟（在 L4 上和 LRU 打平）。
   - 至少补 batch ∈ {1, 4, 8, 16} 的延迟和吞吐，以及"GPU KV 内存固定时能跑的最大 batch"。
   - 应该指出：PCIe 字节减少 9–19% 的收益在 batch 大、PCIe 争用时才可能体现。
5. **[P1] 准确率等价性声明。** ArkVale 的主要指标是任务准确率。我们的方法保持选择固定、每个选中页都会被加载，所以 attention 输出和 Quest-style selector 完全一样。
   - 要在论文里写清楚，并给一个 sanity check：在若干 LongBench 样本上对比 LRU 与 EMA 的生成 token 完全一致（或 logits 差为 0）。
   - 如果有余力，报告 Quest-style selector 本身相对全 KV 的准确率损失（K=64×16=1024 token），让审稿人知道这个 selector 的质量水平，对应 ArkVale 的 Fig. 8。
6. **[P2] 第二个模型和 MHA 模型。** ArkVale 用 LongChat-7B（MHA）。我们只有 Qwen3-0.6B（GQA）。补一个 7B 级模型的 trace（最好就是 LongChat-7b-v1.5-32k，可以直接对标）。
7. **[P2] 漂移的聚合统计。** ArkVale 的 Fig. 4a 只是一个样本。我们可以给"被驱逐页在 h 步内重新被选中的比例"，按层和距离分组。这是 ArkVale 没做、对我们 H1 也有用的数据。

### 在 trace 模拟器里复现 ArkVale 策略

- **现有 npz v2 已经够用**：`scores [T,L,P]` 给出每页本步 Quest 分数，等于 cuboid-max digest 分数。
- **策略**（`kvpolicy.py` 里的 `ScoreCurrent`，加两个选项）：
  - `pinned = {前 ns_pages 个 page} ∪ {最近 nw_pages 个 page}`，每步都强制驻留，并从候选里排除（它们也占 B 的槽位）。
  - victim = 未选中、未固定的驻留页中 `scores[t, l, p]` 最小者；tie 按 LRU。
  - 配置：K = min(1280/16, B/2) 对应 ArkVale 规则；另跑我们的标准 K=64。
  - 前两层：选项 `unbounded_layers=2`（这两层不计入 miss，或者 B=∞）。
- **需要新增的 trace 字段**：
  - `scores_cuboid_mean [T,L,P]`（以及可选的 `sphere_mean`），由 capture 时保存每页 center c 和 r_mean 后计算；或者保存每页逐维 mean|c−k|（每页 2×128 个 float16，0.6B 可以接受）。
  - `true_page_max_attn [T,L,P]`（每页内最大 softmax 概率，即 ArkVale 的真实排序），用于复现 Fig. 7 的 recall accuracy，并检查 Quest bound 在 Qwen3 上的估计质量。需要全注意力，只对 dev 子集 capture 即可。
- **Recall 计数**：模拟器已有 misses；按 (t, l) 汇总 misses 就是 ArkVale 的"每步 recall 数"。字节 = misses × 64 KB。

### 定位上的注意

- ArkVale 的 §2 批评"依赖历史数据驱逐"的方法会丢掉未来重要的 token。我们的结果正好相反：**在可召回缓存里加上历史（EMA）比只看当前分数多关闭约 0.035 的 gap（held-out B=96：0.281 vs 0.246），B=256 时差距更大（0.242 vs 0.145）**。这是可以明确写出的、对 ArkVale 设计选择的补充。但要说明区别：ArkVale 批评的是永久驱逐里的历史分数，不是可召回缓存里的历史分数。
- 不能说 ArkVale 测过 miss rate 或做过驱逐消融，它没有。
- Quest bound = cuboid-max digest，这一点要在论文里点明，避免审稿人以为我们用的是不同的估计器。

## 局限/没读清的地方

- Fig. 4b 的 recall 数是对什么求和（层？head？batch？）没说明；k=c 时 recall 数大于 k，说明不是单层单 head。
- 准确率实验的 batch size、生成长度、两阶段 prefill 对 ArkVale 自己是否也适用，没交代。
- "Select" 开销里是否包含 top-k 的两次 radix select，论文只给堆叠柱状图，没给数值。
- 代码里有跨层 prefetch（用当前层的 hidden state 投影出后面某层的 query 提前选页并 recall，`n_prefetch_layers`），论文没提。这会影响延迟数字，但不影响替换规则。
- OpenReview 评审意见没读到（bot check），不知道审稿人是否要求过驱逐策略消融或 oracle。
- Fig. 8 只有曲线，没有表格；上面的数字是按图读的近似值。
