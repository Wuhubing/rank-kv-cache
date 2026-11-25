# 在 wulab2 上从头重跑：目标、指标、基线和执行步骤

写于 2026-11-25。**本文是交给 mac mini 上的 Claude 自主执行的完整任务说明**（用 `/goal` 盯到完成），执行过程中不需要人工审查。配套文档：[README.md](README.md)（投稿准备）和 [papers/](papers/)（论文阅读笔记）。

本计划按依赖顺序写成一条连续的流水线，目标是一口气做完。每一步都写明输入、做法、输出和**检查条件**：检查不通过就按第 10 节的规则修复后重试，修不好就停下并写明原因，不往下走。

---

## 1. 我们的目标和 thesis

### 1.1 现有材料里写明的目标
| 来源 | 写的是什么 |
|---|---|
| `docs/efficient_policy_dev.md`、`splits/heldout_frozen_v1.json` | 第二条线：分数 EMA 能否在长文档上缩小 LRU 与 Bélády 之间的差距（G），并有 3 条预注册判据 |
| STATUS_REPORT §1 | 提到 proposal 和"蓝图"（A 文献 → B 信号 → C 策略 → D 系统，每阶段有决策关口），但**这两份文件不在仓库里，本机也没找到** |

### 1.2 建议的 thesis（需要你和团队确认）
现有结果实际上回答的是一个比原题更大的问题。建议论文的 thesis 定为：

> **在"稀疏注意力 + KV 卸载 + GPU 页缓存"中，选择分数随时间的延续性决定了替换质量。** 具体有两点：
> 1. 使用所有 page 的分数历史（EMA）能稳定地缩小 LRU 与 Bélády 之间的差距；按层分配预算能进一步减少 miss。
> 2. 在常见硬件上，这些 miss 和字节的减少不一定变成延迟的减少；我们刻画了什么条件下会变。

### 1.3 现有信息够不够
**不够**，以下几项必须先补清楚，否则论文的主张会站不住或被审稿人轻易驳回：

| 缺口 | 为什么重要 | 怎么补 |
|---|---|---|
| **EMA 和已有方法的差距很小** | 现有 dev 结果中，EMA 的 G=0.279，"只看当前分数"（ArkVale 式）是 0.244，InfLLM 式衰减和是 0.269，线性回归是 0.300。"我们提出了更好的策略"这个说法很弱 | thesis 以第 1.2 节的"刻画"为主，不以"新策略最好"为主；用全 page 百分位对照、信号消融和预算扫描说明分数历史为什么有效 |
| 全 page 百分位对照还没结果 | 决定第 1.2 节第 1 点该说"分数大小"还是"全局排序" | 执行步骤 S4 |
| 按层分配缺少相同分配的 oracle | 不然不能说"补上了多少差距" | 执行步骤 S4 |
| 只有 1 个模型、1 种 GPU、batch 1 | workshop 审稿人会质疑泛化 | 执行步骤 S3、S5、S6 |
| proposal 和蓝图原文不在仓库 | 无法逐条核对最初承诺的目标和成功标准 | 请把 proposal 放进 `docs/`；没有的话，以 `splits/heldout_frozen_v1.json` 和 `docs/efficient_policy_dev.md` 为准 |
| 投稿目标和截止时间未定 | 影响篇幅、要不要做系统部分 | 团队决定；ICBINB 2027 届的消息在 2026-12-07 后才有 |
| 团队对 thesis 的共识 | 两条研究线由不同的人推进过 | 团队过一遍第 1.2 节 |

---

## 2. 不能违反的规则
1. **冻结的东西不改**：`splits/heldout_frozen_v1.json`、`splits/longctx_v1.json`、已提交的结果文件都保持原样（其他已冻结文件也一样不动）。新结果只写进 `results_wulab2/`。
2. **原有 held-out 只再评估一次**：用冻结参数在新 GPU 上重新采集，按事先提交的复现预注册评估一次。不在新的 held-out 上调任何参数。
3. **探索只在 dev 上做**：全 page 百分位对照、预算扫描、各种消融都只用 dev。
4. **新模型或新任务的确认性结论要用新的 held-out**：另抽文档，先写预注册再采集。
5. **一个分支只让一台机器写**：`rerun/wulab2` 只从 mini 提交和推送（wulab2 上没有 GitHub 凭据）。wulab2 只负责跑，代码通过 `git pull` 从 GitHub 同步过去（仓库是公开的），小结果文件用 `rsync` 拉回 mini 再提交。
6. **不备份到 Drive**：原始 trace 只放在 `/data/weibing`；仓库里只提交小的结果文件和带 sha256 的 manifest。
7. **只用一张卡**：开跑前用 `nvidia-smi` 挑一张空闲的卡，用 `CUDA_VISIBLE_DEVICES` 固定。全部采集和 benchmark 都在这张卡上做，卡号和 GPU UUID 写进 manifest。长任务放在 tmux 里跑。

## 3. 机器情况（2026-11-25 实测）

| 项目 | 情况 | 影响 |
|---|---|---|
| GPU | 4 张 RTX 6000 Ada（每张 48 GB），只用一张 | 单卡显存够 8B 模型加 64K 上下文 |
| 驱动 | 550.90，最高 CUDA 12.4 | 原来用的 torch 2.14.1+cu130 不能用，要换 cu124 的 torch |
| 根分区 | 只剩 127 GB | 家目录不放大文件 |
| `/data` | 剩 9.9 TB，已有 `/data/weibing` | 模型、HF 缓存、trace、结果都放这里 |
| CPU / 内存 | 48 核，503 GB | CPU 回放开多进程 |
| 网络 | 能访问 Hugging Face 和 GitHub | — |

## 4. 已定的决定（执行时直接照做）

| # | 决定 | 内容 |
|---|---|---|
| D1 | 原 held-out | 冻结参数做跨硬件复现，评估一次（S5） |
| D2 | 代码基线 | 不等 PR 合并。从 `origin/review-fixes/efficient-policy`（PR #7，已包含 #5）开 `rerun/wulab2`。不合并、不关闭任何 PR |
| D3 | 第二个模型 | Qwen3-8B |
| D4 | thesis | 按第 1.2 节执行；S4 第 2 项出结果后，按结果把第 1.2 节第 1 点改成"分数大小"或"全局排序"，并在 `docs/rerun/thesis.md` 写明依据 |
| D5 | 第 1.3 节的缺口 | proposal 原文和投稿目标暂时缺失，不阻塞执行；以 `splits/heldout_frozen_v1.json` 和 `docs/efficient_policy_dev.md` 为准 |

---

## 5. 要报告的指标

根据读过的论文整理（来源见括号，笔记在 `papers/`）。

### 5.1 主指标：替换质量（全部按文档和层汇总后再相除）
| 指标 | 定义 | 为什么要 |
|---|---|---|
| miss rate、repeat miss rate | miss / 访问；repeat 只算之前被选过的 page | 所有同类工作的基本指标（HiSparse、InfLLM、FlashOffloading） |
| compulsory miss 占比 | 首次选中的 miss，任何策略都避免不了 | 说明还剩多少可改进空间 |
| **G** | (LRU − 策略) / (LRU − Bélády)，按 repeat miss 计算 | 与 PARROT 的归一化命中率相同；HiSparse 也用 Bélády 作参照 |
| 相对 LRU 的 repeat miss 降幅 | 同样总槽位下 | 按层分配时唯一能直接比较的数字 |
| **miss–预算曲线** | B/K 从 1 到 8 | 回应 FlashOffloading 的"容量比策略重要"；HiSparse 显示缓存翻倍超过整个差距 |
| **等 miss 容量** | LRU 要多少槽位才能达到策略的 miss | 把收益换算成"省了多少显存"，方便和容量比较 |
| 每层 miss 和 G | 28 层分别报告 | ECHO 显示各层命中率差异大（0.88–0.99），也是按层分配的依据 |

### 5.2 预测质量（机制分析）
| 指标 | 定义 | 来源 |
|---|---|---|
| 预测集重叠 | EMA 预测的前 K 个 page 与下一步真实 top-K 的重叠，集合大小固定为 K | PRR（它的 98% 允许集合大于 K，要说明不可直接比） |
| 相邻步选择重叠 | 第 t 步和第 t+1 步 top-K 的交集比例 | PRR、Scissorhands、AVSG |
| top-K 流失率 | 1 − 相邻步 top-K 的 Jaccard | InertiaKV |
| 增量 AUC 或分层效应 | 控制 recency 和 frequency 后，分数信号对 h 步内复用的预测力 | 文献中没有人做过 |
| Quest 分数与真实注意力的 Spearman ρ | 每层每步 | InertiaKV 的论证需要这一项 |

### 5.3 系统指标（每生成一个 token）
| 指标 | 说明 | 来源 |
|---|---|---|
| PCIe 字节/token | miss × 每页每层 64 KiB（0.6B） | 我们现有指标 |
| TPOT（ms/token）中位数和分布 | 去掉 warmup，多次重复，报告方差 | ECHO、Rethinking KV（它们都没报方差，我们要报） |
| 延迟分解 | 选择器 / host 取数 / 注意力 / 其余 | Rethinking KV 指出 HF 里测的加速不可靠，必须拆开 |
| 策略开销 | 每层 µs；关掉 host 取数的对照 | HiSparse 的 no-IO oracle |
| FLOPs/MACs 开销表 | 选择器、注意力、模型线性层、EMA 更新、驱逐选择 | 见 README 第 2.3 节 |
| 显存 | 分阶段：prefill、迁移、decode warmup、稳态 decode；策略元数据大小 | PR #4 的 F7 |
| 生成一致性 | 不同替换策略下输出逐 token 相同 | ArkVale、ECHO 都强调无损 |

### 5.4 统计
- 按文档的 bootstrap（2000 次），同一批重采样文档用于所有策略，算配对差。
- 每篇文档的 G 分布图和胜过 LRU 的篇数。
- 噪声注入：给分数加噪声，看 G 是否退回 LRU 而不是变得更差。
- 读过的论文几乎都只用一条 trace 或不报方差，这一项是我们的优势，要做足。

---

## 6. 要实现的基线

`kvpolicy.py` 已经实现了不少，dev 上的现有结果（B=96）列在下表。**新采集的 trace 上，所有基线都要重跑一遍。**

### 6.1 已经实现的
| 基线 | 规则 | 现有 dev G | 来源 |
|---|---|---|---|
| LRU（hit 优先） | 同一步内 hit 排在新 fetch 之前，再按 page ID | 0 | HiSparse 的 LRU 就是这个规则 |
| FIFO | 按准入时间 | −1.00 | HiSparse、FlashOffloading |
| LFU、LRU-2、ARC | 经典策略 | −0.84、−0.05、−0.01 | 经典缓存文献 |
| 当前分数（ArkVale 式） | 驱逐本步分数最低的驻留 page | 0.244 | ArkVale；Quest 上界正好等于 ArkVale 的 cuboid-max digest |
| InfLLM 式衰减和 | 所有驻留 page，softmax(分数) 代理注意力质量，d 由 LODO 选 | 0.269（d=0.1 时 0.264） | InfLLM App. A |
| 衰减的选中次数 | 不用分数，d=0.8 | 0.044 | MoE 缓存类做法 |
| 线性 next-use 回归（DAgger） | 学习的上限参考 | 0.300 | 我们自己的 |
| 分数 EMA（z） | e ← αe + (1−α)z | 0.279 | 我们的主策略 |
| 全 page 百分位 EMA（allrank） | 每步所有 page 的百分位排名做 EMA | 待跑 | PR #7 |
| 按层分配 + LRU / + EMA | 每层不同的 B，总槽位相同 | 降幅 5.3% / ~16% | 我们的 |
| Bélády | 最远的下一次使用 | 1 | 上限 |

### 6.2 需要新增的
| 基线 | 精确规则 | 来源 | 需要的 trace 字段 |
|---|---|---|---|
| Random | 在未选中的驻留 page 里随机驱逐，报告多个 seed | HiSparse、FlashOffloading | 现有 |
| ArkVale 完整版 | 当前分数驱逐，额外固定 2 个 sink page 和 2 个最近 page | ArkVale 代码 `select.cu` | 现有 |
| InfLLM 公开代码版 `lru-s` | 被选中 page 的分数重置为本步注意力质量，未选中的驻留 page 乘以衰减 d | InfLLM 代码（与论文 Eq. 1 不同，两个都跑） | 真实注意力质量 `attn_mass`（新字段）；没有时用 softmax 代理，并标明 |
| InfLLM 论文版 Eq. 1 | s ← d·s + 注意力质量，d=0.1 | InfLLM 论文 | 同上 |
| Score（FlashOffloading 式） | 驱逐瞬时 indexer 分数最低的 | FlashOffloading Fig. 9 | 现有（等同于当前分数，作为对照写明） |
| SIEVE、S3-FIFO | 原论文规则 | NSDI 2024、SOSP 2023 | 现有 |
| CO2 式析因 | 衰减 α × FIFO 保护区比例 ρ ∈ {0, 0.25, 0.5} × 准入后最短驻留 N 步 | CO2（它的收益几乎全来自 ρ） | 现有 |
| Lazy 刷新 / 冻结排名 | 每 r 步更新一次 EMA（r ∈ {2, 4, 8}）；只用第一步排名 | InertiaKV | 现有 |
| Holt 双指数平滑 | level + trend，α、β、γ 网格（与 PRR 相同） | PRR（注意它的 α 是新观测权重，与我们相反） | 现有 |
| 幂律年龄惩罚 | 替代指数衰减 | NoHairCache | 现有 |
| 同分配 Bélády 和同分配 LRU | 在按层分配后的预算下分别回放 | PR #7 `validation/alloc_oracle.py` | 现有 |

**关于对照组的写法**：EMA 的 α=0 等于"当前分数"，α→1 接近累积频率。论文里应该画一条 α 从 0 到 1 的连续曲线，把 ArkVale 式和频率式放在两端，而不是把它们当成互不相关的对手。

---

## 7. 执行步骤（按依赖顺序，一口气做完）

### S0：准备分支
1. 在 mini 的 `~/work/rank-kv-cache` 克隆或更新仓库，按 D2 建 `rerun/wulab2` 并推送。
2. 把本文和 `docs/workshop/` 复制进这个分支（来自 `docs/workshop-readiness`），建立 `docs/rerun/STATUS.md`（见第 10 节格式）。
- 检查：分支已推送；本地单元测试在 mini 上通过（mini 只有 Python 3.9，用 `uv` 建 3.10 以上的环境）。

### S1：环境
1. 在 wulab2 的 `/data/weibing/projects/rank-kv-cache` 用 https 克隆仓库，切到 `rerun/wulab2`。
2. 设置 `HF_HOME=/data/weibing/hf` 和 `TMPDIR=/data/weibing/tmp`。
3. 用 uv 建环境：`transformers==4.53.3`，cu124 的 torch（版本写进 `docs/rerun/environment.md`），numpy、pytest、`rouge==1.0.1`。
4. 按 split 文件里固定的 revision 下载 Qwen3-0.6B 和 LongBench；后面 S6 再下 Qwen3-8B。
5. 选卡：`nvidia-smi` 确认空闲，固定 `CUDA_VISIBLE_DEVICES`，记录 UUID。
- 检查：全部单元测试通过（PR #7 上是 155 个通过、1 个跳过）；选择器自检（与暴力计算比对）通过。

### S2：采集脚本
1. 把 `notebooks/long_trace_capture.py`（长文档，schema v2）改成 `collect/long.py`。选择器、prompt 模板、停止规则、精度逻辑原样搬，不改算法。
2. 参数：`--split`、`--subset dev|heldout`、`--docs`（只跑指定文档）、`--out`、`--dense-reference`。
3. 新增可选字段（默认关闭，S6 打开）：`--record-attn-mass`（每个选中 page 的 softmax 后注意力质量，按 head 求和）和 `--record-true-attn-rank`（每层每步分数与真实注意力的 Spearman ρ）。打开这些字段不能改变选中的 page，用测试保证。
4. manifest 记录：GPU 型号和 UUID、驱动、torch、transformers、模型和数据的 revision、prompt token 的哈希、生成的 token、trace 的 sha256。
5. 按文档断点续跑；先写临时文件再改名；进度写进状态文件。
6. 采集后立刻用 #7 的严格读取器校验格式。
- 检查：采 1 篇；同一篇在同一张卡上采两次，trace 逐字节一致；整条链路"采集 → 校验 → 回放 → 出表"跑通；记录单篇耗时。

### S3：重采 dev
1. 14 篇 dev（`splits/longctx_v1.json`），最多 1024 步，thinking 开启。
2. 与原 L4 结果对比：LRU/Bélády 差距、各策略的 G，是否落在原来的 CI 附近。如果能拿到 L4 原始 trace，再比较逐层逐步的选择一致率，以及翻转的 page 是否集中在 top-K 边界。
- 输出：`results_wulab2/dev/`，`docs/rerun/s3_dev_vs_l4.md`。
- 检查：所有文档完整、校验通过；对比小结写好。如果和 L4 差异很大，先停下分析原因，再往下走。

### S4：dev 上的全部基线和分析（CPU 回放）
用 S3 的新 trace，48 核并行。
1. 第 6.1 节全部已实现基线重跑，加上第 6.2 节全部新基线（需要 `attn_mass` 的 InfLLM 两个版本先用 softmax 代理，S6 再用真实值补）。
2. 全 page 百分位对照：z 和 allrank，α 网格和 LODO 选参相同。
3. 同分配 Bélády 和同分配 LRU。
4. 预算扫描 B/K ∈ {1, 1.25, 1.5, 2, 3, 4, 6, 8}，所有策略，加等 miss 容量。
5. α 从 0 到 1 的连续曲线，信号 ∈ {raw, z, allrank}。
6. CO2 式析因、lazy 刷新、Holt、幂律衰减。
7. 第 5.2 节的预测质量指标。
8. 第 5.4 节的统计和噪声注入。
- 输出：`results_wulab2/dev/<实验名>/`，每项一页结论放在 `docs/rerun/s4/`。
- 检查：所有回放的 LRU 和 Bélády 计数与独立实现一致（`validation/audit.py`）；第 2 项的结论写进 thesis（"分数大小"还是"全局排序"）。

### S5：跨硬件复现（先提交预注册，再采 held-out）
1. 写 `docs/replication_wulab2.md` 并**提交、推送后**才能进行下一步。内容：
   - 所有参数原样沿用两份冻结配置，不允许用 S3–S4 的结果改动。
   - 判定标准：EMA 的 G 及其 CI、21 篇中胜过 LRU 的篇数、按层分配后的降幅（同时报告同分配 oracle 的版本）。
   - 评估命令、偏离的定义、复现失败时报告什么。
2. 采 21 篇长 held-out。
3. 按预注册的命令各跑一次。
- 输出：`results_wulab2/heldout/`，`docs/rerun/s5_replication.md`（与 L4 结果并排对比，列出全部偏离）。
- 检查：预注册的提交时间早于第一篇 held-out 的采集时间（看 git 记录和 manifest 时间戳）。

### S6：扩展（第二个模型、长上下文、更多任务）
1. 从 LongBench 和 RULER 为新设置另抽文档，写新的 split 文件（dev 和 held-out 分开，按源文档划分）。
2. 采集时打开 `--record-attn-mass` 和 `--record-true-attn-rank`。
3. Qwen3-8B 在原长文档任务上采 dev。64K 上下文下权重约 16 GB、KV 约 9.4 GB，单卡放得下。
4. 0.6B 上加一档 64K 上下文，以及 SCBench 多轮。
5. page 大小 ∈ {8, 16, 32, 64}、K ∈ {32, 64, 128} 的扫描（0.6B，dev）。
6. 采样解码：temperature > 0，3 个 seed（0.6B，dev）。
7. 用真实注意力质量重跑两个 InfLLM 基线；报告 Quest 分数与真实注意力的秩相关。
8. 如果要对新设置下确认性结论：先提交预注册，再采对应的 held-out，评估一次。
- 输出：`results_wulab2/ext/`，`docs/rerun/s6_extensions.md`。
- 检查：每个新设置都有 dev 结论；确认性结论都有预注册在先。

### S7：系统测量
1. 用 #7 修好的 `gpu_bench.py`（warmup 生效，显存按阶段报告）在同一张卡上测，测之前确认卡上没有别人的任务。
2. 策略：LRU、当前分数、EMA、按层分配 + EMA、Bélády（用 trace 的未来信息）。
3. 报告第 5.3 节的全部指标：TPOT 分布（多次重复）、延迟分解、策略开销 µs、关掉 host 取数的对照、显存分阶段、PCIe 字节、FLOPs/MACs 表的实测部分。
4. 验证所有策略的输出逐 token 一致。
5. 刻画收益区间：B/K、上下文长度、0.6B 与 8B；能实现的话再加 batch > 1。
- 输出：`results_wulab2/gpu/`，`docs/rerun/s7_systems.md`。
- 检查：LRU 和 Bélády 的实测 miss 与模拟器完全一致。

### S8：汇总
1. 更新 README、STATUS_REPORT 和 workshop README 中的数字和图，原 L4 结果保留并并排展示。
2. 按第 1.2 节（或团队修改后）的 thesis 逐条核对：每一条都有对应的结果和图，没有证据的说法删掉或改弱。
3. 开 PR（`rerun/wulab2` → `efficient-policy`），说明做了什么、结果是什么、和原结果的差异、所有偏离。不合并。

---

## 8. 风险
| 风险 | 应对 |
|---|---|
| 根分区满了 | 所有路径指向 `/data/weibing`；启动前检查 `HF_HOME` 和 `TMPDIR` |
| 选定的卡被别人占用 | 每次开跑前检查；被打断可按文档续跑。必须换卡时记录新 UUID，并在新卡上重跑一篇已有文档，确认 trace 一致 |
| cu124 的 torch 与原来数值行为不同 | 本来就是新硬件；版本写进 manifest，复现报告里说明 |
| 分数存为 float16，平局更多 | 照原样保留以保证可比；复现报告里分析平局比例 |
| 新 trace 上 EMA 的优势消失 | 这本身是一个结论，按复现预注册如实报告；thesis 相应调整 |
| 原始 trace 只有一份 | manifest 记录 sha256；丢失后按同一配置重采 |

## 9. 交付物
- `rerun/wulab2` 分支和对应的 PR：采集脚本、新基线和分析脚本、`results_wulab2/` 下的小结果文件。
- `docs/rerun/`：决定记录、环境、每一步的结论、复现报告、扩展和系统测量报告。
- `docs/replication_wulab2.md`：复现预注册（S5 之前提交）。
- 更新后的 README、STATUS_REPORT 和 workshop 文档。

---

## 10. 给 mini 上的 Claude 的执行说明

### 10.1 连接和分工
| 机器 | 做什么 | 怎么连 |
|---|---|---|
| mini（`~/work/rank-kv-cache`） | 写代码、提交、推送；调度和检查 | 本机 |
| wulab2（`/data/weibing/projects/rank-kv-cache`） | 采集 trace、CPU 回放、GPU 测量 | 在 mini 上 `ssh wulab2`（已配好免密） |

- wulab2：Ubuntu，Python 3.10，`uv` 在 `~/.local/bin`，tmux 3.2a，**没有 gh、没有 git 身份和凭据**。
- 原始 trace 放在 `/data/weibing/projects/rank-kv-cache/data/`（加进 `.gitignore`），**不提交、不上传 Drive**。
- 代码同步：mini 推送后，在 wulab2 上 `git pull`。结果同步：`rsync -a wulab2:/data/weibing/projects/rank-kv-cache/results_wulab2/ ~/work/rank-kv-cache/results_wulab2/`，只同步 CSV、JSON、TXT、PNG 等小文件。

### 10.2 必须遵守
1. 第 2 节的规则全部有效。特别是：不改冻结文件；held-out 只在 S5 预注册提交之后采集、只评估一次；探索只用 dev。
2. 不合并、不关闭任何 PR；不改 `main`、`efficient-policy` 等已有分支。
3. 每完成一个有意义的单元就提交并推送，提交信息写清楚做了什么。
4. wulab2 是共享服务器：只用一张卡；不 kill 别人的进程；不往根分区写大文件。
5. 等待长任务时不要反复调用模型：用 tmux 跑，进度写进文件，定时用普通命令检查。
6. 指南里记录过的坑（`mac-mini-guide.md` 第 11 节）：远程命令在远程机器上生成，不要在 mini 上展开 `~` 再发过去；`while read` 循环里的 `ssh` 要加 `-n`；实验进程要脱离调度脚本独立运行；macOS 没有 `timeout`。

### 10.3 检查不通过时
1. 先定位原因并修复，重新执行这一步；同一个检查最多重试 3 次。
2. 修复不能违反第 2 节。如果唯一的修法会违反（例如需要改冻结文件、需要重新评估 held-out），就**停下**。
3. 停下时在 `docs/rerun/STATUS.md` 写 `BLOCKED`、原因、已经试过的办法和需要人决定的问题，提交推送后结束。
4. 结果不符合预期（例如 EMA 在新 trace 上不再胜过 LRU）**不算**检查失败，照实记录并继续。

### 10.4 进度文件 `docs/rerun/STATUS.md`
每完成一步更新一次并推送：

```
当前步骤: S3
状态: RUNNING | DONE | BLOCKED
使用的 GPU: 卡号 + UUID
已完成: S0 (commit abc123), S1 (commit def456), ...
最近一次检查: 通过/不通过 + 一句话
下一步: ...
BLOCKED 原因（如有）: ...
```

### 10.5 停止开关
如果 wulab2 上存在 `/data/weibing/projects/rank-kv-cache/STOP`，不要开始新的采集或回放任务，正在跑的任务让它跑完；在 STATUS.md 写明"按停止开关暂停"。

### 10.6 完成的定义
- S0–S8 全部是 DONE，或者停在 BLOCKED 并写明原因。
- S8 的 PR 已开，PR 描述里逐条对照第 1.2 节的 thesis 写明哪些被证据支持、哪些被改弱或删除。
