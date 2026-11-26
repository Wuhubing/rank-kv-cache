# KV cache 项目独立复验任务书

**给执行 agent：请代表一名复验者，亲自完成下列全部步骤，并交回一份带证据的测试报告。** 每位复验者使用自己的输出目录和自己的 L4 会话。先从输入生成结果，再打开归档作对照。文末数值是验收参考，不能用来生成输出。CPU 回放、归档重算、重新采集和 GPU 实测分别记录；某一步受阻时继续所有可独立完成的步骤。

要检验三个结论：历史 rank 能预测复用，但完整 rank 策略未达到 10%；当前 Quest 分数使重复 miss 降幅跨过 10%，EMA 有额外贡献；少搬页在 L4 单层路径上带来了较小的实测耗时收益。最终分别判断每个结论。

## 1. 准备输入与环境

从本分支根目录开始。四本 Colab notebook 在 [notebooks/](notebooks/)，五个冻结 ZIP 在 [reproduction/archives/](reproduction/archives/)，[原 proposal](proposal/6_5940_Proposal.pdf)、[报告正文和主图](docs/final_report_v1/) 均随仓库提供。使用 Git LFS 获取 ZIP 的实际内容。从下面的命令开始，保留本次检出的 commit ID。

```bash
git clone --branch xiaoyang/solo-project --single-branch https://github.com/wxy20050403/rank-kv-cache.git
cd rank-kv-cache
git rev-parse HEAD
git lfs pull --include="reproduction/archives/*.zip"
python3 reproduction/prepare_inputs.py
```

已检出本分支的 agent 从 `git rev-parse HEAD` 开始。prepare_inputs.py 逐包检查 SHA-256，并从交接 ZIP 中恢复第 2 节需要的三个短 trace ZIP 到 data/short/。四个 L4 结果包按第 3、6 节在本次复验输出目录中展开。两本硬件 notebook 在 Colab 的上传框使用 reproduction/archives/ 中的原始 ZIP。长 trace 来自项目原始共享目录：[14 篇 dev](https://drive.google.com/drive/folders/1ZwuV8ksLYh7yZYEBxQKzJc0_uW2F-Npu)、[21 篇 held-out](https://drive.google.com/drive/folders/1oBfSLtI3JOSr8CTIgIPLdEHyiYmLBmPl)。下载到两个独立目录。用 splits/longctx_v1.json 核对清单，分别设置 LONG_DEV 和 LONG_HELD。报告中记录每个输入的哈希、环境、命令和运行状态。新 Colab 结果包保存到单独目录。如 agent 无法启动 L4，由操作者运行 notebook 并交回新包，等待期间继续 CPU 核查。

以下命令在 Git 分支根目录执行。使用 Python 3.13 或与包内依赖兼容的环境。

```bash
python3 -m venv .venv-audit
source .venv-audit/bin/activate
python -m pip install -r requirements-handoff.txt
AUDIT_DIR="$(mktemp -d)"
export AUDIT_DIR
echo "$AUDIT_DIR"
python -m pytest -q
```

## 2. 从原始短 trace 复验 proposal 的 H1 和 H2

```bash
DEV=data/short/h2-development-20261120T045605Z.zip
HELD=data/short/heldout-20261120T043814Z.zip
python validation/rank_decisions.py "$DEV" --out "$AUDIT_DIR/rank-dev"
python validation/short_policy_eval.py "$DEV" "$HELD" --out "$AUDIT_DIR/short"
```

- 检查 dev 20 篇、held-out 40 篇，内部文档 ID 无交集。复算 dev 的 25 组权重，只使用 dev 选定的配置评估 held-out。按键对照 docs/results/solo_rank_dev/ 和 docs/results/solo_short_heldout/，检查缺行、重复行和多余行。
- **H1** 在过去 8 步选择频率相同的组内，比较排名 1–16 和 49–64 的页在未来 8 步内的复用率。排除后续观测窗口不足的步。独立复算调整后的差值 **8.41 个百分点 [8.06, 8.74]**。另外检查可驱逐候选中上次排名前 16 的占 **1.16%**，相同缓存状态下有 **1,053 / 35,975** 个驱逐步因加 rank 改变牺牲页集合。
- **H2** 在 B=128 分别核对 LRU、近期访问加频率、再加 rank 三组重复 miss：**162,369 / 161,850 / 161,712**。固定频率权重后加 rank 省 **138 [65, 215]** 次；完整策略相对 LRU 省 **657** 次，即 **0.405%**。在 B=256 核对 12 次、区间跨零。在 B=64 核对各策略总 miss 都为 1,498,591；此时缓存只能容下当步选中的 64 页。核对 B=128 Bélády 的重复 miss 为 58,764，将它作为离线参照。

## 3. 独立验证短 trace 的排名、分数和四种策略

**先验证已归档 trace，再验证重新采集。** 分别解开两个 Short_*_Quest_results.zip，读取 scores/*.npz、original_manifest.json 及配套采集代码的字段定义。对每个文档、层和步，按当步有效历史页范围、归档精度和稳定排序规则从原始 FP32 分数重建有序 top-64。同时逐元素比较它与本次采集存储的排名、历史输入 trace 的排名，分两列报告。20 篇 dev 有 **71,120** 行，40 篇 held-out 有 **142,240** 行。每行 64 个页必须逐位置比较。原 dev 与新 L4 dev 来自不同环境，两个来源可以不同；四种策略必须消费同一次采集的排名和分数。已归档 L4 held-out 与原 held-out 的 142,240 行有序排名完全一致。

在自己的输出目录写一个简洁 CPU 参考实现，直接读取 NPZ。独立实现缓存更新，避免调用项目 simulate、evict 或 GPU kernel。冻结 B=128、K=64、α=0.5，每个文档的每一层单独从空缓存开始。

1. 每步先保护当前选中的页。需要腾位时只从未选中的驻留页驱逐，为本步全部 miss 分配槽位。记录 accesses、misses、compulsory；每个流中此前从未访问的页计为首次 miss。重复 miss = 总 miss − 首次 miss。
2. LRU 优先驱逐最久未选择的页；最近选择步相同时，当时为 miss 的页先出，再按较大页 ID 先出。其他策略的主键相同时沿用该次序。当前分数策略优先驱逐当步分数最低的候选页。
3. rank EMA 的输入为选中页的 (K−rank+1)/K，rank 从 1 开始，未选中页输入 0。score EMA 的输入为 (页分数−第 K 名分数)/(第 1 名分数−第 K 名分数+1e−6)。先按归档参考精度更新所有旧驻留页，再比较候选；新页以本步输入初始化，驱逐后清除 EMA，后来重新加载时重新初始化。
4. 用手算例先验收：K=2、B=2、依次请求 [0,1]、[1,2]、[0,2]，期望逐步 miss 为 2、1、1，共 3 次首次、1 次重复。另构造 B>K、分数同分、首尾分差为零、未入选驻留页仍更新、驱逐后再入等小例。改变未来的请求或分数，验证当前及以前的驱逐决定不变。
5. 输出逐文档×层×策略的三个整数，按键逐项比较 by_layer.csv。dev 应有 2,240 行，held-out 应有 4,480 行。首次不同处附文档、层、步、选中页、候选页、驱逐键和两侧的牺牲页。

| 归档 trace 的重复 miss | LRU | rank EMA | 当前分数 | score EMA |
|---|---:|---:|---:|---:|
| 短 dev 20 篇 | 80,897 | 80,195 | 68,165 | 65,894 |
| 短 held-out 40 篇 | 162,369 | 161,379 | 137,275 | 132,928 |

再在 Colab L4 分别完整运行两本 Short_*_Quest.ipynb。核对模型/数据 revision、文档和 prompt、GPU 环境及完成状态，对**新生成的**两个包再执行本节检查。若生成 token、分数或有序排名与已归档 L4 trace 出现差异，定位最早分歧，分别报告固定 trace 的复验和新采集上的策略结果，不把两次采集拼在一起。

## 4. 从逐文档计数重算配对区间

从自己生成的计数先合计每篇各层的重复 miss。对每一对策略，在同一组被抽样的文档上计算 1 − 策略重复 miss 合计 / 基线重复 miss 合计。短 score 实验按任务×长度四个分层，在层内以整篇文档为单位有放回抽样 10,000 次，两种策略共享每轮样本。按包内实现核对随机种子、文档排列和百分位规则。先独立计算，再与 summary.csv、ema_increment.csv 比较。

- 短 held-out 的当前分数相对 LRU 为 **15.45% [13.76%, 16.88%]**，score EMA 相对 LRU 为 **18.13% [16.76%, 19.32%]**。分别判断点估计和区间下界是否都高于 10%。
- EMA 相对当前分数少 **4,347** 次，除以 137,275 得 **3.17% [2.77%, 3.63%]**，逐篇胜 **40/40**；以 LRU 为分母时对应 **2.68 个百分点**。dev 上额外省 **2,271** 次。历史 rank 策略单独对照其同批 LRU。
- 在输出中列出基线、分子、分母、单位和胜/平/负。层与解码步共享同一文档，不能作为独立文档样本计算上述区间。

## 5. 从原始 NPZ 复验长上下文

把两套 NPZ 的目录填入 LONG_DEV 和 LONG_HELD。先逐文件核对内部 rid、prompt_tokens 和 splits/longctx_v1.json；核对 14/21 篇互斥、冻结参数来源以及每档各层容量之和为 28×B。冻结文件 splits/heldout_frozen_v1.json 的归档 SHA-256 为 24bce803f986d730bf92e84ec85eadb123896b647681f14f62bebaf1ce65dea1。

```bash
python validation/longctx_development.py "$LONG_DEV" --out "$AUDIT_DIR/long-dev"
python heldout.py eval "$LONG_HELD" --frozen splits/heldout_frozen_v1.json --out "$AUDIT_DIR/long-held" --procs 3
python validation/compare_longctx_heldout.py "$AUDIT_DIR/long-held"
```

检查 layer_check.csv 的 (doc, layer, B, config) 是否唯一，确认新旧 accesses、misses、compulsory 三列逐项相同。另查冻结 α 是否实际用于回放。

- dev 在 B=96 上核对 14×28×4＝1,568 行；当前分数、EMA 相对 LRU 分别减少 10.11%、11.58%。
- held-out 覆盖 B=72、80、96、128、256，核对 **22,344 行、67,032 个整数**全匹配。B=96 的 LRU、当前分数、EMA、分层容量+EMA、均匀容量 Bélády 重复 miss 依次为 **1,511,044 / 1,357,122 / 1,335,270 / 1,260,181 / 886,318**。
- 按整篇文档配对抽样 2,000 次，seed 0。B=96 当前分数、EMA、分层+EMA 对 LRU 的降幅分别为 **10.19% [9.91%, 10.45%]、11.63% [11.44%, 11.83%]、16.60% [16.23%, 16.99%]**。用均匀容量下 LRU 与 Bélády 的差作为分母，复算 EMA 弥合的差距比例 G=0.2814，分层后 G=0.4016，增量 0.1202，核对冻结的三项判据。

## 6. 两次单层 L4 benchmark 都要复跑

先从原硬件结果包的 timings.csv.gz（rank 在 full/ 下）重算报告使用的历史结果，再用自己的 Colab L4 会话运行两本 cache-path notebook。保留 GPU 型号、软件版本、执行日志、完整逐步计时和 smoke 检查。以下计数对应**同一固定输入的一次逻辑回放**；计时重复轮不能重复计入流量。

| notebook 与输入 | smoke 验收 | 完整回放验收 |
|---|---|---|
| KV_L4_cache_path.ipynb；上传 KV_Cache_Agent_Handoff.zip | 页内容和槽位映射正确；LRU、完整 rank 各 488 次 miss，重复 miss 均为零 | 40 篇×28 层；重复 miss 为 162,369 / 161,712，相差 657 页 |
| KV_L4_Score_Cache_Path.ipynb；上传归档 KV_L4_Short_Heldout_Quest_results.zip | 四个分层各取一条流，三策略累计验证 97,536 个页内容；按 LRU、当前分数、score EMA 汇总 smoke 总 miss 为 4,003 / 3,665 / 3,604 | 同序总 miss 为 388,049 / 362,955 / 358,608，各减 225,680 次首次 miss 后与第 3 节逐层结果相同。每页 64 KiB，逐步核对 H2D 字节数 |

每条流冷启动，保留预热、3 轮交错的同步 wall 计时和单独的分项计时。按原分析的逐文档配对方式重算 wall、resolver、host pack、H2D；统一使用“基线减策略”为正向节省。先复现旧包，再将新会话结果并排列出，记录新区间是否跨零、方向是否维持。**固定 trace 的计数与页内容要求精确一致；新会话的延迟用实测值及其区间报告。**

| 旧包配对比较 | 每层每步节省的 wall 时间 | 耗时降幅 |
|---|---:|---:|
| 原 LRU → 完整 rank | 0.202 μs [0.130, 0.273] | 0.17% |
| 分数实验 LRU → 当前分数 | 3.473 μs | 2.76% |
| 分数实验 LRU → score EMA | 3.781 μs [3.096, 4.422] | 3.01% |
| 分数实验当前分数 → score EMA | 0.308 μs [0.177, 0.443] | 0.25% |

归档分数实验三种策略平均为 125.603 / 122.130 / 121.822 μs。EMA 相对当前分数有 27/40 篇耗时更低。核对 EMA 相对 LRU 的**全部搬页降幅 7.59%**与**重复 miss 降幅 18.13%**。EMA 相对当前分数的 resolver 多用 2.247 μs，host pack 和 H2D 合计省 0.510 μs。分项和同步 wall 分轮测量，路径还包含描述符回传与槽位写入；从 wall 判定净收益，并单独报告分项。两本 notebook 各自使用同批 LRU，不能跨 benchmark 相减。

## 7. 交回测试报告

提交一个新目录，包含 TEST_REPORT.md、自己的 CPU 参考实现和运行命令、输入哈希及环境、逐层差异表、逐文档配对结果，以及新 Colab 结果包的位置。在报告开头填完下表，每行写实际观察值和证据路径。状态使用「通过」「有差异」「待验证」；因缺少输入或 GPU 而未执行的项目写明具体依赖。

| 检查项 | 状态 | 本次观察值及区间 | 对照差异或第一处失败 | 证据文件 |
|---|---|---|---|---|
| 原 proposal：H1、驱逐机会、H2、10% 目标 | | | | |
| 归档短 dev / held-out：有序 top-64、逐层四策略 | | | | |
| 新 L4 采集：20 / 40 篇及独立 CPU 回放 | | | | |
| 短 held-out：10% 下界、EMA 增量及逐篇一致性 | | | | |
| 长 dev / held-out：冻结配置、逐层匹配与三项判据 | | | | |
| 两个硬件归档：smoke、计数、wall、分项 | | | | |
| 新 L4 会话：原 rank 与 score 两次完整 benchmark | | | | |
| 论文主图、分母、计时范围及需修订的句子 | | | | |

最后独立回答三句话：原 rank 策略有没有达到 proposal 目标？当前分数与 EMA 各贡献了多少？你自己的 L4 会话里，EMA 相对当前分数是否仍有可分辨的净收益？硬件结论限于所测单层 cache path，输入排名和分数已驻留 GPU。跨会话出现零收益或反向结果也要原样报告。
