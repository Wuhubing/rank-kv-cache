# 复核说明（给 agent 执行）

你要独立复核分支 `xiaoyang/score-ema-final` 的全部结论。按顺序做完下面每一节，把结果写成报告（模板在文末）。
**不要修改、提交或推送仓库里任何已跟踪的文件**；需要写文件时写到 `/tmp/review/` 或仓库外。某一步失败时如实记录并继续下一步，不要为了让检查通过去改代码或数据。

## 0. 背景（一段话）

研究问题：HiSparse 式两级 KV cache（全部 KV page 在 host 内存，每层 GPU 上 B 个 slot），Quest 式选择器每步每层选 K=64 个 16-token page，替换策略决定驱逐哪些"驻留但本步未选中"的 page。指标 G = (LRU miss − 策略 miss) / (LRU miss − Bélády miss)。主要结论：
1. proposal 的 rank 策略无效（B=128 时只比 LRU 少 0.39% repeat miss，目标 10%）；
2. score EMA（每 page 一个浮点数）在 B=96 填上约 28% 的差距，加按层分配约 40%；在 L4 与 Apple M1 Max 两种硬件上各自预注册检验通过，在新文档和新任务上也成立；
3. 层级 GPU 测试（L4）：每 page 传输约 5.3 µs，B=96、batch ≥ 8 时每步提速 EMA +1.5%、分配 + EMA +2.1%、Bélády +5.4%。

## 1. 准备

```bash
git clone https://github.com/wxy20050403/rank-kv-cache.git && cd rank-kv-cache
git checkout xiaoyang/score-ema-final
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pytest matplotlib
mkdir -p /tmp/review
```

记录 `git rev-parse HEAD`、Python 与 numpy 版本、操作系统。

## 2. 单元测试

```bash
python -m pytest -q
```

**通过**：75 passed（允许 1 条 `All-NaN slice` 警告）。`tests/test_kvpolicy.py` 若因缺少 transformers 被跳过，记录跳过数即可。

## 3. 预注册的完整性

1. 哈希必须完全一致：

   ```bash
   shasum -a 256 splits/heldout_frozen_mps.json splits/longctx_v2.json splits/longctx_v1.json
   ```

   | 文件 | sha256 |
   |---|---|
   | splits/heldout_frozen_mps.json | f4bf17a4b6645640ce31d58b2b9f0fe5b516cb2b38538e8780434602e05a8e33 |
   | splits/longctx_v2.json | e3bfe54c9d111c71e563d3eea24f6d143e91222c7fcd28b6781a3ed586f0a3bd |
   | splits/longctx_v1.json | 1b4c763e09e48061d0605ee945ccd989e5ab082fcb7b85e488400a371449d590 |

2. 时间顺序：`git log --date=iso --format='%h %ad %s' -- docs/preregistration_mps.md docs/heldout_results_mps.md docs/results/mps_heldout`。预注册提交 `1b6fe70` 必须早于结果提交 `29e59ee`；`git log --format=%h -- splits/heldout_frozen_mps.json` 里不应有晚于 `1b6fe70` 的提交。
3. 划分无泄漏（自己写脚本检查）：
   - `splits/longctx_v1.json` 中 dev 与 heldout 的 `rid` 和 `ctx` 都不相交；
   - `splits/heldout_frozen_mps.json` 的 `dev_rids` 恰好是 longctx_v1 的 14 篇 dev；
   - `docs/results/mps_heldout/trace_inventory.json` 的 `traces` 键恰好是 longctx_v1 的 21 篇 heldout；`v2_fresh`、`v2_ood` 的分别恰好是 longctx_v2 中 `split` 为 `heldout`（3 篇）和 `heldout_ood`（8 篇）的 rid；
   - longctx_v2 的任何 rid 末段（LongBench `_id`）都不出现在 `splits/longctx_v1.json` 或 `splits/longbench_v1.json` 中；`ctx` 不出现在 longctx_v1 中。

## 4. 独立实现模拟器（最重要的一项）

**不要阅读或导入 `kvpolicy.py`、`replay.py`、`efficient_dev.py`。** 只按下面的规格自己写，用 `docs/results/review_slice/review_slice.npz` 计算，与 `docs/results/review_slice/expected_misses.csv` 的 18 行逐行比对。

数据：`d{i}_ranked` 为 int16 `[T, 3, 64]`（每步每层的 top-K page id，按分数从高到低），`d{i}_scores` 为 float16 `[T, 3, P]`（每步每层所有历史 page 的分数），`layers` = 原始层号 `[0, 11, 25]`，`expected_misses.csv` 的 `slice_layer` 是切片里的层索引。每个 (文档, 层) 是独立的缓存，B=96。

每一步 t（行 r = 第 t 步的 64 个 page id，`r[0]` 分数最高，`r[-1]` 是第 K 名）：
1. （仅 EMA）先用本步分数更新**所有当前驻留** page：`e[p] = 0.5*e[p] + 0.5*z(p)`，其中 `z(p) = (s[p] − s[r[-1]]) / (s[r[0]] − s[r[-1]] + 1e-6)`，`s` 为本步该层分数转成 float32 后再用双精度计算。
2. 命中 = r 中已驻留的 page；miss = r 中未驻留的 page。若 `驻留数 + miss 数 > B`，从"驻留且不在 r 中"的 page 里驱逐 `驻留数 + miss 数 − B` 个，键最小的先走：
   - LRU：`(last[p], hit[p], −p)`，`last` = 最近一次被选中的步，`hit` = 那一步它是否为命中（False 先走）；
   - EMA：`(e[p], last[p], hit[p], −p)`；
   - Bélády：`(−下次被选中的步, p)`，从不再被选中记为 −∞，即先驱逐最远的；平局先驱逐小 id。
3. r 中所有 page 设为驻留，`last[p]=t`，`hit[p]=(p 本步命中)`；（仅 EMA）本步 miss 的 page 设 `e[p]=z(p)`。
4. 计数：accesses += 64，misses += miss 数；compulsory = 整个流中出现过的不同 page 数。

**通过**：18 行的 `(accesses, misses, compulsory)` 全部完全相等。作者用同一规格的独立实现已得到 18/18。

## 5. 从已提交的 CSV 重算所有数字（只用 CSV）

自己写脚本重算（可以参考 `recompute_heldout.py` 的定义，但请独立实现后再与它对照）：`misses.csv` 的列为 `config, doc, layer, B, accesses, misses, compulsory`；按文档对各层求和；G 按全部文档合并计算；CI = 对文档有放回抽样 2000 次的 2.5% / 97.5% 分位（`numpy.random.default_rng(0)`，每次 `rng.integers(0, n, n)`）。

| 数据 | 文件 | B=96 期望 |
|---|---|---|
| M1 Max held-out（主检验） | docs/results/mps_heldout/misses.csv | EMA G 0.280 [0.276, 0.283]；EMA < LRU 的文档 21/21；ΔG(alloc_ema − ema) 0.120 [0.111, 0.128]；LRU 6.02%，Bélády 3.85% |
| L4 held-out | docs/results/longctx_heldout/misses.csv | EMA G 0.281 [0.277, 0.285]；21/21；ΔG 0.120 [0.113, 0.128] |
| 新 GovReport | docs/results/v2_fresh/misses.csv | EMA G 0.284；3/3 |
| NarrativeQA | docs/results/v2_ood/misses.csv | EMA G 0.283；8/8；alloc_ema G 0.411 |
| M1 Max dev | docs/results/mps_dev/misses.csv（config `lru`、`ema:0.5`、`belady`，B=96） | 合计 accesses 17,554,432；miss 1,065,097 / 959,392 / 684,401；G 0.278 |
| L4 dev | docs/results/longctx_dev/misses.csv（同上） | miss 1,178,410 / 1,060,207 / 754,799；G 0.279 |

再核对 M1 Max held-out 的 budget 曲线（G，EMA / alloc_ema）：B=72 0.256/0.256，80 0.275/0.306，96 0.280/0.399，128 0.265/0.526，256 0.235/0.235。

**通过**：G 与表中三位小数一致；CI 端点误差 ≤ 0.001；文档计数完全一致。然后运行 `python3 recompute_heldout.py <file>` 对照四个 held-out 文件，结果应与你的脚本一致。

## 6. 跨硬件一致性

对 `docs/results/mps_heldout/misses.csv` 与 `docs/results/longctx_heldout/misses.csv`（同 21 篇，不同硬件采集）：算每篇文档的 EMA G（B=96）。**期望**：L4 范围 0.261–0.304，M1 Max 0.267–0.296；同一篇两边之差的绝对值最大 0.015；alloc_ema 的逐篇差最大 0.030。

## 7. GPU 测试结果的内部一致性（只读 CSV）

1. `docs/results/mps_heldout/miss_schedule.npz`：键 `"<i>/<policy>@<B>"` 为 int16 `[T, 28]`，`rids` 给出文档。对每篇、每个 policy（lru、ema、alloc_ema、belady）、B∈{96,128}，数组总和必须等于 `docs/results/mps_heldout/misses.csv` 中同一 (config, doc, B) 的 miss 合计。
2. `docs/results/batch_bench_l4/batch_bench.csv`：检查 `step_ms = decode_ms + scoring_ms + transfer_ms_mean`，`speedup_vs_lru = LRU 的 step_ms / 本行 step_ms − 1`，`us_per_page = 1000 × transfer_ms_mean / pages_per_step`。**期望**：batch ≥ 4 时 us_per_page 在 5.3–5.7；B=96、batch=64 的 speedup：ema +1.46%，alloc_ema +2.08%，belady +5.40%。

## 8. 代码审查

1. `git diff b6b75d5 -- '*.py'`：逐个说明本分支对已有文件的改动是否会改变任何策略或指标的结果。预期只有：`heldout.py` 加了按存储 rid 的检查和从冻结文件读 α；`kvpolicy.py` 加了可选参数 `per_step`；`checks.py` 从旁边的 json 读 prompt 长度；`recompute_heldout.py` 把 18/21 按文档数换算。
2. 比较 `local/capture.py` 中 `Quest.__call__` 与 `notebooks/long_trace_capture.py` 中 `QuestPages.__call__`：选择器的数学（page 上下界、打分、排序、拼接 attention 的 token）应完全相同。
3. `local/make_split.py`：确认排除规则与第 3 节一致，抽样是确定性的。
4. 列出你认为有问题的地方，并说明严重程度。

## 9. 可选：需要特定硬件

- **Apple silicon Mac**：`pip install -r requirements-local.txt` 后运行 `python local/capture.py check`，应有三行 PASS（selector_bounds、stable_sort、rerun_identical）。
- **Colab GPU**：上传运行 `notebooks/batch_transfer_bench.ipynb`（第一格上传 `docs/results/mps_heldout/miss_schedule.npz`）。L4 上应得到每 page 约 5.3 µs，B=96、batch 64 时 EMA / alloc_ema / Bélády 提速约 +1.5% / +2.1% / +5.4%（各 ±0.3 个百分点）。换用其他 GPU 时记录 GPU 型号与数字，不判定通过与否。

## 报告模板

写到 `/tmp/review/REVIEW_REPORT.md`，再发给 Xiaoyang：

```markdown
# 复核报告
- 复核人 / agent：
- commit：            Python / numpy / OS：
| 节 | 检查 | 结果（PASS / FAIL / SKIP） | 实际数字或说明 |
|---|---|---|---|
| 2 | 单元测试 | | |
| 3.1 | 哈希 | | |
| 3.2 | 预注册时间顺序 | | |
| 3.3 | 划分无泄漏 | | |
| 4 | 独立模拟器 18/18 | | |
| 5 | CSV 重算（6 组数据 + budget 曲线） | | |
| 6 | 跨硬件一致性 | | |
| 7.1 | miss 计划表总和 | | |
| 7.2 | batch 结果内部一致 | | |
| 8 | 代码审查 | | 问题列表 |
| 9 | 可选硬件检查 | | |
## 发现的问题（按严重程度）
## 你认为报告里不应该这样说的地方
```
