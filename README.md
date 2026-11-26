# Can Selection Rank Improve KV Cache Replacement?

A single-GPU study of HiSparse-style hierarchical KV caching (MIT 6.5940, Fall 2026, Project 10).

We test whether a page's sparse-attention top-K selection rank predicts reuse beyond
recency and frequency, using Qwen3-0.6B traces on LongBench (MultiFieldQA-en, GovReport).

**Current status (2026-11-25):** the rank study ended negative (preregistered 40-document held-out test: the full rank policy cuts repeat misses by only 0.39% [0.20%, 0.57%] against LRU, S1 not met; S2 is met as preregistered but also changes β, and a post-hoc comparison with β held fixed puts the rank term alone at 0.14%). A follow-up line, a score-EMA replacement policy on long Quest traces, passed its preregistered 21-document held-out test but gives no latency gain on an L4. The Chinese walkthrough below explains it from the start; [docs/STATUS_REPORT.md](docs/STATUS_REPORT.md) has the full log, data and how each conclusion was reached.

**Followup-v3 (2026-11-26, branch `mechanism-v3`, [docs/v3/](docs/v3/)):** a third preregistered held-out set (21 new
GovReport documents, both models) confirms the EMA again (G 0.281 / 0.277) and two new policies chosen on dev: a normalised
power-law kernel of the score (+0.048 / +0.043 G over the EMA, 21/21) and a global slot pool ranked by the score percentile that
needs no offline allocation curves (+0.119 / +0.233 over uniform budgets). Mechanism: the policy must predict the selector's next
~16 steps; true attention is a worse eviction signal than the selector's own score, and the remaining gap is not reachable with
richer learned features. An independent replication on Apple M1 Max by Xiaoyang Wu (PR #9) passes the original held-out criteria as well (G 0.280; [docs/v3/external_xiaoyang.md](docs/v3/external_xiaoyang.md)). Summary: [docs/v3/thesis_v3.md](docs/v3/thesis_v3.md); fixes to replication-v2:
[docs/rerun/STATUS_fixes.md](docs/rerun/STATUS_fixes.md).

**Replication-v2 (2026-11-25, branch `replication-v2`):** everything was re-collected and re-evaluated from scratch on an RTX 6000 Ada ([docs/rerun/](docs/rerun/)). The preregistered held-out result replicates (EMA G 0.279 [0.275, 0.282], 21/21; L4: 0.281), and a new preregistered **Qwen3-8B** held-out set confirms it (G 0.272 [0.269, 0.276], 21/21; allocation + EMA 0.273 of the gap at the same allocation). An all-page percentile control shows that the signal is the **global ordering** of the unselected pages' scores, not score magnitude. It also holds at 64K (YaRN), on RULER and on SCBench multi-turn sessions. Measured batching (1–32) shows that the 8–10 % byte saving does not become latency or throughput (the batch-32 difference on 0.6B comes from LRU's long-tail steps with the same median step and is not attributed to the policy; none on 8B). Summary: [docs/replication_v2.md](docs/replication_v2.md) (preregistration), [docs/rerun/thesis.md](docs/rerun/thesis.md) (what the evidence supports).

## 高效替换策略：从问题到结果

下面从头讲一遍：问题是什么，策略怎么来的，最后得到什么结果。详细数据和推导见 [docs/STATUS_REPORT.md](docs/STATUS_REPORT.md) §9。

### 1. 问题：GPU 放不下所有 KV

- 长文本解码时，KV cache 很大（2–3 万 token）。分层的做法是**把所有 KV page 放在主机内存**（每个 page 16 个 token），**GPU 上每层只留 B 个槽位**作为缓存。
- 每生成一个 token，Quest 选择器给所有 page 打分（用每个 page 的 key 最小、最大值估计它和当前 query 的相关度），取**分数最高的 K=64 个 page** 做注意力。
- 这 64 个必须都在 GPU 上。不在的就是 **miss**，要通过 PCIe 从主机拷过来；如果槽位满了，还得**换出**一个旧 page。
- **我们研究的问题：换出哪个？** 换得好，后面 miss 就少。

### 2. 基准和上限

- **LRU**（HiSparse 用的）：换出最久没被选中的 page。
- **Bélády**（理论最优，需要知道未来）：换出下一次被使用最晚的 page。
- 两者之间有一段差距（B=96 时 miss rate 是 6.03% 对 3.86%）。**G = 新策略补上了这段差距的比例**：0 表示和 LRU 一样，1 表示达到最优。

### 3. 第一轮：rank 策略为什么失败

- 最初的想法是用 page 在 top-K 里的**排名**来决定换出谁。
- 结果 G ≈ 0.027，几乎等于 LRU。
- 反思后发现：v1 trace 只记录了被选中 page 的排名，**丢掉了两样关键信息**：分数的具体大小，以及没被选中的 page 当前的分数。排名只告诉你"第几"，不告诉你"差多少"，也不告诉你"刚掉出 top-K 的 page 还差一点就能回来"。（这两样里到底哪一样起作用，现有消融还分不清，见 §6。）

### 4. 核心直觉：分数会延续

- 一个 page 这一步分数高，下一步大概率也高，因为相邻 token 的 query 很像。
- 所以分数本身就是"它很快还会被用到"的信号，**包括没被选中的 page**：分数排在第 65 名的 page，比排在第 500 名的更可能很快被选中。
- LRU 只看"上次什么时候被选中"，完全忽略了这个信号。

### 5. 策略本身：分数 EMA

每一步、每一层：

1. **归一化分数**：`z = (s − s_K) / (s_1 − s_K)`，其中 s_1 是最高分，s_K 是第 K 名的分数。这样每一步的分数尺度一致，第 K 名为 0，第 1 名为 1。
2. **对 GPU 上每个驻留 page 更新**：`e ← α·e + (1−α)·z`，即指数滑动平均，α 是衰减系数，B=96 时为 0.5。
3. **需要换出时**，在没被选中的驻留 page 里换出 **e 最小**的。

开销：每个 page 只多存 1 个浮点数，每步做一次乘加。

### 6. 一步步验证"为什么有效"（消融实验）

| 变体 | G | 说明 |
|---|---|---|
| 只看当前一步的分数（KVDrive 式） | 0.244 | 分数本身就有用 |
| **分数 EMA** | **0.279** | 平滑掉噪声后更好 |
| 只在 page 被选中时才更新 | EMA 版 −1.596；InfLLM 式 0.028 | 远差于或几乎等于 LRU，说明**必须更新所有驻留 page** |
| 用 top-K 内排名代替分数（未选中 page 一律记 0） | 0.016 | 几乎没用。但这个变体**同时**去掉了分数大小和未选中 page 之间的排序 |
| **全部可选 page 的分数百分位排名做 EMA**（`signal="allrank"`，replication-v2 新跑） | **0.271** | 保留排序、去掉大小后，以 LRU 为基线仍有 EMA 的 97%；以当前分数（α=0，三种信号此时相同，0.240）为基线，历史带来的收益保留约 80%（0.031 对 0.038）：**排序占历史收益的约 4/5，大小占约 1/5**（followup-v3 R3 更正）（[s4/02](docs/rerun/s4/02_allrank_control.md)） |
| 衰减计数（只数被选中几次） | 0.044 | 频率信号很弱 |
| LFU、ARC、LRU-2 等传统策略 | ≤ 0 | 都不如 LRU |
| 线性回归预测下一次使用时间（学习式） | 0.300 | 只多一点，代价是每页存约 5 个数 |

### 7. 叠加：按层分配槽位

- 观察到差距集中在少数几层：有的层 page 选择很稳定，有的层变化剧烈。
- 用 dev 上每层的 miss 曲线，在总预算 28·B 不变的前提下，给"难"的层多分槽位、"简单"的层少分。
- 和分数 EMA 几乎线性叠加：0.279 + 0.127 ≈ **0.409**（G 相对的是**每层均匀预算**的 LRU 和 Bélády；槽位在层间移动后，均匀预算的 Bélády 不再是这个分配下的最优，所以这不是"补上差距的比例"）。
- replication-v2 补上了同分配锚点（`validation/alloc_oracle.py`）：held-out 上，相对同分配的 Bélády，分配 + EMA 补上 **35%** 的差距（0.6B）和 **43%**（8B）；在同一分配内只看替换部分，EMA 仍补上 28%（0.6B）/ 27%（8B），和均匀预算时一样。也就是说按层分配和替换策略是**相加**的。

### 8. 结果

**模拟器（dev，14 篇）**：G 0.279，14/14 篇好于 LRU。换模型、换提示词、换 5 种预算都成立。

**预注册后的 held-out（21 篇，只评估一次）**：
- 分数 EMA：G **0.281** [0.277, 0.285]，**21/21** 篇好于 LRU。
- 在总槽位相同（每篇 28·96）的前提下，repeat miss 比 LRU 少 **11.6%**；加按层分配后少 **16.6%**。
- 加按层分配后 G 为 0.402，但这是相对**均匀预算**的 LRU/Bélády 算的，不能读作"补上 40% 的差距"（见 §7）。
- 三条成功标准全部达到。

**真实 GPU（L4，Qwen3-0.6B）**：
- miss 数和模拟器一致，PCIe 流量少约 9.3–9.8%（EMA 单独）、13–19%（EMA + 按层分配）。
- 第一版未融合实现里，EMA 反而慢一点（93.1 对 86.6 ms/token），因为多了很多小 kernel。
- **把策略写成融合 Triton kernel 后**：EMA 的策略开销每 token 只比 LRU 多 0.084 ms；LRU 62.3、EMA 62.3、分配 + EMA 61.6、Bélády 62.1 ms/token，四者在噪声以内。
- 也就是说，**EMA 不增加延迟，同时少传约 9.3–9.8% 的数据（加按层分配为 13–19%）**；但 batch 1 下传输只占 GPU 时间约 1%，所以也没变快。（原先按单 page 7.3 µs 线性外推的"batch 32 每步省 2.3 ms"已被下面的实测取代。）

**replication-v2（RTX 6000 Ada，与上面的 L4 结果并排）**：

| | L4（原结果） | RTX 6000 Ada（replication-v2） |
|---|---|---|
| dev 14 篇，B=96：LRU miss / EMA G | 6.03% / 0.279 | 6.04% / 0.279（每篇生成长度不同，汇总几乎不变） |
| 0.6B held-out 21 篇（预注册，只评估一次） | G 0.281 [0.277, 0.285]，21/21，ΔG(分配) 0.120 | G **0.279** [0.275, 0.282]，21/21，ΔG 0.119 |
| Qwen3-8B 新 held-out 21 篇（新预注册） | — | G **0.272** [0.269, 0.276]，21/21，ΔG 0.252，同分配内 0.273 |
| 64K（YaRN）/ RULER / SCBench 多轮（dev，探索性） | — | EMA G 0.284 / 0.255 / 0.285，均 14/14 |
| batch 1 延迟（融合 kernel，B=96） | LRU 62.3，EMA 62.3 ms/token | LRU 30.3，EMA 30.3 ms/token（8B：40.6 / 40.6） |
| batch > 1 | 未实现（外推） | 实测 batch 1–32：吞吐近似线性增长；batch ≤ 24 时 EMA 少传 8% 字节但吞吐 0.90–1.00× LRU；batch 32（0.6B）观察到的差别来自 LRU 的长尾步，中位步长相同（92.7–93.0 ms），不能归因于策略（策略按固定顺序运行，每组只有 2 次重复）；8B batch 32 在噪声内 |

详见 [docs/rerun/](docs/rerun/)：s3（dev 对比）、s4（全部基线和分析）、s5（复现）、s6（扩展）、s7（系统）。

### 9. 一句话总结

> 从 rank 失败中发现"未被选中 page 的当前分数 + 全体驻留 page 都要更新"是关键（replication-v2 的全 page 排名对照表明，起作用的是未选中 page 之间的全局排序，而不是分数大小），于是用一个每页只存 1 个数的分数滑动平均来做换出决策。它稳定拿到 LRU→最优差距的 28%；加按层分配后，在总槽位相同时 repeat miss 比 LRU 少 16.6%（相对均匀预算锚点的 G 为 0.40，同分配 Bélády 待算），真实系统里 PCIe 流量减少约 9.3–9.8%（EMA 单独）、13–19%（加按层分配），用融合 kernel 实现后开销和 LRU 相同。但在小模型、单卡、batch 1 上，这些节省换不成延迟，只是持平。

**定位**：这个规则和 InfLLM 的衰减分数规则很接近，本质是把它用在 Quest 分数上。贡献在于系统比较、消融分析、按层分配的叠加，以及"什么时候省下的 miss 能变成延迟"的真实测量。

## Team

Jiayu Li, Mingxi Wang, Weibing Wang, Xiaoyang Wu, Xihang Yu

## Collecting traces

1. In [Google Colab](https://colab.research.google.com/), select **File → Upload notebook** and choose [first_trace.ipynb](notebooks/first_trace.ipynb).
2. Select **Runtime → Change runtime type → GPU** and **Runtime → Run all**. The notebook includes the replay functions and downloads pinned public model and data revisions. It uses BF16 on supported GPUs and FP16 otherwise.
3. Download the ZIP from the final cell. It contains `trace.jsonl`, a shared `manifest.json`, one manifest per document under `manifests/`, `comparison.csv` and `comparison.png`. The same files remain under `results/` while the session is alive.

The notebook collects the eight `dev_collection` prompts of the frozen split (see below), uncropped, one unpadded request at a time:
- Request IDs have the form `task/band/_id`.
- Dense prefill produces token 1. Subsequent tokens come from greedy sparse decoding with the full KV store on device. With a 128-token cap, at most 127 sparse steps are logged per document.
- A step selects 64 completed 16-token historical pages per layer and also attends to the pinned page containing the current query token.
- Each document's manifest records the source row, tokenizer and model revisions, prompt template, token-ID hash, GPU, precision and generated IDs.

The first pilot (commit `0f35894` and earlier) used a single GovReport document cropped to a 4,096-token prompt. That document is in dev.

## Trace format (v1)

Use UTF-8 JSON Lines, one object per request, layer and decode step. For example:

```json
{"request_id":"toy","layer_id":0,"decode_step":0,"ranked_page_ids":[7,2,4]}
```

| Field | Meaning |
| --- | --- |
| `request_id` | Nonempty string identifying one generation run. Use different IDs for independent generations. |
| `layer_id` | Zero-based nonnegative layer index. Each request and layer has its own initially empty cache. |
| `decode_step` | Integer starting at 0, contiguous and increasing within its stream. Streams can be interleaved. |
| `ranked_page_ids` | Ordered, unique, nonnegative page IDs, best first. Index 0 means rank 1. An empty list is valid when no historical page exists. |

For the Qwen pilot, token positions and physical-page boundaries are zero-based. Historical page ID `p` covers positions `16*p` through `16*p+15`. The page holding the current token is pinned through that query. It becomes eligible starting with the next page. Its accesses and storage are excluded from historical-page counters and budget. Decode scores use per-coordinate extrema of post-RoPE keys, summed signed query bounds and mean over query heads, with smaller page ID first on a score tie. The separate manifest carries the page size and selector settings. Handwritten schema examples can use smaller K.

## Replay locally

From the repository root, run the unit tests with Python 3.10 or newer. The replay and analysis scripts need only the standard library. Matplotlib is optional and adds the PNG figures; without it, the CSV files are still written.

```bash
python3 -m unittest discover -s tests -v
python3 -m pip install matplotlib
python3 replay.py examples/handwritten.jsonl --budgets 2 3 4 --out results/handwritten
```

For a downloaded Colab trace:

```bash
python3 replay.py path/to/trace.jsonl --budgets 64 128 256 --out results/govreport
```

All policies observe the same set simultaneously, protect that entire selection, admit every miss and evict only as many unselected residents as necessary. Choose them with `--policies` (default `lru belady`):

| Name | Evicts first | Ties |
| --- | --- | --- |
| `lru` | Oldest last selection step | Fills before same-step hits, then larger page ID |
| `fifo` | Oldest admission; hits do not refresh | Larger page ID |
| `rank` | Worst rank at its last selection (rank 1 is best) | LRU order |
| `rf` | Lowest score with `delta=0`: recency plus frequency | LRU order |
| `rrank` | Lowest score with `beta=0`: recency plus rank | LRU order |
| `full` | Lowest score with both weights | LRU order |
| `score:D:B` | Lowest score with explicit weights, for example `score:2:1` | LRU order |
| `belady` | Farthest next use; no future use counts as infinity | Smaller page ID |

The score is `tau + delta * [rank <= ceil(K/4)] + beta * min(f, 4)`:
- `tau` is the page's last selection step and `rank` is its rank at that step.
- `f` counts the selections in the eight steps before the current one.
- `K` is `--top-k`, defaulting to the widest selection in the trace.
- `--delta` and `--beta` are in decode steps and feed `rrank`, `rf` and `full`.

With both weights at zero, the score policy reproduces LRU eviction step by step. Only the oracle sees future steps. For example:

```bash
python3 replay.py path/to/trace.jsonl --budgets 128 256 \
  --policies lru fifo rank rf rrank full belady --delta 4 --beta 2 --out results/govreport
```

The left panel divides total misses by all selected historical page accesses. The right divides repeat misses by repeat accesses. A first selection is always compulsory within its request/layer stream. An undefined rate has an empty CSV field. Counts are pooled across streams before division.

`--by task band cell` also replays each task, length band and task/band cell separately and writes every group to `comparison_by_group.csv`; the plot and `comparison.csv` keep the pooled `all` group. Bands come from request IDs of the form `task/band/_id`.

`comparison.csv` has one row per budget and policy. Its columns are `group`, `budget`, `policy`, `delta`, `beta` (empty for policies without weights), `top_k`, `accesses`, `misses`, `compulsory_misses`, `repeat_misses`, `repeat_accesses`, `miss_rate` and `repeat_miss_rate`. At the primary budget of 128, subtract the oracle miss rate from the LRU miss rate to measure the available headroom in percentage points. Store the raw file to replay future rank policies on identical selections.
**Handwritten check.** The eight accesses select four distinct pages. At budgets 2, 3 and 4, LRU has 5, 5 and 4 total misses. Bélády has 5, 4 and 4. The unit suite also checks protected selections, tie behavior, stream isolation, an independently enumerated optimum, step-by-step agreement of the zero-weight score with LRU, and the oracle as a lower bound for every policy.

## Rank predictiveness (H1)

```bash
python3 analysis.py path/to/trace.jsonl --samples selected candidate lru --lru-budget 128 \
  --by task layer-group --out results/h1
```

`--samples` chooses which pages are observed at decode step t. The default is `selected candidate`.

| Sample | Pages observed at step t | Role |
| --- | --- | --- |
| `selected` (H1-selected) | Pages selected at t | Contrast only. These pages are protected anyway, so they never face eviction. |
| `candidate` (H1-candidate) | Pages selected before t but not at t | Primary sample. The set does not depend on the policy. It is a superset of the pages a replacement policy chooses among: it also contains pages a finite cache has already evicted, and steps at which nothing is evicted. |
| `lru` | Candidates that an LRU cache of `--lru-budget` slots actually holds when step t arrives | The residents LRU chooses among when it must evict (steps without an eviction are included). |

Each observation records:
- its last rank, the rank at its most recent selection τ (τ = t for `selected`);
- the selection width n at τ;
- recency, t − τ (`new` for a first selection);
- frequency, its selections in steps t−8 to t−1;
- whether it is selected again within 1, 2, 4 and 8 steps.

Only steps followed by `max(--horizons)` observed steps are used, so every horizon is measured on the same observations. Short traces print a message and need smaller `--horizons`.

The effect compares the top quartile (last rank ≤ ⌈n/4⌉) with the bottom quartile (last rank > n − ⌈n/4⌉):
- It is computed within strata of frequency × recency bin (`new`, 1, 2, 3–4, 5–8, 9–16, 17–32, 33–64, >64).
- Strata are weighted by size, and only strata containing both quartiles count.
- The output also gives the raw difference and the effect stratified by recency alone, which shows how much of the raw gap recency and frequency explain.
- The 95% interval is a percentile bootstrap (`--bootstrap 1000`, `--seed 0`) that resamples whole requests, meaning source documents.
- With a single request, the interval is reported as n/a. `--cluster stream` resamples (request, layer) pairs instead, which is useful for a pilot but ignores correlation across layers. Intervals over sparse strata, such as small layer groups, can be lopsided.

Outputs in `--out`:
- `h1_effects.csv` has one row per sample, group and horizon. Its columns are `sample`, `group` (`all`, `task <prefix>`, `band <band>`, `cell <task/band>` or `layers a-b`), `horizon`, `observations`, `top_share` and `bottom_share` (fractions of observations whose last rank is in the top or bottom quartile; for `lru` this is close to how often a top-quartile rank bonus could apply; `candidate` also counts pages no longer resident), `clusters`, `strata_used`, `effect` (stratified by frequency × recency), `ci_low`, `ci_high`, `effect_recency_only`, `raw_top`, `raw_bottom` and `raw_difference`.
- `rank_curve.csv` gives `sample`, `horizon`, `rank_decile` (1 is best), `observations` and `reuse_rate`.
- `features.csv` is written with `--features`. It has one row per observation: `sample`, `request_id`, `task`, `layer_id`, `decode_step`, `page_id`, `rank`, `n_selected`, `quartile`, `recency`, `recency_bin`, `frequency` and `reuse_h`.
- `h1.png` plots, for each sample, reuse by last-rank decile and the raw, recency-only and frequency × recency effects against horizon.

**Caveat:** rank is relative within a step. Because every step selects exactly K pages, a page outranked by frequent competitors can show lower reuse through competition alone, even when its own propensity depends only on frequency. The synthetic tests in `tests/test_analysis.py` show this effect with a small page pool. They also show that a strong H1-selected signal need not carry over to candidates.

## Preregistered evaluation

The held-out evaluation follows [docs/preregistration.md](docs/preregistration.md). Its machine-readable configuration is [splits/preregistration.json](splits/preregistration.json): policies, tuned δ and β, frozen logistic-regression coefficients, budgets, metrics, success criteria and H1 settings.

```bash
python3 preregister.py results/dev_full --tuning-csv results/dev_full/tuning.csv   # dev only
python3 evaluate.py results/heldout --out results/heldout/eval
python3 truncate_trace.py results/heldout
python3 evaluate.py results/heldout --trace trace_truncated.jsonl --out results/heldout/eval_truncated
python3 longbench_score.py results/heldout --out results/heldout/longbench
```

`evaluate.py` writes the following to `--out`:
- `policies.csv`: per budget and policy, repeat and total misses, G and the repeat-miss reduction against LRU (each with a document-bootstrap CI), and H2D MiB per generated token at 64 KiB per page per layer;
- `policies_by_group.csv` (task, band) and `policies_by_doc.csv`;
- `success.json` with S1 and S2;
- the H1 outputs of `analysis.py`.

Post-hoc analyses, added after the held-out evaluation and not part of the preregistration, write to new `posthoc_*` folders and never overwrite the preregistered outputs. `posthoc_rank_increment.py` compares `score:8:2` with `score:0:2` (β fixed) and splits S2 into the rank term and the β change, with the same paired document bootstrap. `posthoc_censoring.py` (numpy) counts logistic training labels that are right-censored at the trace end:

```bash
python3 posthoc_rank_increment.py results/heldout/trace.jsonl --out docs/results/heldout/posthoc_rank_increment
python3 posthoc_censoring.py results/dev_full --out docs/results/dev_full/posthoc_censoring
```

`longbench_score.py` scores the sparse reference generations and the dense generations in the manifests with LongBench's own metrics. The notebook collects dense generations when `DENSE_REFERENCE = True`. Results are summarized in [docs/STATUS_REPORT.md](docs/STATUS_REPORT.md) §4.7.

## Truncating repeated answers

MultiFieldQA answers often finish within a line and then repeat "Answer: …" until the 128-token cap. That repetition inflates page reuse. `truncate_trace.py` cuts each MultiFieldQA request after the decode steps that produce its first answer, i.e. up to the first generated token containing a newline. GovReport requests are copied unchanged. It reads the per-document manifests and needs `transformers` for the tokenizer.

```bash
python3 truncate_trace.py results/dev_pilot
python3 analysis.py results/dev_pilot/trace_truncated.jsonl --samples selected candidate lru --by task band
```

It writes `trace_truncated.jsonl` and, next to it, `trace_truncated.json` with the number of steps kept per request. Short answers leave few steps with a complete 8-step window, so MultiFieldQA contributes far fewer H1 observations after truncation.

## Frozen LongBench split

`splits/longbench_v1.json` freezes the evaluation sample. It was produced once by `split.py`, which needs `transformers` and `huggingface_hub` for the tokenizer only.
- It holds 60 prompts: 15 per cell of task (MultiFieldQA-en, GovReport) × length band (4–8K, 8–16K Qwen tokens of the full, uncropped prompt).
- Each cell is split 5 dev and 10 held-out by source document. A document contributes at most one prompt, so no document is in both splits.
- The pilot GovReport document is in dev.
- `dev_collection` lists the eight dev prompts (two per cell) that the notebook collects.
- MultiFieldQA-en uses LongBench's own prompt template. GovReport keeps the pilot's summary prompt.

Tune only on dev. Do not collect or inspect held-out prompts until the final evaluation. `tests/test_split.py` checks the file against `split.py` and the notebook's configuration cell.

## Related work

[docs/related_work_matrix.md](docs/related_work_matrix.md) compares ArkVale, HiSparse, Quest, H₂O and PARROT. It records which claims we can and cannot make, and which items remain unverified.
