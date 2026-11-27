DONE
任务: ArkVale 完整版基线（replication-v2 追加），任务书 ~/work/rkv-arkvale/TASK.md
已完成:
- A0 读官方代码（pku-liang/ArkVale @ 0c3b45e，clone 在 ~/work/rkv-arkvale/ArkVale），规则对照表写在仓库 docs/rerun/s4/09_arkvale_full.md（R1–R10，带文件行号）
  要点：代码只实现 cuboid-mean digest（infer_state.py:448-462）；默认 n_sink=1、n_win=2（含正在写的页）、n_unlimited_layers=2、page 32；驱逐=最低当前估计分（select.cu:122-143）；窗口滑出且缓存满时直接丢弃（kv_cache.py:162-194）
- 代码：kvpolicy_v2（forced_pages/with_forced/simulate_arkvale/unlimited_budget）、collect/long.py（--record-arkvale-digest，--selector arkvale-mean）、analysis_v2/arkvale_full.py（compare/replay/audit/summarize）；测试 tests/test_arkvale_full.py + tests/test_collect_long.py 新增，mini 上全部通过
- A3 采集（0.6B dev）：data/dev_digest（带 5 种 digest 读出）14 篇 17:44–17:54Z；compare：14/14 篇 ranked/scores/生成 token 与 data/dev 完全一致（不改选择）
- A4 采集：data/dev_arkvale_sel（--selector arkvale-mean，1 sink + 1 recent + 62）17:54–18:02Z；与 Quest 生成的相同前缀中位 28 token（Quest 与 dense 本身中位 14）
- A1/A2/A3 回放（main，3010 jobs，17:54–19:21Z）：audit 33 536 条 LRU/Bélády 0 mismatch；结论：ArkVale 完整版（page 流、口径 a、cuboid-mean、窗口规则）B=96 G −0.219 vs EMA 0.278（ΔG −0.497，0/14）；最佳 digest（Quest 分数）0.200（ΔG −0.078）
- A4 回放（e2e，448 jobs）：audit 6 676 条 0 mismatch；ArkVale 自身选择器下 EMA 仍胜（ArkVale 0.197 vs EMA 0.265，ΔG −0.068，0/14）
- 8B dev 带 digest 采集：卡 2 上两次 CUDA 错误（illegal instruction、CUBLAS_STATUS_EXECUTION_FAILED，非代码问题）后第 3 次完成；14/14 与旧 8B trace 完全一致
- 额外全层回放（0.6B，B=192/384/512，19:30–20:18Z）：口径 b 总 miss 对比；audit 合计 38 240 条 0 mismatch
- 8B 回放（19:23–20:56Z）：ArkVale 完整版 B=96 G −0.344 vs EMA 0.270（ΔG −0.614，0/14）；audit 27 632 条 0 mismatch
- A5：结论页 docs/rerun/s4/09_arkvale_full.md 完成；REPORT.md 写好
已更新文档: docs/rerun/s4/09_arkvale_full.md、thesis.md、pr_description.md（第 5 条偏离已改写）、s4/README.md、s4/04_new_baselines.md
GPU 使用已结束（卡 2 无本任务进程）
GPU: 卡 2，GPU-28e78787-bb85-6a09-709b-46aeef3757c5（free-gpus 1 选定，17:44Z；卡 0/1/3 被他人占用）
清理（20:57Z）: wulab2 无 tmux 会话、无本任务 GPU/python 进程；卡 2 29 MiB / 0 %
