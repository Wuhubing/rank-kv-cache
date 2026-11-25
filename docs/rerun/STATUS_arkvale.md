RUNNING: A1
任务: ArkVale 完整版基线（replication-v2 追加），任务书 ~/work/rkv-arkvale/TASK.md
已完成:
- A0 读官方代码（pku-liang/ArkVale @ 0c3b45e，clone 在 ~/work/rkv-arkvale/ArkVale），规则对照表写在仓库 docs/rerun/s4/09_arkvale_full.md（R1–R10，带文件行号）
  要点：代码只实现 cuboid-mean digest（infer_state.py:448-462）；默认 n_sink=1、n_win=2（含正在写的页）、n_unlimited_layers=2、page 32；驱逐=最低当前估计分（select.cu:122-143）；窗口滑出且缓存满时直接丢弃（kv_cache.py:162-194）
- 代码：kvpolicy_v2（forced_pages/with_forced/simulate_arkvale/unlimited_budget）、collect/long.py（--record-arkvale-digest，--selector arkvale-mean）、analysis_v2/arkvale_full.py（compare/replay/audit/summarize）；测试 tests/test_arkvale_full.py + tests/test_collect_long.py 新增，mini 上全部通过
- A3 采集（0.6B dev）：data/dev_digest（带 5 种 digest 读出）14 篇 17:44–17:54Z；compare：14/14 篇 ranked/scores/生成 token 与 data/dev 完全一致（不改选择）
- A4 采集：data/dev_arkvale_sel（--selector arkvale-mean，1 sink + 1 recent + 62）17:54–18:02Z；与 Quest 生成的相同前缀中位 28 token（Quest 与 dense 本身中位 14）
进行中（wulab2 tmux）: wb-rkv-ark-replay-main（A1/A2/A3 回放，3010 jobs，约 2.3 h，17:54Z 起）；wb-rkv-ark-replay-e2e（6 进程）；wb-rkv-ark-collect-8b（8B dev 带 digest，卡 2）
下一步: 汇总 → 结论页 → 8B 回放（精简：plain/page/token × quest/cuboid_mean）
GPU: 卡 2，GPU-28e78787-bb85-6a09-709b-46aeef3757c5（free-gpus 1 选定，17:44Z；卡 0/1/3 被他人占用）
