RUNNING: A1
任务: ArkVale 完整版基线（replication-v2 追加），任务书 ~/work/rkv-arkvale/TASK.md
已完成:
- A0 读官方代码（pku-liang/ArkVale @ 0c3b45e，clone 在 ~/work/rkv-arkvale/ArkVale），规则对照表写在仓库 docs/rerun/s4/09_arkvale_full.md（R1–R10，带文件行号）
  要点：代码只实现 cuboid-mean digest（infer_state.py:448-462）；默认 n_sink=1、n_win=2（含正在写的页）、n_unlimited_layers=2、page 32；驱逐=最低当前估计分（select.cu:122-143）；窗口滑出且缓存满时直接丢弃（kv_cache.py:162-194）
- 代码：kvpolicy_v2（forced_pages/with_forced/simulate_arkvale/unlimited_budget）、collect/long.py（--record-arkvale-digest，--selector arkvale-mean）、analysis_v2/arkvale_full.py（compare/replay/audit/summarize）；测试 tests/test_arkvale_full.py + tests/test_collect_long.py 新增，mini 上全部通过
下一步: 推送 → wulab2 pull + 测试 → GPU 采 0.6B dev 带 digest（data/dev_digest）与 ArkVale 选择器（data/dev_arkvale_sel）→ compare 验证不改选择 → replay（A1/A2/A3 同一次回放）
GPU: 尚未选卡
