RUNNING: R11-6
- 2026-11-27T22:55Z 开始。wulab2 四张卡都被占用（他人进程），先做非 GPU 项。
- R11-1 复现与 REVIEW 一致（k128 batch32：LRU 5879/5232，nofetch 1924/1170，allmiss 118272/114688）。e_summarize 改为只用计时步（9114ef7），summary 重生成（88f2c39）。传输占比全部与验收表一致；每页成本/上界与 REVIEW 不同：REVIEW 从 allmiss 的 miss 里扣了 nofetch 记下但没有传输的 miss（得 2.69 µs），本仓库按文档用 K·R·L 作分母（2.50 µs，D2 b=2.48 µs 佐证）。见 review_response。
- R11-2 复现一致（batch1 35/35、batch4 35/35、batch8 2/35）；clean 列、other_procs_seen=run 数、max_load=before/after 最大值；throughput.csv 16 行 clean=False。
- R11-3 复现一致（旧 CSV）；修复后 batch 32：EMA/分配+EMA 低于预测 0.4–1.0 点，Bélády 低 0.17–0.25 点。文档已改。
- R11-5：nofetch 检查输出 n/a；显存表；replication-v2 944a9ac 标 LODO，已合并进 mechanism-v3（0fc99d5）；c2 探索性说明。
- 测试（wulab2，CPU，CUDA_VISIBLE_DEVICES 置空）：466 passed, 9 skipped。偏离：第一次运行 pytest 时没设 CUDA_VISIBLE_DEVICES、也没限线程（约 40 核），发现后立即停掉；未产生 GPU 进程。
- GPU：tmux wb-rkv-r11-gpu（runs/run_r11.sh）23:12Z 开始等卡（每 30 分钟查一次，每项最多等 6 小时）。

- R11-4 完成（ffc37cf）：5 条规则逐条对照 pool_run 核实；第 2 条 REVIEW 写 /K，代码是 /本步选中页数（层内页数 ≥ K 时等于 K），附录按代码写。preregistration_v3.md 只在末尾追加附录（前缀 MD5 82f2606… 未变，0 行删除）；tests/test_pool_spec.py 127 组逐位一致。
- R11-6 第一次尝试 23:42Z 卡 0：00:01Z 他人（joshhan）训练任务出现在卡 0，停掉自己的 e_bench（18 个 run 未写盘，日志保留为 batch_8b_rerun.card0_aborted.log）；00:01:35Z 在卡 3（GPU-850490f2-9baf-0ef9-bf4e-a84a10d01f50）重启，每个 run 最多等 10 分钟、重跑 1 次。
