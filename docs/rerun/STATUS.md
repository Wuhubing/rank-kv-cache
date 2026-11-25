RUNNING: S8
使用的 GPU: 卡 0，GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3（RTX 6000 Ada）
已完成: S0 (6816e85), S1 (189d524), S2 (b0696af), S3 (78c277a), S4 (d11bae5), S5 (8789fca), S6（8B 确认性评估 92e70c2；64K/RULER/SCBench/真实注意力 50b2165），S7（af4fa9d：batch 1 延迟/分解/显存/正确性，batch 1–16，8B，FLOPs）
进行中: wulab2 上 page/K 扫描和采样解码的回放（run_s7.sh 尾部）；之后 0.6B batch 24/32 和计时分解（run_s7b.sh）、8B batch 32（12K token，run_s7c.sh）
最近一次检查: S7 通过（LRU/Bélády 实测 miss 与模拟器完全一致，batch 内每个请求一致）
下一步: 补完 S6 第 4、5 节和 S7 batch 32；README/STATUS_REPORT/workshop README 已更新；开 PR；REPORT.md；清理 tmux 和 GPU 进程
BLOCKED 原因（如有）: 无
计时: S0 ~10 min；S1 ~20 min；S2 ~25 min；S3 采集 12 min；S4 CPU 78 min；S5 采集 18 min + 评估 3 min；S6 GPU 07:25–10:36；S7 GPU 10:35–12:34
磁盘: 项目目录 27 GB（trace 21 GB，.venv 5.5 GB），HF 缓存 18 GB；/data 剩 9.9 TB
wulab2 输出目录: out/results_v2/（不在 git 里，rsync 到 mini 的 repo/results_v2/）
