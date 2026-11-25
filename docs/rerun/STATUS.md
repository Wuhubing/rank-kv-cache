RUNNING: S7
使用的 GPU: 卡 0，GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3（RTX 6000 Ada）
已完成: S0 (6816e85), S1 (189d524), S2 (b0696af), S3 (78c277a), S4 (d11bae5), S5 (8789fca), S6 主体（8B 预注册 328059e 于 08:35:29Z，第一篇 8B held-out 08:35:54Z；四条判据全部通过；64K/RULER/SCBench/真实注意力结果已写入 docs/rerun/s6_extensions.md；page/K 扫描和采样解码已采集，分析排在 S7 之后）
进行中: S7（tmux wb-rkv-s7-bench，run_s7.sh，logs/s7.log）：0.6B 单请求延迟/分解/显存/正确性 → batch 1–32 → 8K 上下文 → B=80/256 → 8B 单请求 → 8B batch → FLOPs；之后跑 page/K 扫描和采样的回放
最近一次检查: S6 8B 确认性评估通过（audit 0 mismatch）；S7 冒烟测试通过（LRU/Bélády 实测 miss = 模拟器，batch 内每个请求 = 模拟器）
下一步: S7 结论 s7_systems.md；补完 S6 第 4、5 节；S8 汇总和 PR
BLOCKED 原因（如有）: 无
计时: S0 ~10 min；S1 ~20 min；S2 ~25 min；S3 采集 12 min；S4 CPU 78 min；S5 采集 18 min + 评估 3 min；S6 GPU 07:25–10:36（与其他采集并行）
wulab2 输出目录: out/results_v2/（不在 git 里，rsync 到 mini 的 repo/results_v2/）
