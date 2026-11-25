RUNNING: S4
使用的 GPU: 卡 0，GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3（RTX 6000 Ada）
已完成: S0 (6816e85), S1 (189d524), S2 (b0696af), S3 (78c277a：14 篇 dev 重采，与 L4 对比，EMA G 0.279 不变), S5 (8789fca：复现预注册 86f1166 于 06:58:09Z 提交，第一篇 held-out 06:58:24Z 开始采集；三条判据全部通过，EMA G 0.279 [0.275,0.282]，21/21，ΔG 0.119)
进行中: S4 CPU 回放（tmux wb-rkv-s4-replay，run_s4b.sh，logs/s4b.log；约 2.5 h）；S6 采集排队（tmux wb-rkv-s6-collect：0.6B dev 带 read-out → 8B dev → 64K → RULER → SCBench 单会话检查）；S6 split 文件已提交（b2d2ab1）
最近一次检查: S5 通过（预注册早于 held-out 采集；audit 0 mismatch）
下一步: S4 结论页；S6 8B 冻结 + 预注册 + held-out；S7 gpu_bench batch>1
BLOCKED 原因（如有）: 无
计时: S0 ~10 min；S1 ~20 min；S2 ~25 min；S3 采集 12 min；S5 采集 18 min + 评估 3 min
wulab2 输出目录: out/results_v2/（不在 git 里，rsync 到 mini 的 repo/results_v2/）
