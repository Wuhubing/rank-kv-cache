RUNNING: S6
使用的 GPU: 卡 0，GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3（RTX 6000 Ada）
已完成: S0 (6816e85), S1 (189d524), S2 (b0696af), S3 (78c277a), S4 (d11bae5：全部基线、预算扫描、α 曲线、析因、预测质量、统计和噪声；thesis 第 1 点定为"全局排序"), S5 (8789fca：复现三条判据全部通过)
进行中: S6 采集（tmux wb-rkv-s6-collect：8B dev 12/14 → 64K → RULER → SCBench 单会话检查）；0.6B dev 带 read-out 已采完，与 S3 trace 逐位相同，真实注意力 InfLLM 结果已出
最近一次检查: S4 通过（audit 0 mismatch；rank control 交叉核对通过）
下一步: 8B 冻结 + 预注册（docs/preregistration_8b.md 草稿已写，未提交）→ 8B held-out；page/K 扫描、采样解码、SCBench 批量；S7
BLOCKED 原因（如有）: 无
计时: S0 ~10 min；S1 ~20 min；S2 ~25 min；S3 采集 12 min；S4 CPU 78 min（06:57–08:15 UTC）；S5 采集 18 min + 评估 3 min
wulab2 输出目录: out/results_v2/（不在 git 里，rsync 到 mini 的 repo/results_v2/）
