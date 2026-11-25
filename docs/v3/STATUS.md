RUNNING: A1/A2/A3/A5/A7（回放中）；B4 采集中
- 2026-11-25T21:00Z 开始。前提：rkv-arkvale DONE；wulab2 无 wb-rkv- 会话。
- F1 完成：候选 2262 篇（GovReport 2205 + QMSum 57，去重后），取 21 篇（全部 GovReport），F 可做。输出在 mini ~/work/rkv-followup/f1/，将随 mechanism-v3 提交 splits/longctx_v3.json。
- replication-v2 提交：08db627（A1/A2/A5/A6/A7 代码与测试）、fb0a8fa（A4/A8/A9/A10/A11 文档与 provenance）
- mechanism-v3 已推送（4a83520 B4 采集代码），之后合并修正完的 replication-v2。
- 正在跑（wulab2）：wb-rkv-fu-a-replay（fix_v3 回放 + 审计 + predict，14 进程，CPU）；wb-rkv-fu-b4-collect（GPU 0，GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3）
- 下一步：A1/A2/A3/A5/A7 文档与 STATUS_fixes.md；然后 B 部分
- mini 工作目录：~/work/rkv-replication/repo/.claude/worktrees/rv2（replication-v2）、.../v3（mechanism-v3）；wulab2：/data/weibing/projects/rank-kv-cache（replication-v2）、/data/weibing/projects/rkv-v3（mechanism-v3）
