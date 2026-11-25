RUNNING: B/C/D（CPU 回放排队中）；G2 采集中
- 2026-11-25T21:00Z 开始。前提：rkv-arkvale DONE；wulab2 无 wb-rkv- 会话。
- A 部分完成（replication-v2）：08db627、fb0a8fa、a8fb184、42c2b06、851d567（STATUS_fixes.md 列出 R1–R13）。已合并进 mechanism-v3（46fcff1）。
- F1 完成：2262 篇候选，取 21 篇（31a1ae7，splits/longctx_v3.json，docs/v3/f1_pool.md）。
- mechanism-v3 代码：B4 采集（--record-attn-all）、kvpolicy_v3、analysis_v3（run/experiments、b2_stats、b2_model、noise、c4_alloc、instrument、learn、e_bench）、gpu_bench_batch2（E1，GPU 测试通过）。
- B4 采集完成：data/dev_attn_all（14/14 与 dev 逐位一致），b4 回放完成（审计 0 不一致）。初步：真实注意力作为驱逐信号 G −0.86（远差于 Quest 分数 0.24）。
- 正在跑（wulab2）：wb-rkv-fu-b-cpu（B 回放队列）→ wb-rkv-fu-cd-cpu（C/D）→ wb-rkv-fu-more-cpu（instrument/lru_fine/b2 模型/学习模型）；wb-rkv-fu-g2-collect（8B 长解码，GPU 0 = GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3）
- 注意：卡 2 两次报 "illegal instruction"（ArkVale 任务一次、本任务 smoke test 一次），脚本优先避开卡 2。
- 下一步：B/C/D 结论页；C 门槛 → F2 预注册 → F3；E 在主机空闲时做（load < 4）
