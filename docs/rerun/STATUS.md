BLOCKED: 无法创建 PR——mini 上的 GH_TOKEN 是细粒度令牌，没有 Pull requests 写权限（gh pr create 和 REST API 都返回 403 "Resource not accessible by personal access token"）；其余 S0–S8 全部完成
使用的 GPU: 卡 0，GPU-a74391df-5c27-152f-d0c0-6b86f8ea07d3（RTX 6000 Ada），全程未换卡
已完成: S0 (6816e85), S1 (189d524), S2 (b0696af), S3 (78c277a), S4 (d11bae5), S5 (8789fca：复现三条判据全部通过), S6 (92e70c2 8B 四条判据全部通过；2838d8f 扩展全部完成), S7 (0713e80：含 batch 32), S8（README/STATUS_REPORT/workshop README/gpu_benchmark.md 已更新；thesis.md 逐条核对；PR 描述写好在 docs/rerun/pr_description.md）
最近一次检查: 全部通过——S5/S6 预注册早于对应 held-out 第一篇采集（06:58:09Z < 06:58:24Z；08:35:29Z < 08:35:54Z）；所有回放 audit 0 mismatch；S7 实测 LRU/Bélády miss 与模拟器完全一致；wulab2 无 wb-rkv- tmux 会话、无本任务 GPU 进程（nvidia-smi 空）
下一步: 由人开 PR（见下）；PR 开好后把本文件第一行改为 DONE
BLOCKED 原因（如有）:
  - 试过：gh pr create（GraphQL）和 gh api POST /repos/Wuhubing/rank-kv-cache/pulls（REST），都是 403。mac-mini-guide.md 第 7 节也记录了这个令牌是细粒度令牌、权限有限。没有使用其他凭据。
  - 需要人做的二选一：
    1) 在有权限的机器上，从 ~/work/rkv-replication/repo 运行：
       gh pr create --repo Wuhubing/rank-kv-cache --base efficient-policy --head replication-v2 --title "replication-v2: full re-run on RTX 6000 Ada, Qwen3-8B confirmatory held-out, batch > 1 measurements" --body-file docs/rerun/pr_description.md
    2) 或在 GitHub 上给 mini 的令牌加 "Pull requests: Read and write"（仅本仓库），再让 agent 重新运行上面的命令。
  - PR 不合并（按任务书）。
计时: S0 ~10 min；S1 ~20 min；S2 ~25 min；S3 12 min；S4 CPU 78 min；S5 21 min；S6 GPU 07:25–10:36 + 回放到 13:25；S7 10:35–12:34 + 13:25–14:14（UTC，2026-11-25）
磁盘: wulab2 项目目录 27 GB（原始 trace 21 GB，.venv 5.5 GB），HF 缓存 18 GB；/data 剩 9.9 TB；未上传 Drive（用户决定）
报告: ~/work/rkv-replication/REPORT.md
