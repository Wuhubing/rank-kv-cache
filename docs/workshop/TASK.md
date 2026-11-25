# 交接任务书：score-EMA KV 页缓存研究的完整重跑（replication-v2）

> 放置位置：mini 的 `~/work/rkv-replication/TASK.md`。完整计划在仓库 `docs/workshop/rerun_plan.md`（`docs/workshop-readiness` 分支），本任务书是它的执行入口，两者冲突时以计划为准。

## 目标
在实验室工作站 wulab2 上，把 score-EMA KV 页缓存研究从头重新采集和评估一遍，补齐计划第 5、6 节列出的指标和基线，完成 Qwen3-8B 的扩展和 batch > 1 的系统测量，最后开一个 PR 汇报结果。

## 完成标准（全部满足才算完成）
1. GitHub 上存在分支 `replication-v2`（从 `origin/efficient-policy` 开出），其中包含：
   - `docs/rerun/` 下的 `environment.md`、`s3_dev_vs_l4.md`、`s4/`（每项实验一页结论）、`s5_replication.md`、`s6_extensions.md`、`s7_systems.md`、`thesis.md`、`STATUS.md`；
   - `docs/replication_v2.md`、`docs/preregistration_8b.md`、`splits/heldout_frozen_8b.json`；
   - `results_v2/` 下各步骤的小结果文件（CSV、JSON、TXT、PNG）。
2. 两份预注册（`docs/replication_v2.md` 和 `docs/preregistration_8b.md`）的提交时间，都早于对应 held-out 第一篇 trace 的采集时间（看 git 记录和 manifest 时间戳），并在 `s5_replication.md` 和 `s6_extensions.md` 里写明这两个时间。
3. GitHub 上有一个打开的 PR：`replication-v2` → `efficient-policy`，描述里逐条对照计划第 1.2 节的 thesis 写明哪些被证据支持、哪些被改弱或删除，并列出所有偏离。PR 不合并。
4. `~/work/rkv-replication/REPORT.md` 写好：每一步的结论、用了哪张卡（编号和 UUID）、各阶段耗时、占用的磁盘、所有偏离。
5. wulab2 上没有残留的 `wb-rkv-` tmux 会话和本任务的 GPU 进程（`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` 确认）。
6. `~/work/rkv-replication/STATUS.md` 第一行是 `DONE`。

## 约束
- 遵守 `~/work/CLAUDE.md` 的全部规则，以及计划第 2 节和第 10 节。
- 两处由用户决定、覆盖 `CLAUDE.md` 的地方：
  1. **不上传 Google Drive**（覆盖第 11 条）。原始 trace 只放在 wulab2 的 `/data/weibing/projects/rank-kv-cache/data/`。
  2. **无人值守**：`CLAUDE.md` 里"先停下来问用户""等用户决定"的地方，一律改为在 `STATUS.md` 第一行写 `BLOCKED: 原因` 后停下。
- 只用 1 张卡。第一次用 `free-gpus 1` 选定后固定使用；tmux 会话名以 `wb-rkv-` 开头。
- 不改冻结文件；原有 held-out 只在复现预注册提交之后采集、只评估一次；所有探索只用 dev。
- 不合并、不关闭任何 PR；不改 `main`、`efficient-policy` 等已有分支；不 force push。
- 代码改动只在 mini 的 `~/work/rkv-replication/repo` 里做，由 mini 提交推送；wulab2 用 `git pull` 同步代码，用 `rsync` 把小结果文件拉回 mini。

## 每一轮怎么开始
先读 `STATUS.md`、本任务书和计划第 7、10 节，从 `STATUS.md` 记录的位置接着做。会话可能被压缩上下文或在 mini 重启后被重新拉起，`STATUS.md` 是唯一可靠的进度来源。开始任何 GPU 任务前，先确认 `ssh wulab2 'tmux ls'` 里没有同名任务在跑。

## 进度记录
- `STATUS.md` **第一行只写状态**：`RUNNING: S<步骤号>`、`DONE` 或 `BLOCKED: 原因`。下面按计划第 10.4 节的格式写详细进度。
- 每完成一步更新一次，同步到仓库的 `docs/rerun/STATUS.md`，提交并推送。
- 等待 wulab2 上的长任务时，用一条阻塞的 shell 命令轮询（每 60 秒查一次，单条命令最多等约 9 分钟），不要每查一次就结束一轮。

## 什么时候停下来而不是继续
停下时把 `STATUS.md` 第一行改成 `BLOCKED: 原因`，写明试过的办法和需要人决定的问题，同步并推送后结束。
- wulab2 没有空闲卡（或选定的卡被占用），每 30 分钟查一次，等满 6 小时仍然没有。
- 同一个检查修复后重试 3 次仍不通过。
- 唯一的修法会违反规则（例如要改冻结文件、要重新评估 held-out、要用多张卡）。
- `/data` 剩余空间少于 200 GB。
- wulab2 上出现 `/data/weibing/projects/rank-kv-cache/STOP` 文件：不开始新任务，等正在跑的任务结束后停下（第一行写 `BLOCKED: 按停止开关暂停`）。

结果不符合预期（例如 EMA 在新 trace 上不再胜过 LRU）**不算**停下的理由，照实记录并继续。
