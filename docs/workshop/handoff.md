# 交给 mini 执行：启动、查看和介入

依据 `mac-mini-guide.md` 第 4 节"方式 D"（后台会话 + `/goal`）。所有命令都在 MacBook 上执行，通过 SSH 在 mini 上运行。

## 1. 启动前（只做一次）

**1.1 确认没有同名会话在跑**（同一个任务发两次会起两个会话同时干活）：

```bash
ssh ww@100.125.41.1 '~/.local/bin/claude-headless agents --json'
```

列表里不能有名为 `rkv-replication` 的会话。

**1.2 拒绝规则（2026-11-25 已加上）。** 无人值守必须用 `claude-bypass`，它跳过所有确认，只有 `~/work/.claude/settings.json` 的 `deny` 规则还会拦截。`CLAUDE.md` 和任务书只是给 agent 的指示，不是拦截。建议在 `deny` 里加上：

```json
"Bash(gh pr merge:*)",
"Bash(gh pr close:*)",
"Bash(gh repo delete:*)",
"Bash(gh api -X DELETE:*)",
"Bash(git push --force:*)",
"Bash(git push -f:*)",
"Bash(git push origin main:*)",
"Bash(git push origin efficient-policy:*)",
"Bash(git push origin --delete:*)",
"Bash(rclone delete:*)",
"Bash(rclone purge:*)"
```

这些规则按命令前缀匹配，挡不住写在 `ssh wulab2 '…'` 里面的远程命令，所以 wulab2 上的安全仍然靠 `CLAUDE.md` 的规则。它们不影响任务需要的操作：推送 `replication-v2`、`gh pr create` 都照常可用。

已于 2026-11-25 加进 mini 的 `~/work/.claude/settings.json`，修改前的文件备份为同目录的 `settings.json.bak-20261125`。

**1.3 告诉管理员。** 指南建议：mini 自动登录 wulab2 长时间跑任务，最好和管理员（`yining`）或组里说一声。这次只用 1 张卡，但可能连续跑好几天。

**1.4 准备运行目录和任务书：**

```bash
ssh ww@100.125.41.1 'mkdir -p ~/work/rkv-replication && cd ~/work/rkv-replication && git clone https://github.com/Wuhubing/rank-kv-cache.git repo && git -C repo fetch origin docs/workshop-readiness && git -C repo show origin/docs/workshop-readiness:docs/workshop/TASK.md > TASK.md && printf "RUNNING: S0\n" > STATUS.md && ls'
```

## 2. 启动

```bash
ssh ww@100.125.41.1 'cd ~/work && ~/.local/bin/claude-bypass --bg -n rkv-replication "/goal rkv-replication/TASK.md 里的完成标准全部满足，或者 rkv-replication/STATUS.md 第一行以 BLOCKED 开头，或者已经做满 300 轮。先读 rkv-replication/TASK.md、rkv-replication/STATUS.md 和 CLAUDE.md，再读仓库 rkv-replication/repo 里 origin/docs/workshop-readiness 分支上的 docs/workshop/rerun_plan.md，然后照任务书执行。"'
```

- 成功会输出 `backgrounded · 会话编号 · rkv-replication`。发出后可以合上 MacBook。
- **必须从 `~/work` 启动**，`settings.json` 和 `CLAUDE.md` 只对从 `~/work` 启动的会话生效。
- 轮数上限设为 300。指南里只验证过 40 轮以内的小任务，这个任务大得多。如果因为做满 300 轮而停下（`STATUS.md` 第一行还是 `RUNNING`），先 `rm` 掉旧会话，再用同一条命令重新启动，它会从 `STATUS.md` 接着做。

## 3. 查看进度

| 想看什么 | 命令 |
|---|---|
| 进度（最直接） | `ssh ww@100.125.41.1 'cat ~/work/rkv-replication/STATUS.md'` |
| 会话状态（`waiting` 表示卡在等人） | `ssh ww@100.125.41.1 '~/.local/bin/claude-headless agents --json'` |
| wulab2 上在跑什么 | `ssh wulab2 'tmux ls; nvidia-smi --query-compute-apps=pid,used_memory --format=csv'` |
| 在 GitHub 上看 | `replication-v2` 分支的 `docs/rerun/STATUS.md` 和提交记录 |

## 4. 介入

| 情况 | 做法 |
|---|---|
| 想纠正方向 | `ssh -t ww@100.125.41.1 '~/.local/bin/claude-headless attach rkv-replication'`，对话后按 `←` 退出（不中断） |
| 想暂停（让正在跑的任务跑完，不再开新任务） | `ssh wulab2 'touch /data/weibing/projects/rank-kv-cache/STOP'`；恢复时删掉这个文件并重新启动会话 |
| 立即停止 | `ssh ww@100.125.41.1 '~/.local/bin/claude-headless stop rkv-replication'`，再到 wulab2 上确认 `wb-rkv-` 开头的 tmux 会话是否要关 |
| mini 重启后 | `ssh ww@100.125.41.1 '~/.local/bin/claude-headless respawn <会话编号>'`（会话编号在 `agents --json` 里） |
| 停在 BLOCKED | 看 `STATUS.md` 里写的原因和问题，处理后把第一行改回 `RUNNING: S<步骤号>`，`rm` 旧会话后用同一条命令重新启动 |
| 结束后清理 | `ssh ww@100.125.41.1 '~/.local/bin/claude-headless rm rkv-replication'` |

## 5. 指南里还没验证过的（这次会第一次碰到）
- 跑几小时到几天的长任务。
- 会话被压缩上下文后，能否靠 `STATUS.md` 接着做。任务书已要求每轮先读 `STATUS.md`。
- mini 重启后 `respawn` 的实际表现。
- 没有空闲卡时是否会按规则等待和停下。

前几个小时建议隔一段时间看一下 `STATUS.md` 和 `agents --json`，确认它在按步骤推进，而不是卡在 `waiting` 或重复同一件事。
