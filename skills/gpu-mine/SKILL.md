---
name: gpu-mine
description: 查看本项目（chence/WAMattack）自己在 nnmc60–65、nnmc71、nnmc76 上占了哪些 GPU、各自在干什么、是否还活跃；GPU 不够时给出“可停哪些自己闲置任务来腾卡”的计划，计划须经用户明确批准后才能执行停止。
---

集群账号 `user1` 是多人共用的，`ps -u user1` 和 `nvidia-smi` 里绝大多数进程不是本项目的。本 skill 只认**本项目的任务**：进程自身或其祖先（到登录 shell / agent 会话为止）的工作目录或命令行含 `/workspace/chence/`（`--marker` 可追加）。codex/claude/t3 等 agent 进程本身不算任务。

脚本：`python3 skills/gpu-mine/scripts/gpu_mine.py`（只用标准库；`probe.py` 经 ssh stdin 发到各节点执行，不依赖各节点独立的 `/home`）。

## 1. 查看占用（只读）

```bash
python3 skills/gpu-mine/scripts/gpu_mine.py status          # 加 --json 得机器可读结果
```

每个任务按“任务根进程”归组（同一 launcher 下的多个 GPU 进程算一个任务），列出：节点、卡号、显存、卡利用率（默认 3 次采样平均）、运行时长、命令、cwd、采样期 CPU%、最近写出的文件和距今时长、监听端口和已建立连接数、所属 systemd user unit。结论分三类：

- **ACTIVE**：卡利用率 ≥5%、CPU ≥10%、`--idle-min`（默认 60）分钟内写过文件、服务有活动连接，或刚启动不足 `--idle-min` 分钟，任一成立即算 ACTIVE。
- **IDLE**：上述信号全部安静。这类才可以作为腾卡候选。
- **UNKNOWN**：有信号测不出（卡利用率 N/A、与他人共用的卡在忙、连接数查不到），一律不进候选。

节点连不上会标 UNKNOWN，此时该节点上的自有任务看不到，要如实告诉用户。快照存到 `~/.cache/gpu-mine/last_status.json`。用户问“我占了多少卡/在跑什么”时直接跑 `status` 汇报。

## 2. 卡不够时：生成腾卡计划（不停任何东西）

先按 gpu-preflight 查空闲卡。确认不够时再运行：

```bash
python3 skills/gpu-mine/scripts/gpu_mine.py candidates --need <卡数> [--min-free-mib <每卡显存>] [--hosts ...]
```

规则：需求要在**同一节点**上满足。一张卡上的进程必须**全部属于 IDLE 的自有任务**才算可腾出；凡是有他人进程的卡都不算。脚本先算已空闲的卡，再挑最少的 IDLE 任务补足。如果已经够了，会提示无需停止；如果没有可腾的，提示不要停任何东西、向用户报告短缺。生成的计划带编号，30 分钟内有效，存在 `~/.cache/gpu-mine/plans/<id>.json`，其中记录每个目标的节点、PID 和进程指纹（启动时刻 + 命令行）。

## 3. 审批（必须）

**未获用户明确批准，不得执行 stop，也不得用 kill / systemctl 等任何方式自行停止任务。**

1. 把计划原样展示给用户：计划编号、节点、PID、卡号、命令、cwd、判为闲置的证据、停掉后能腾出哪些卡、是否足够。说清楚停掉的后果，例如服务停了以后依赖它的评测要重新启动它。
2. 用提问工具（Claude Code 用 AskUserQuestion；Codex 直接在对话里问）请用户批准这个**具体计划编号**。用户只批准一部分时，用更小的 `--need` 或 `--hosts` 重新生成计划后再问。
3. 用户没回复、回复含糊（如“看着办”）、或计划已过期，都视为未批准。之前批准过的计划不能沿用到新计划。

## 4. 执行已批准的计划

```bash
python3 skills/gpu-mine/scripts/gpu_mine.py stop --plan <id> --approved '<用户原话>'
```

- 执行前逐个重新核对指纹。PID 已退出或被复用时报 `mismatch`，不发任何信号。
- 如果任务独占一个 systemd user unit，就 `systemctl --user stop`，避免被自动重启；否则对任务根及其全部子进程发 SIGTERM，最多等 `--grace` 秒（默认 60）。
- 没退干净的会列为 survivors。要 SIGKILL 必须另外得到用户同意，再加 `--kill-after` 重跑新计划。
- 计划只能执行一次；结果写回计划文件，并追加到 `~/.cache/gpu-mine/audit.jsonl`。
- 停完后**重新跑 gpu-preflight** 确认卡已释放，再按 preflight 流程启动新任务。

## 注意

- 这里的审批是流程约束，不是技术锁：脚本无法验证 `--approved` 里的话是不是用户本人说的。代理必须如实引用用户的原话。
- 只停本项目的任务。他人任务即使闲置也不提议、不停止。
- 调度器、等卡启动器等可能在你停掉子任务后又往卡上派任务。计划里的“任务根”已经尽量上溯到启动它的脚本；如果根是调度器，提议时要说明停掉它等于暂停整个队列。
- 渲染（EGL）进程在 nvidia-smi 里是 G 类型，只占少量显存、卡利用率很低，主要靠 CPU% 判活。
