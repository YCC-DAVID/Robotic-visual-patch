---
name: gpu-preflight
description: 检查 nnmc60–nnmc65、nnmc71 及 nnmc76 的空闲 NVIDIA GPU，并在每次 GPU 训练、推理、渲染、评估或计算任务启动和资源分配前执行资源预检。
---

每次启动需要 GPU 的新任务、重启任务或为队列分配 GPU 前使用此 skill。检查仅查询资源，不启动计算、不停止现有任务。

**用户明确禁止在 nnmc62 运行本项目任务。** 全节点只读预检仍包含 nnmc62，但该节点的 GPU 一律 `policy_allowed=false`、`eligible=false`，即使物理 `idle=true` 也不可分配。不得通过指定节点、CLI 覆盖、已有服务复用或新控制器绕过此限制。

1. 执行本 skill 下的 `scripts/check_gpus.py`（Python 3 标准库，无额外依赖）：
   ```bash
   python3 /home/user1/.codex/skills/gpu-preflight/scripts/check_gpus.py
   ```
   需要机器可读结果时加 `--json`；按任务显存需求加 `--min-free-mib N`。默认检查 nnmc60 到 nnmc65、nnmc71 及 nnmc76（型号和显存以实时查询为准），当前节点本地查询，其余使用当前用户的 SSH 配置及非交互认证。
2. 默认空闲标准：没有计算进程、显存占用不超过 256 MiB、GPU 利用率为 0%。UNKNOWN（连接失败、超时、命令错误或无法解析）不能分配；单卡进程或利用率为 `[N/A]` 时仅该卡标为 UNKNOWN(faulty)，同节点其余卡照常判断（2026-11 实测 nnmc62 GPU5、nnmc64 GPU3 即此类坏卡，CUDA 不可用）。非零退出码表示至少一个节点未知，可使用其余已成功检查且符合条件的节点。不要为取得空闲卡而清理或结束他人进程。阈值调整需明确解释原因，不能把已占用的卡改称空闲。
3. 根据任务所需显存、卡数及型号选择符合条件的卡；多卡任务优先同节点。检查项目记录的硬件故障或保留卡限制（例如 HANDOFF.md 中 nnmc63 GPU0 的历史 ECC），空闲不代表硬件健康。没有足够资源时，可用 gpu-mine skill 的 `candidates` 生成腾卡计划，只涉及本项目自己闲置的任务；计划须经用户明确批准才能执行 `stop`。否则报告短缺并等待用户指示，不擅自挤占。
4. **启动前立即再次查询所选节点**（`--hosts nnmcXX`），确认全部所选卡仍符合要求。快照不构成资源锁；集群若有调度/预留机制，应通过该机制取得资源。不要声称此脚本提供原子预留。
5. 启动任务时显式设置 `CUDA_VISIBLE_DEVICES`（优先使用所选 GPU UUID），记录节点、物理卡号/UUID、检查时间、任务及 PID/日志。CUDA 可见设备编号会重新从 0 开始；任务内部参数应使用映射后的编号。并发分配应串行确认，避免给多个任务重复分配同一张卡。
6. **EGL 渲染不能假设 EGL 编号等于 NVIDIA 物理卡号。** 启动渲染前，在最终 `CUDA_VISIBLE_DEVICES` 下查询原生 EGL 设备的 `EGL_CUDA_DEVICE_NV`，再用 CUDA driver UUID 核对它确实指向预检分配的卡；同时记录物理号、UUID 和原生 EGL 号。仅查询映射不需要创建渲染/计算上下文。2026-11-05 实测 nnmc71：物理 GPU3（UUID `GPU-b3371e7f-1ad4-c399-8b02-dd0287aa7c71`）对应原生 EGL0，直接设置 EGL3 会选错卡；该映射每次分配仍须复核。项目提供 CPU 核对工具 `pi05probe/egl_device_audit.py --uuid <UUID> --egl-index <原生EGL号>`（环境中 `CUDA_VISIBLE_DEVICES=<UUID>`）。旧 robosuite 对可见卡字符串有过时断言，不能通过额外暴露未分配卡来绕过；应在保持仅所选 UUID 可见的条件下核实兼容性，无法保证映射则不启动。

用户只要求检查时，返回空闲卡及未知节点，不启动计算。检查结果是即时快照，每次新分配都必须重新执行，不能沿用旧结果。
