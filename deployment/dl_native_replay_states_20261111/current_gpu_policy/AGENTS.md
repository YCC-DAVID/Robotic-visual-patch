# GPU 任务分配

**用户限制：本项目禁止在 nnmc62 启动或分配任何新的训练、推理、评估、渲染、计算任务及控制器，即使该节点空闲也不可使用。资源预检可保留对 nnmc62 的只读查询；已有服务复用也不得绕过此限制。**

在本项目中，每次需要 GPU 的计算任务（包括训练、推理、评估、渲染、重启和队列中的新任务）开始前，必须先读取并执行 [gpu-preflight skill](skills/gpu-preflight/SKILL.md)，检查 nnmc60–nnmc65、nnmc71 及 nnmc76当前空闲 GPU，再根据显存、型号和卡数要求分配任务。启动前再次确认所选卡仍空闲，并显式设置 CUDA_VISIBLE_DEVICES。未知或占用的卡不可分配。

不得终止他人任务腾卡。空闲卡不够时，可以用 [gpu-mine skill](skills/gpu-mine/SKILL.md) 找出本项目自己闲置的任务并生成腾卡计划。只有用户明确批准该计划编号后，才可用 `gpu_mine.py stop` 停止；未经批准不得用任何方式停止任务。查询“我占了哪些卡、在跑什么”也用 gpu-mine 的 `status`。

已安装副本：`/home/user1/.codex/skills/gpu-preflight/SKILL.md`、`/home/user1/.codex/skills/gpu-mine/SKILL.md`。项目中的 `skills/` 为维护源，更新后同步安装副本。
