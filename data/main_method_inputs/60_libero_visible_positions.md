# LIBERO：按任务选择可见位置，分别训练 render 代理 patch 与 EOT patch

2026-11-06 10:34 PST 更新：**7/7 全量扫描与选点完成**，所有缓存经源码哈希、全训练回合独立遮罩校验门禁复核有效。旧 H100 控制器在 11月5日14:30 PST 首个标定帧触发 cached-clean 背景不等断言后退出，未开始训练。本次在 nnmc64 空闲 L40S 上用相同状态核对 3 个回合：原生 clean 与缓存、manual/native、贴纸外背景、静态物体位姿均逐像素/逐值一致，无需降低校验标准。

已恢复到 nnmc64 **GPU0/EGL2、GPU1/EGL1**，最终控制器 **4116526**。碗放盘子与碗放柜顶两项标定已完整保存，通过每帧背景相等检查；CPU 代理核对的两相机最坏归一化 RGB RMSE 分别为 0.002356/0.014541 和 0.002354/0.007041，均低于0.02。两项完成输出后曾因旧池缺少 outcomes、完成日志遍历空值而退出，数据文件有效，已修复完成日志并补充 sidecar 读取；同时修复负 x,y 位置的命令行解析。当前复用这两份标定，继续余五项，本轮纹理训练仍 **0/14**。任务资源检查现串行执行，全节点范围含 nnmc76，EGL/UUID 每次核实。科学协议、帧、种子和预算不变。

诊断：[背景回放检查](../pi05probe/out/visible_pos/recovery/background_check_nnmc64/results.json)；有效进度与 PID：[progress.json](../pi05probe/out/visible_pos/progress.json)、[latest_restart.json](../pi05probe/out/visible_pos/latest_restart.json)。

2026-11-05 10:05 PST 最终恢复：控制器 **4065996**、打开炉子扫描 **4066153**，均运行于 nnmc71 物理 GPU3。第一次迁移启动的 EGL 编号错误（EGL3 不等于 CUDA3），触发 RGB 自检后退出，未产生有效扫描。已核实物理 GPU3 的 UUID 对应原生 **EGL0**，模拟器设置 `CUDA_VISIBLE_DEVICES=GPU-b3371e7f-1ad4-c399-8b02-dd0287aa7c71`、`MUJOCO_EGL_DEVICE_ID=0`，每次分配在所选节点复查前增加无上下文的 EGL→CUDA UUID 核对。修正后 3 回合×3 帧×2 相机，每帧3次重复渲染均与原生观测逐像素相等；正式扫描已启动，GPU3 实测占用471MiB、利用率11%。前三份扫描源码和有效缓存保留。自检：[results.json](../pi05probe/out/visible_pos/recovery/nnmc71_render_check/results.json)；最新进程：[latest_restart.json](../pi05probe/out/visible_pos/latest_restart.json)。

2026-11-05 10:00 PST 更新：已迁移恢复到 nnmc71 物理 GPU3（H100 NVL，UUID `GPU-b3371e7f-1ad4-c399-8b02-dd0287aa7c71`），控制器 PID 4058731，打开炉子扫描 PID 4058856。用户指定卡在首次检查时有进程占用，18:00 UTC 复查已释放（1MiB、0%、无计算进程），控制器又独立执行全节点及所选节点预检后启动。前三个任务扫描（碗放盘子、柜顶、酒瓶）完整校验通过，直接复用；余四项继续。原 nnmc63 GPU6 在 11月4日20:35 PST 开始下一项前被占用，旧控制器已安全退出，当前没有原流程在并行运行。科学协议及预算未变。启动记录：[latest_restart.json](../pi05probe/out/visible_pos/latest_restart.json)。

2026-11-04 16:18 PST 已恢复，当前完整扫描中，尚无本轮训练或闭环结果。实时阶段见 [progress.json](../pi05probe/out/visible_pos/progress.json)，启动记录见 [latest_restart.json](../pi05probe/out/visible_pos/latest_restart.json)。

用户要求：7 个任务分别选位置，再各自进行单任务 render 代理训练，对照 nominal 与 texture-plane EOT。沿用同一 pi05_libero checkpoint，本轮每个任务独立训练纹理。

原定协议保持：13cm patch；训练初态 0–9，验证 10–11；测试 40–49 不参与选点。470 个合法候选位置中，先要求至少 90% 训练帧的主相机可见像素数 >=300，再选腕部平均可见像素最多的位置。每任务 nominal/EOT 各 500 步×batch 8，共 14 张纹理；评测 clean/random/nominal/eot，共 280 条真实 MuJoCo 闭环。[原协议](../pi05probe/out/visible_pos/protocol.json)

三项补采 clean 池已经完成：碗放炉子 12/12、奶酪放碗 12/12、打开中间抽屉 11/12 成功；其余四项复用 #59 clean 池。

旧控制器已在选点断言处退出（`put_the_bowl_on_the_plate` 无合格位置），没有进入训练。旧扫描不可信，逐任务归档后重算。

故障原因：扫描循环把旧 `ctx/sim/model/data` 引用保留到下一次环境 reset 之后。新上下文创建后，旧渲染对象才析构，MuJoCo/robosuite 的 GL 资源释放可能破坏新上下文。修复把每帧渲染引用限定在单独函数内，保证它们在下一次 reset 前释放。原有相机名到 ID 的查询已经正确。

修复实测：碗放盘子训练池的 10 个回合，每回合首/中/尾，共 30 帧×40 个分散位置。扫描后选 3 个位置，每回合首尾两帧、两路相机，与将贴纸直接写在该位置的独立新场景比对。120 份单相机遮罩（60 个帧位置×两路相机）全部像素一致，可见面积差和完整 224 遮罩差均为 0。重置后的直接 RGB 与环境原生观测一致。[冒烟校验](../pi05probe/out/visible_pos/recovery/smoke_bowl/visibility.json)

每份完整扫描都必须通过独立场景全遮罩比对，覆盖每个训练回合。缓存要求新 schema、当前扫描源码 SHA256、完整扫描标志、维度一致和所有差异为 0；旧缓存、冒烟缓存、过期源码、缺回合、空校验或像素不一致均拒绝选点。CPU 正反例检查与三个修改脚本的语法检查通过。

训练前另加代理核对：随机纹理经代理与真实渲染，在采样帧贴纸内部的归一化 RGB RMSE 必须 <=0.02（约 5/255），超限则停止，不做纹理更新。该阈值是本次新增的实现完整性门禁，选点规则和训练预算未变。

本次完整流程运行于 nnmc63，物理 GPU6，UUID `GPU-6ae322e7-1c38-6d43-cb8e-32c002dfeec4`。控制器 PID 2852756，首个扫描 PID 2852888。每个新任务均执行全节点预检与所选节点立即复查，同一 GPU 上任务串行；server 与闭环模拟组成一次任务分配。模拟器显式设置数字 `CUDA_VISIBLE_DEVICES=6` 与物理 EGL 设备号，模型进程使用 UUID。占用、未知或故障卡均拒绝分配，资源不可用时记录 blocked 并退出。

日志：[pipeline.log](../pi05probe/out/visible_pos/pipeline.log)；GPU 预检快照、任务 PID/命令/日志记录：[resources](../pi05probe/out/visible_pos/resources)。旧日志与恢复源码保留在 [recovery](../pi05probe/out/visible_pos/recovery)。完整流程耗时尚无可靠实测估计。
