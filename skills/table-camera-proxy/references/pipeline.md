# 桌面—相机几何估计与成像代理 pipeline

本流程把现有视频中的已知尺寸物体，转换成“米制桌面坐标 → 各相机图像像素”的映射。**默认目标只是桌面相对相机的几何关系**，打印标定、IK 和完整三维重建可以后置。patch 投影与操作相关区域重叠时，需要在合成阶段处理遮挡。

所有代码路径相对 `WAMattack` 根目录。适配器支持 G1 `simple_pick` 的单帧三视角，以及参考物保持静止的短窗口联合估计。已运行 episode 18/frame 0 单帧与 frame 0–19 多帧；多帧方法及边界见第 8 节。

## 1. 输入、已知量与假设

| 输入 | 当前文件或数值 | 作用 |
|---|---|---|
| 三路同一时刻的图像 | `realprobe/simple_pick/table_proxy_v0/frames/` | 共享场景的几何观测 |
| 图像角点及物理点 ID | `realprobe/simple_pick/table_proxy_v0/annotations.json` | 各相机的 2D—3D 对应；当前是助手近似目视标注 |
| 方块边长 | `scene_geometry.json` 中 `0.052 m` | 建立米制尺度与非共面角点 |
| 容器已知边 | 同文件中后部上沿外边 `0.105 m` | 辅助多视角约束；只确定该边长度 |
| 粗内参先验 | `annotations.json` 每路 `focal_prior_px` | 初始化并约束欠确定方向 |
| 本地视频 | `realprobe/simple_pick/dataset/videos/chunk-000/` | 单帧可替代缓存 PNG；多帧必须提供 |
| RoboTwin 灰阶捕获，可选 | `robotwinprobe/out/render_transfer/calibration/seed10100.npz` | 成像步骤借用的五点响应 |

默认尺寸文件：`realprobe/simple_pick/scene_geometry.json`。只有后三个相机名称及当前图像尺寸可直接运行：

```text
cam_left_high       头部，640×480
cam_left_wrist      左腕，640×480
cam_right_wrist     右腕，640×480
```

当前假设：方块在选定帧直立并接触桌面；绿色后部上沿平行于桌面；三路名义同帧对应同一静止场景；相机主点居中、横纵焦距相同、无畸变。时间同步和这些先验都不是已实测确认的事实。

多帧还要求方块和容器在整个窗口保持静止、每路相机内参不变。相机本身可以移动；头部也不强制静止。

## 2. 共享坐标与角点

以方块**底部前左角**为原点，X 向右、Y 向后、Z 向上；桌面 `Z=0`，边长记为 `s`。方向按物理方块定义，不随相机画面改变。

| 点 ID | 含义 | 三维坐标 |
|---|---|---|
| 0 | 上部后左 | `(0,s,s)` |
| 1 | 上部后右 | `(s,s,s)` |
| 2 | 上部前右 | `(s,0,s)` |
| 3 | 上部前左 | `(0,0,s)` |
| 4 | 底部后左 | `(0,s,0)` |
| 5 | 底部后右 | `(s,s,0)` |
| 6 | 底部前右 | `(s,0,0)` |
| 7 | 底部前左 | `(0,0,0)` |
| 8、9 | 后部上沿外边两端 | 位置和方向待估，两点距离固定为已知边长 |

遮挡或看不见的点不填假坐标。换帧时要检查相同 ID 的物理一致性；单色方块各面外观相近，单凭各视图的四角排序可能相差 90°。

`annotations.json` 的接口为：

```text
episode, frame, image_size_wh
cube_order, rim_order, assumptions
views[camera].focal_prior_px
views[camera].points[point_id] = [u_px, v_px]
views[camera].holdout_ids = [不进入验证拟合的点 ID]
```

原图像素坐标以左上为起点，u 向右、v 向下。当前点是在原始 640×480 上标注；不能直接用于缩放、裁切后的图像。完整实例保存在已有 `annotations.json`，不在 skill 中复制易失的角点数据。

## 3. 联合几何拟合

实现：`realprobe/build_table_proxy.py`。

1. 为三路相机分别初始化焦距与位姿，使用已知立方体的 PnP 候选，优先选择正深度、位于桌面上方的解。
2. 联合优化三路相机及容器上沿位置。每路参数是旋转向量 3、平移 3、log 焦距 1，共 21 个；上沿有 XYZ 位置和水平 yaw 共 4 个，总计 25 个参数。
3. 最小化全部可见点的重投影误差，加入焦距软先验及几何有效性惩罚。当前用 `soft_l1`，方块观测权重对应 1.5 px、上沿对应 2 px。它们是本轮优化设置，并非观测误差的实测标准差。
4. 导出每路的内参、相对位姿、相机中心和桌面单应矩阵。

约定为：

```text
X_camera = R @ X_table + t
camera_center_in_table = -R.T @ t
H = K @ [R[:,0], R[:,1], t]
pixel_homogeneous ~ H @ [X_m, Y_m, 1]
```

固定已知边长解决尺度；使用立方体非共面点和底部接触关系，把顶面与桌面区分开。之前的单面拟合没有充分使用这些关系，因此焦距和相机距离的耦合更明显。

此处不需要知道桌面相对 pelvis 的坐标。若以后要用 EEF 推算腕部相机运动，再补手眼和桌面—pelvis 关系。

## 4. 检查几何是否可用于粗投影

自动检查与诊断分别处理：

| 检查 | 如何做 | 能说明什么 |
|---|---|---|
| 优化与矩阵有效性 | 收敛状态、有限参数、旋转正交且 det=1、相机在桌面上方 | 实现与基本几何一致性 |
| H 与完整三维投影 | 同一组 `Z=0` 点分别计算两种投影 | 坐标方向、米／像素约定无混用 |
| 输入哈希 | 核对标注与实际 PNG 的哈希 | 输出对应本次输入 |
| 已标点重投影 | 逐点误差及叠图 | 当前观测的拟合程度 |
| 留出角点 | 排除指定点重新优化，直接预测被排除点 | 同帧几何诊断；不是新图像验证 |
| 焦距先验敏感性 | 改变先验强度，看固定桌面 patch 投影变化 | 对选定设置的敏感性；不是置信区间 |
| 多视角纹理检查 | 同一个非对称棋盘格在三路投影 | 内容透视、朝向与世界位置是否一致 |

留出角点没有再用于最终验证位姿优化，但当前脚本的 PnP 初始化仍读取全部可见方块点。应称为**同帧排除残差诊断**，不作为严格独立的盲验证。需要严格验证时，初始化也应排除这些点，或使用独立帧／目标。

实现检查失败时修正输入或实现并在新目录复跑。实现通过但拟合或叠图可疑时，保留粗估结果和具体问题；不能仅凭低 RMS 宣称完整标定完成。

## 5. 可选成像与模糊诊断

`--with-imaging` 调用 `realprobe/table_render_proxy.py`：

```text
共享世界纹理 Q
  → 各相机桌面 UV
  → sRGB 解码 + 双线性采样
  → 4×4 子像素积分
  → RoboTwin 五点响应先验
  → 合成到 640×480 背景
  → 面积缩放到客户端 320×256
```

响应取 RoboTwin seed10100 主相机可见 patch 区域的空间中位数，三路暂时共用；它不是 G1 的实测颜色标定。几何、颜色响应、图像链路分开存储。

实现检查包含纹理梯度、有限差分、patch 外背景保持和 OpenCV `INTER_AREA` 一致性。JPEG quality 90 只保存为预览；当前前向不包含 JPEG 或未确认的策略服务端预处理。

**操作区域重叠时的遮挡要求：**各相机、各帧检查投影与机械臂、夹爪、被操作物体及目标容器是否重叠；按前后关系保留前景，仅在可见桌面合成 patch。可先用人工确认或分割的前景 mask，随后再完善深度关系，并记录可见面积／比例及时间变化。无遮挡区域继续使用现有平面映射，不要求先搭完整 3D 场景。

第 19 页的报告图已使用 `realprobe/render_table_occlusion_preview.py --source-dir <20帧运行目录>` 生成局部遮挡版，保存在该目录的 `occlusion_preview/`。该适配器仅覆盖 episode 18、frame 0／5／10／19、6 cm patch 中心 `(-0.09,0.15) m`。新增训练适配器覆盖 0–19 帧及中心附近 2 cm／±30°，见第 10 节；通用轨迹和任意位置仍未验证，不能直接照搬局部阈值或把旧无挡图当正确遮挡结果。

`--with-edge-blur` 调用 `realprobe/estimate_edge_blur.py`，沿现有方块轮廓拟合 erf 边缘曲线。输出有效模糊宽度，**不自动写入成像参数**。换帧或换目标时，要先核对该脚本中 `EDGES` 对应的可见轮廓；此适配器尚不是通用边缘检测器。

## 6. 一条命令复现

当前可用解释器：`/home/user1/miniconda3/envs/openpi-server/bin/python`。几何依赖 NumPy、SciPy、OpenCV、PyAV、Pillow；成像额外依赖 PyTorch。pipeline 自身不下载数据、不加载策略权重，也不运行机器人。

在 WAMattack 根目录，先查看计划；这一步不创建输出文件：

```bash
/home/user1/miniconda3/envs/openpi-server/bin/python realprobe/run_table_proxy_pipeline.py \
  --out realprobe/simple_pick/pipeline_runs/table_geometry_01 --plan
```

只估桌面与三路相机关系：

```bash
/home/user1/miniconda3/envs/openpi-server/bin/python realprobe/run_table_proxy_pipeline.py \
  --out realprobe/simple_pick/pipeline_runs/table_geometry_01
```

包含成像和边缘诊断的完整复现，使用另一个新目录：

```bash
/home/user1/miniconda3/envs/openpi-server/bin/python realprobe/run_table_proxy_pipeline.py \
  --out realprobe/simple_pick/pipeline_runs/table_proxy_01 \
  --with-imaging --with-edge-blur
```

联合估计从标注帧开始的前 20 帧，先查看同一命令加 `--plan` 的计划：

```bash
/home/user1/miniconda3/envs/openpi-server/bin/python realprobe/run_table_proxy_pipeline.py \
  --out realprobe/simple_pick/pipeline_runs/table_first20_01 --num-frames 20
```

短窗口入口目前只做几何，不接受 `--with-imaging` 或 `--with-edge-blur`。`geometry.json` 是多帧拟合的首帧兼容导出；逐帧矩阵在 `geometry_sequence.json`，不能把兼容文件当作整段代理。

输出目录必须不存在或为空，已有记录不会覆盖。可用 `--source-dir` 指定另一份 `annotations.json` 及 `frames/`，用 `--scene-geometry` 指定尺寸文件，用 `--data-root` 指定本地数据根目录。当前适配器仍要求相同三路命名、640×480、同样的物体点模型；这些参数不意味着任意数据集已经支持。

单帧缓存图缺失时按 `episode/frame` 从视频解码；缓存存在时优先使用缓存。多帧始终从视频解码，并记录实际源视频哈希。使用新数据前核对缓存内容与帧索引，文件名本身不证明图像来自正确帧。

## 7. 输出与状态

| 产物 | 内容 |
|---|---|
| `annotations.json`、`scene_geometry_input.json`、`frames/` | 本次输入快照 |
| `pipeline_status.json` | 阶段进度、完成或失败状态、日志位置 |
| `run_manifest.json` | 原始输入、脚本、产物 SHA-256 及解释器、UTC 时间 |
| `geometry.json` | K、T、H、相机中心、拟合与敏感性结果 |
| `geometry_checks.json` | 矩阵、投影及输入一致性检查 |
| `geometry_overlay.png` | 观测角点、拟合点及示例 patch 轮廓 |
| `imaging_prior.json`、`proxy_checks.json`，可选 | 借用的成像参数、可微性与缩放检查 |
| `three_camera_demo.png`，可选 | 原图、透视、成像近似三行对照 |
| `edge_blur_estimates.json`，可选 | 原始边缘剖面及有效模糊拟合 |
| `geometry.log` 等 | 各阶段完整运行输出 |
| `tracks.json`，多帧 | 亚像素起点、逐帧 LK 跟踪及剔除理由 |
| `fit_01_frames.json`、`fit_15_frames.json`、`fit_N_frames.json`，多帧 | 同一起点的帧数对照；N 小于 15 时自动缩短 |
| `geometry_sequence.json`，多帧 | 共享 K、每帧 T/H、低冗余标记、覆盖与拟合诊断 |
| `sequence_checks.json`、`sequence_status.json`，多帧 | 各窗口收敛、每个相机位姿及来源一致性检查 |
| `track_fit_overlay.png`、`sequence_perspective_demo.png`，多帧 | 代表帧叠图和同一物理纹理的几何投影 |

`succeeded` 表示所选程序阶段与实现检查通过，不代表标定精度、实拍效果、策略训练或攻击成功率。完整历史解释见 `result/18_g1_table_proxy.md`，早期数据检查见 `result/17_g1_geometry_audit.md`。

## 8. 短窗口联合估计与后续扩展

实现：`realprobe/fit_table_sequence.py`；统一入口 `--num-frames N`。流程为：

1. 保留首帧手工物理点 ID，尝试局部亚像素修正；修正超过 3 px 则保留原标点。首帧建立度量世界坐标，后续不重置。
2. LK 连续跟踪，筛选正反向误差 ≤0.8 px、首帧直达与逐帧累计跟踪差 ≤1.5 px、图像边界和有效性。丢失点之后持续剔除，不用旧位置冒充观测。
3. 每路相机共享一个焦距，容器边共享 XYZ 和 yaw；每路每帧有独立 6 自由度位姿。主点居中、等焦距、零畸变与单帧一致；无时间平滑、没有将头部位姿锁死。
4. 至少 4 个非共面参考点（含至少 2 个方块点）才参与该帧求解。仅 4 点的帧标记 `minimal_four_point`，内参和容器位置依靠整个窗口共享约束，不将其当作独立完整标定。
5. 联合最小化鲁棒重投影误差和焦距先验，按每相机的有效帧数平均数据项，避免仅因复制相关帧就削弱先验。分别拟合 1／min(15,N)／N 帧。导出各帧 K、T、H 和矩阵检查。

1／15／N 对照共用亚像素修正后的起点，因此不能把新 20 帧与旧手工单帧的所有差异归因于帧数。跟踪会继承初始标点的系统偏差；时间连续帧也不是独立标注。冻结首帧投影与逐帧投影的残差对照使用拟合内观测，只说明固定矩阵不适合运动相机，不是盲测精度。

当前已运行结果见 `result/19_g1_multiframe_geometry.md`。后续仍需：覆盖更长轨迹、抓起后改用静止参考、跟踪重定位、独立角点／轨迹验证；若用 EEF 传播相机位姿，再估计手眼及桌面—pelvis 关系。不要把短窗口残差直接当成几何 EOT 的部署误差分布。

## 9. 迁移到其他场景时保留的原则

保留同一物理坐标系、尺寸单位、跨视角点身份、位姿方向及数据来源记录。重新确定物体尺寸、角点、图像处理链路和内参先验；不要照搬本例相机高度、焦距或容器高度。先搭能运行的粗投影，再针对实际误差完善几何与成像。

## 10. 已接通的短窗口 patch 训练

用户已请求训练时，可运行 `realprobe/g1_patch_train.py`。它使用 `realprobe/sequence_render_proxy.py`，而非把单帧 `TableProxy` 外推到整段轨迹。现有适配器固定 episode 18/frame 0–19：0–13 训练、14–19 评估，6 cm 共享纹理，桌面中心 `(-0.09,0.15) m`，均匀圆盘位移半径 2 cm、旋转 ±30°。这些是本轮试验设置，不是换场景时的普遍默认值。

前景 mask 从 clean RGB 分割绿色容器、粉色物体和暗色夹爪，填充轮廓并外扩 1 px。当前 EOT 支持区域已看过边界投影；左腕大量遮挡是真实视角限制，应保留。换中心、尺寸、范围或轨迹后重新做遮挡检查。代理按投影包围框采样，不为无限多个随机位姿缓存全图网格。

成像包含原 RoboTwin 先验，以及客户端 640×480→320×256 的 AREA 缩放／JPEG90、服务端 PIL 双线性 letterbox224。离散前向使用实际操作，反向用缩放直通近似；JPEG 不是可微算子。旧单帧 API 仍仅到 client resize，不能把新增链路误写成旧实现自带。

模型依赖用户提供的 `realprobe/third_party/openpi-fintune`，commit `29030046fd6a2810201db67b9804f243e0af3218`。EEF6D transform 已验证四元数16→旋转6D20，quantile normalization 后补零至32；动作长度采用 checkpoint 的32。当前仓库 G1 配置 `discrete_state_input=False`，历史 simple-pick 完整配置未保留，须记录这一依据与待确认项，不应悄悄使用默认 pi0.5 的 True。代码依赖的 pytest 已隔离安装到 `realprobe/runtime_py`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/user1/miniconda3/envs/openpi-server/bin/python realprobe/sequence_render_proxy.py \
  --validate realprobe/simple_pick/new_validation

CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/user1/miniconda3/envs/openpi-server/bin/python realprobe/g1_patch_train.py \
  --out realprobe/simple_pick/new_training --iters 500 --batch 4
```

GPU 编号以实际空闲情况为准；当前 Codex 受限执行环境看不到设备，已授权 GPU 作业需通过可访问 CUDA 的执行环境运行。训练目录不可复用；读取 `progress.json`、`failure.json`、`train.log` 判断实际状态。训练结束自动生成 Random／gray 对照及新 noise 评估。`result/tools/build_g1_patch_eot_report.py --run <训练目录>` 生成曲线与逐位姿统计，使用具备 matplotlib 的解释器。

模型冻结，只更新 Q；L0 是固定同状态、同 noise 的首步速度平方差。留出帧与训练帧时间上相关，几何又使用全部20帧拟合，因此不能称为跨轨迹验证。没有闭环时不报告 ASR；没有同预算无 EOT 对照时不宣称 EOT 的因果收益。运行说明见 `result/20_g1_patch_eot.md`。
