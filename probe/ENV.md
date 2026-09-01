# ENV.md — 环境、版本、已知坑

本文件记录**实际跑通用到的**版本与 commit，以及踩过的坑。复现和排错以此为准。

最后更新：2026-07-30
运行主机：`nnmc65` (10.145.87.65)，8× NVIDIA L40S (46 GB)，driver 580.159.03

---

## 1. 结论速查

| 项 | 值 |
|---|---|
| conda env | `wamattack`（Python 3.10） |
| FastWAM | `github.com/yuantianyuan01/FastWAM` @ `45d8e1458921d83f8ad6cf9ce993d371208dabd0` |
| LIBERO | `github.com/Lifelong-Robot-Learning/LIBERO` @ `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| mujoco | 3.3.2（必须与 LIBERO 数据版本一致，见 §5.1） |
| robosuite | 1.4.0 |
| torch | 2.7.1+cu128 / torchvision 0.22.1+cu128 |
| 渲染后端 | `MUJOCO_GL=egl`（headless 可用，见 §5.2） |
| LIBERO ckpt | `checkpoints/fastwam_release/libero_uncond_2cam224.pt`（12,041,735,140 B） |
| 是否有 history conditioning | **无**，单帧输入（§4.1） |
| action head 是否随机 | **有**，flow matching；但 **ε 可外部固定** （§4.2） |
| 模型输入图 | 224×448（两路相机各 224×224 **横向拼接**）（§4.3） |
| action chunk H | 32；实际执行 H_exec = `replan_steps` = **10**（§4.4） |

---

## 2. 目录布局

```
WAMattack/
├── probe/                    # 本阶段产出（见项目 spec 第 2 节）
├── checkpoints/
│   └── fastwam_release/
│       ├── libero_uncond_2cam224.pt                 # 12 GB
│       └── libero_uncond_2cam224_dataset_stats.json # 40 KB
└── third_party/
    ├── FastWAM/             # pinned 45d8e14
    └── LIBERO/              # pinned 8f1084e
```

## 3. 安装步骤（实际执行的）

按 FastWAM README 的 Environment Setup，独立环境，不复用旧环境（原因见 §5.3）：

```bash
conda create -n wamattack python=3.10 -y
conda activate wamattack
export PYTHONNOUSERSITE=1          # 关键，见 §5.3
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e third_party/FastWAM
pip install -e third_party/LIBERO
pip install mujoco==3.3.2 robosuite==1.4.0
pip install huggingface_hub==0.29.2 hf_transfer
```

权重（公开，无需 token）：

```python
from huggingface_hub import hf_hub_download
for f in ["libero_uncond_2cam224.pt", "libero_uncond_2cam224_dataset_stats.json"]:
    hf_hub_download("yuanty/fastwam", f, local_dir="./checkpoints/fastwam_release")
```

## 4. FastWAM 模型事实（spec 第 6 节要回答的）

全部直接读代码得到，未猜测。文件相对 `third_party/FastWAM/`。

LIBERO 走的是 `FastWAM.infer_action()`（`src/fastwam/models/wan22/fastwam.py:906`），
不是 `infer()`/`infer_joint()` —— 即"不做 test-time future imagination"那条路径。

### 4.1 无 history conditioning —— 确认单帧

`infer_action` 显式校验输入只能是单张图（`fastwam.py:928-933`）：

```python
if input_image.ndim == 3:
    input_image = input_image.unsqueeze(0)
if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
    raise ValueError(f"`input_image` must have shape [1,3,H,W] or [3,H,W], got ...")
```

并且要求 `video_attention_mask_mode == "first_frame_causal"`（`fastwam.py:923`），
内部只算 `first_frame_latents`。代码里没有 `n_obs_steps` / `frame_stack` / deque 之类的历史缓存。
**结论：单帧，无记忆。** spec 第 6 节的三条推论成立。

### 4.2 action head 有随机性（flow matching），但初始噪声 ε 可以外部固定 ✅

`fastwam.py:952-958`：

```python
generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
latents_action = torch.randn(
    (1, action_horizon, self.action_expert.action_dim),
    generator=generator, device=rand_device, dtype=torch.float32,
).to(device=self.device, dtype=self.torch_dtype)
```

然后是标准 flow-matching 迭代（`fastwam.py:1024-1044`）：
`infer_action_scheduler.build_inference_schedule(num_inference_steps=...)`
→ 逐步 `_predict_action_noise_with_cache(...)` → `scheduler.step(pred, delta, latents)`。

- 采样器：flow matching / rectified flow，**不是** DDPM 式 diffusion。
- 去噪步数：eval 用 **10** 步（`configs/train.yaml:24` `eval_num_inference_steps: 10`；
  注意 `infer_action` 的函数默认值是 20，别被它骗了）。
- `text_cfg_scale: 1.0` 且 `infer_action` 里 `pred_action = pred_action_posi`，**没有 CFG 分支**，
  所以不存在 cfg 随机丢弃带来的额外随机性。
- 唯一的随机源就是上面那个 `torch.randn`。

**关键**：eval 脚本已经把 seed 一路传进去了
（`experiments/libero/eval_libero_single.py:402`）：

```python
"seed": None if cfg.get("seed") is None else int(cfg.seed),
```

⇒ 按 spec 第 6 节的分支，我们走"**有随机性 → 让 clean 与 patched 两次前向共享同一 ε**"这条，
不需要退化到多次采样取平均。

**对验收 5 的直接含义**（两个都要测，否则数字没有意义）：
- 固定 `seed` 重复前向：‖A − A'‖ **实测恰好为 0（逐位相同）**，见 §4.7。
- `seed=None` 重复前向：‖A − A'‖ 是采样方差，量级会大得多。**这个不是**噪声地板，
  不要拿它当显著性阈值，否则会把真实 influence 全部埋掉。

### 4.7 ⭐ 噪声地板实测 = 0，但 text encoder 不是 batch 不变的

这是目前最重要的方法论结论，直接决定后面所有 FD / influence 数字能不能信。
实测脚本：`probe/diagnose_noise_floor.py`。

**(A) 策略前向是逐位确定的。** 复用**同一个** `context` 张量、同一张图、同一个 seed，
连跑 4 次：

```
||A_1 - A_0|| = 0.000000e+00
||A_2 - A_0|| = 0.000000e+00
||A_3 - A_0|| = 0.000000e+00
bitwise identical across repeats: True     (||A_0|| = 8.056)
```

⇒ **噪声地板恰好是 0**。没有采样噪声、没有 GPU 抖动。
这对 FD 探针是最好的情形：**任何非零的 ‖ΔA‖ 都是真信号**，不存在显著性下限的问题。
不需要多次采样取平均。

**(B) ⚠️ 最大的坑：`_load_registered_model` 不调 `.eval()`，text encoder 的 dropout 是开着的。**

如果你像我一样为了省显存**单独**把 text encoder 抽出来用
（`fastwam/models/wan22/helpers/loader.py:_load_registered_model`），
注意它 **只做 `model.to(device, dtype)`，从不调 `.eval()`**。
而 `WanTextEncoder` 里有 `nn.Dropout(0.1)`（`wan_video_text_encoder.py:249`，
在 `:262, :268` 生效）。⇒ **默认是 training 模式，dropout 在跑。**

实测同一条 prompt 连编 3 次（`probe/diagnose_text_encoder.py`）：

| | 未调 `.eval()` | 调了 `.eval()` |
|---|---|---|
| 同一 prompt 重复编码 ‖Δ‖₂ | **31.70** | **0.000000（逐位相同）** |
| 占 ‖context‖ 的比例 | **103 %** | 0 % |
| 不同 prompt 之间 ‖Δ‖₂ | 33–35 | 25–27 |
| 信噪比（between / within） | **1.1×** | **∞** |

也就是说：**不调 `.eval()` 时，同一条指令编码两次得到的向量几乎是不相关的**
（差值和向量本身一样大），语义差异被 dropout 噪声完全淹没，信噪比只有 1.1。

**修法**（一行）：

```python
text_encoder = _load_registered_model(...)
text_encoder.eval().requires_grad_(False)     # ← 必须
```

FastWAM 官方路径**不会**踩到这个坑，因为 `infer_action` 开头有 `self.eval()`
（`fastwam.py:922`），会递归传播到 `text_encoder` 子模块。
**只有你把组件单独拆出来用时才会中招。**

**这个 bug 造成的假象**（记录下来，避免以后重复误判）：它让噪声地板看起来是
0.307 / 0.460 而不是 0，从而把 paraphrase / wrong_target / empty 全误判成
"与真指令不可区分"。我一度以为原因是"umT5 在 bf16 下不是 batch 不变的"，
**那个结论是错的** —— 当时的测量本身就被 dropout 污染了。bf16 的 batch
不变性问题即便存在也远小于此，未单独验证过，不要引用。

**规则（后面每一个对比实验都必须遵守）**：
1. **拆出来的任何子模块都要显式 `.eval()`**，不要假设加载函数替你做了。
   在跑对比实验前，先 `assert not module.training`。
2. **每条 prompt 单独编码（batch size 1）**，与官方 eval 一致 ——
   `infer_action(prompt=...)` 内部就是拿单个字符串调 `encode_prompt`。
3. **clean 与 patched 对比时，context 张量 encode 一次然后复用同一个对象**，
   不要在循环里重新编码。
4. 通用自检：**任何"只改一个变量"的实验，先做一次"什么都不改"的重复跑，
   确认差值恰好为 0。** 这一步花几分钟，能挡住整类静默失效 ——
   本次就是靠它才发现问题的。

### 4.3 输入图：224×448，两路相机横向拼接

- LIBERO 渲染分辨率 `LIBERO_ENV_RESOLUTION = 256`（`experiments/libero/libero_utils.py:16`）。
- 两个相机 obs key：**`agentview_image`** 和 **`robot0_eye_in_hand_image`**
  （`libero_utils.py:45-57`），正好是 LIBERO `ControlEnv` 的默认 `camera_names`。
- **预处理是旋转 180°，不是单纯上下翻转**（`libero_utils.py:47,51`）：

```python
img       = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
# IMPORTANT: rotate 180 degrees to match train preprocessing
```

  `[::-1, ::-1]` 同时翻 H 和 W。**只翻一个轴是错的**，patch 在图上的位置会左右镜像。
- 再各自 `_center_crop_resize` 到 224×224（`eval_libero_single.py:155-165`，BILINEAR + center crop），
  然后 `concat_multi_camera: "horizontal"` 沿 axis=1 拼接（`eval_libero_single.py:217-218`）
  ⇒ 最终 `video_size: [224, 448]`（`configs/data/libero_2cam.yaml:31`）。
  448 = 28×16、224 = 14×16，满足 `infer_action` 的"必须是 16 的倍数"检查。

⚠️ 对 io_schema 的含义：spec 要求 `rgb` 存"送进模型的那张图"。那张图是
**旋转 180° → center-crop-resize → 横向拼接后的 224×448**，
不是 `obs["agentview_image"]` 原图。但 `seg` / `depth` / `patch_visible_px` 必须在
**原始渲染分辨率、未旋转**的坐标系里算（因为要和 mujoco geom_id、相机内参对应）。
两套坐标系要在 schema 里分开存清楚，否则遮挡曲线和热图会错位。

### 4.4 action chunk H = 32，实际执行 H_exec = 10

- `num_frames: 33`、`action_video_freq_ratio: 4`，注释写明"32 action, 9 video frames"
  ⇒ **action_horizon = 32**（`configs/data/libero_2cam.yaml:28-30`）。
  `sim_libero.yaml` 里 `action_horizon: null`，由 data/processor 配置解析。
- **`replan_steps: 10`**（`configs/sim_libero.yaml:25`）就是 spec 里的 `exec_prefix_len` / H_exec：
  每次前向出 32 步，只执行前 10 步就重新推理。
- action dim 7 = eef_pose(6) + gripper(1)；proprio dim 8 = eef_pose(6) + gripper(2)。
- 归一化：`norm_default_mode: min/max`、`use_stepwise_action_norm: False`（全局 min/max）。
  反归一化在 `_denormalize_action`（`eval_libero_single.py:259-274`）走
  `processor.normalizer.normalizers["action"][key].backward(...)`。
- `delta_action_dim_mask: [T,T,T,T,T,T,F]` —— **前 6 维是 delta，gripper 是绝对值**。
  算 ‖A − A'‖ 时要注意这个混合语义。

### 4.5 rollout 循环的其他关键参数（`configs/sim_libero.yaml`）

| 参数 | 值 | 说明 |
|---|---|---|
| `task_suite_name` | `libero_spatial` | 默认评测套件 |
| `num_trials` | 50 | 每个 task 的 episode 数 |
| `num_steps_wait` | 30 | 开局先走 30 步 dummy action，等物体落稳 |
| `replan_steps` | 10 | = H_exec |
| `binarize_gripper` | true | gripper 输出二值化 |
| `env_num` | 1 | 单环境，非 vector env |
| `text_cfg_scale` | 1.0 | 无 CFG |
| `rand_device` | `cpu` | ε 在 CPU 上采，跨 GPU 可复现 |

dummy action = `[0, 0, 0, 0, 0, 0, -1]`（`libero_utils.py:43`）。

⚠️ `libero_utils.py:36-38` 有一条重要注释，抄在这里免得后面踩：

```python
env.seed(seed)
# IMPORTANT: seed seems to affect object positions even when using fixed initial state
```

即**即使 set_init_state 固定了初始状态，env seed 仍会影响物体位置**。
clean / patched 两条 rollout 必须用同一个 env seed，否则比较的根本不是同一个场景。

### 4.6 官方评测基线（供验收 1 对照）

`README.md` 给出的 LIBERO 评测命令（8 GPU）：

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

`MULTIRUN.task_suite_names` 默认跑 4 个套件：`libero_10 / libero_goal / libero_spatial / libero_object`，
每套件 10 个 task、每 task `num_trials=50` ⇒ 官方一轮是 40 task × 50 = 2000 episode。

manager 本身是个 **tmux 调度器**，不是 torch launcher：它按 `MULTIRUN.num_gpus` /
`max_tasks_per_gpu=2` 起 tmux pane，每个 pane 跑一个
`python experiments/libero/eval_libero_single.py task=... ckpt=... EVALUATION.task_suite_name=... EVALUATION.task_id=...`。
我们只需要单条 rollout，所以**直接调 `eval_libero_single.py`**，不要用 manager（省掉 tmux 依赖）。

⚠️ **`EVALUATION.instruction_type` 对 LIBERO 不存在** —— 它只在 RoboTwin 路径里
（`configs/sim_robotwin.yaml:21`，消费点在 `third_party/RoboTwin/script/eval_policy.py`）。
LIBERO 的指令永远是 BDDL 里那一条 `task.language`，外面套上
`robot_video_dataset.py:23` 的模板：

```python
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
```

没有 seen/unseen 之分、没有随机性。所以验收 1 与官方数字对照时**不存在指令口径问题**。

每 episode 的步数上限（`eval_libero_single.py:432-442`）：
`libero_spatial / libero_object / libero_goal` = **400**，`libero_10 / libero_90` = 700；
循环上界是 `max_steps + num_steps_wait`（spatial ⇒ 430 步）。
成功判据就是 `env.step` 返回的 `done`（LIBERO 里 `done = self._check_success()`）。

初始状态是**确定性的**，不采样：`initial_states = task_suite.get_task_init_states(task_id)`，
第 `trial_idx` 条 episode 用 `initial_states[trial_idx]`；不够 `num_trials` 时循环复用。

## 5. 已知坑

### 5.1 mujoco 版本必须钉死 3.3.2
FastWAM README 明确写 `pip install mujoco==3.3.2`，且其 LIBERO 数据集目录名就叫
`data/libero_mujoco3.3.2`。mujoco 版本与 LIBERO 数据版本不一致会导致物理/渲染漂移。

### 5.2 headless 渲染：只有 EGL 可用
实测三种后端（本机无显示器）：

| `MUJOCO_GL` | 结果 |
|---|---|
| `egl` | ✅ 可用。RGB/depth/segmentation 全部正常 |
| `osmesa` | ❌ `AttributeError: 'NoneType' object has no attribute 'glGetError'`（无 OSMesa 库） |
| `glfw` | ❌ `FatalError: an OpenGL platform library has not been loaded`（无 X display） |

所以所有脚本必须 `MUJOCO_GL=egl`。

**EGL 的伪错误**：进程退出时 `mujoco/egl/__init__.py` 的 `GLContext.__del__`
会抛 `EGLError: <exception str() failed>`。这是**析构阶段**的噪音，渲染本身已成功、
数据已拿到。判断成功要看 stdout，不要看 stderr 尾部。

### 5.3 不要复用已有的 `fastwam` conda 环境
本机存在一个旧的 `fastwam` env，它**是坏的**，不要用：
- `import fastwam` 和 `import libero` 都 `ModuleNotFoundError`：两者是 editable 安装，
  分别指向 `/home/user1/workspace/arash/FastWAM/src` 和 `/home/user1/fastwam_libero`，
  **这两个目录在本机已被删除**。
- 环境里同时存在 `lib/python3.10/site-packages` 和一个来历不明的 `lib/python3.1/site-packages`，
  内容重复。`bin/huggingface-cli` 的 shebang 因此失效
  （`ModuleNotFoundError: No module named 'huggingface_hub'`），但
  `python -c "import huggingface_hub"` 却是好的 —— 症状极具误导性。
- `sys.path` 里混进了 `/home/user1/.local/lib/python3.10/site-packages`（user-site 泄漏），
  会让"装了什么"不可复现。新环境一律 `export PYTHONNOUSERSITE=1`。

### 5.4 `~/.libero/config.yaml` 是全局单例，且当前指向别的 LIBERO
LIBERO 在 import 时读 `~/.libero/config.yaml`。本机该文件当前内容指向
`/home/user1/workspace/arash/openpi/third_party/libero/libero/libero`
（**另一份** LIBERO），不是我们 clone 的这份。若不处理，`libero` 会去那棵树里
找 `assets` / `bddl_files` / `init_files`，行为不可控。

处理方式见 `probe/` 里的实现（不改全局文件，改为在进程内指向我们的 clone）。

### 5.5 element-level segmentation 的 id 就是 mujoco geom_id，但 id 0 有歧义
实测 robosuite 1.4.0 `camera_segmentations="element"`：
seg 值**直接等于 mujoco `geom_id`，无 ±1 偏移**。用 geom 名反查验证过：

```
segval  0 -> geom_id  0  floor
segval  8 -> geom_id  8  table_visual
segval 70 -> geom_id 70  gripper0_hand_visual
segval 84 -> geom_id 84  cube_g0_vis
```

**坑**：`geom_id == 0`（`floor`）与"背景"在数值上都可能是 0，无法区分。
我们注入的 patch geom 的 id 必然 > 0，所以对本项目无影响，但不要写
`seg == 0` 当背景判据。

### 5.6 depth obs 是原始缓冲，不是米
`obs["{cam}_depth"]` 是 OpenGL 归一化深度（实测范围 0.98–0.996），**不是米**。
转米必须过 `robosuite.utils.camera_utils.get_real_depth_map(sim, depth)`
（同一帧转换后为 0.55–2.71 m）。schema 里存哪一种要写清楚，否则后面几何全错。

### 5.7 gym 弃用警告
`import robosuite` 会打一堆 `Gym has been unmaintained since 2022 ...` 和
`[robosuite WARNING] No private macro file found!`。无害，可忽略。

### 5.8 显存：整模型放一张卡要 ~24 GB，本机 8 张卡都被别人占着
本机 8× L40S（每张 44.4 GiB 可用），但实测**每张卡都已被其他人的进程占用 ~28 GB**，
只剩 ~18 GB 空闲。整模型一次性加载会 OOM（实测我方进程涨到 16.85 GiB 时被打死）。

各部件的量级（bf16）：

| 部件 | 大小 | 推理必需？ |
|---|---|---|
| video DiT (Wan2.2 5B) | ~10 GB | 是（随机初始化后由 ckpt 覆盖） |
| **umT5-XXL text encoder** | **~11 GB** | 是，但**可以先算完就扔** |
| Wan2.2 VAE | ~1.4 GB | 是（编码输入图） |
| ActionDiT / action expert | ~1.5 GB | 是 |

⇒ **省显存的正确做法**：先只建 text encoder 把所有 instruction 的 embedding 算完，
`del` 掉、`torch.cuda.empty_cache()`，再用 `model.load_text_encoder=false` 建策略，
把缓存的 `context`/`context_mask` 直接传进 `infer_action`
（`infer_action` 原生支持 `prompt` 与 `context` 二选一）。
峰值降到 `max(11, 13) ≈ 13 GB`，在 18 GB 空闲下能跑。
`probe/run_instruction_sweep.py` 就是这么做的（`--text-device` 控制在哪算 embedding）。

这个拆分对后面的 FD 实验也是必要的：instruction 固定时根本不需要 T5 常驻，
省下的 11 GB 可以用来跑更大的 batch / 更多并行探针。

另外记得加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少碎片，
并用 `CUDA_VISIBLE_DEVICES` 手动挑一张空闲卡（先 `nvidia-smi` 看一眼）。

---

## 6. 已验证的渲染/相机事实（供 io_schema 使用）

在 robosuite 1.4.0 + mujoco 3.3.2 + EGL 下，`camera_heights=camera_widths=224`
实测（`Lift`/`Panda`/`agentview`）：

| obs key | shape | dtype |
|---|---|---|
| `{cam}_image` | (224, 224, 3) | uint8 |
| `{cam}_depth` | (224, 224, 1) | float32（归一化，见 §5.6） |
| `{cam}_segmentation_element` | (224, 224, 1) | int32（= geom_id，见 §5.5） |

相机内外参（`robosuite.utils.camera_utils`）：

```python
K = get_camera_intrinsic_matrix(sim, "agentview", 224, 224)
# [[270.392, 0, 112], [0, 270.392, 112], [0, 0, 1]]
E = get_camera_extrinsic_matrix(sim, "agentview")   # 4x4 camera->world
```

---

## 7. patch geom 的硬性约束（下一阶段，先记下来别丢）

### 7.1 必须无碰撞：`contype=0`、`conaffinity=0`

patch geom **只可见、不参与碰撞**。否则机械臂/夹爪可能撞到它，
动作变化里就混进了**物理碰撞**的贡献，而不是纯视觉扰动 ——
influence 会同时包含两种完全不同的机制，归因彻底糊掉，整张热图失效。

这同时也更物理真实：真实贴在桌面上的一张打印纸厚度可忽略，本来就不该产生碰撞。

### 7.2 用薄长方体，不用零厚度 plane

尺寸量级 10 cm × 10 cm × 1 mm，即 mujoco half-size `size="0.05 0.05 0.0005"`
（mujoco 的 box size 是半长）。理由：
- 有厚度就不会出现零厚度面在某些视角下渲染消失；
- 能正常接受光照、投射阴影；
- 照样分配独立 geom id ⇒ segmentation 计数不受影响。

另外 mujoco 的 `type="plane"` 是无限大/仅适合静态地面，纹理映射语义也不同，**不要用 plane**。

### 7.3 已验证的实现形式

两条路都实测可用（geom 出现在编译后的模型里、element-level seg 能数到像素、
被前方物体正确遮挡）：

**A. `set_xml_processor` 注入原始 XML**（最轻，`env.reset()` 后仍生效）：

```python
g.set("type", "box"); g.set("size", "0.05 0.05 0.0005")
g.set("group", "1")                       # 必须！否则 RGB 和 seg 里都看不见
g.set("contype", "0"); g.set("conaffinity", "0")   # 7.1 的要求
g.set("material", "wam_mat"); g.set("mass", "1e-8")
```
注意挂载点是 `env.env.set_xml_processor(...)` —— `ControlEnv` **不转发**这个方法。

**B. `BoxObject(obj_type="visual")`** —— robosuite 的 visual 模板本身就是
`{"conaffinity": "0", "contype": "0", "mass": "1e-8", "group": "1"}`
（`robosuite/models/objects/objects.py:504-512`），**天然满足 7.1**，
而且注册成 `MujocoObject` 后还能额外拿到唯一的 instance-level seg id。

### 7.4 其他已验证的坑（下一阶段会踩）

- **`group="1"` 是强制的**：LIBERO 默认 `render_visual_mesh=True` / `render_collision_mesh=False`，
  即只显示 group 1。不写 `group` 默认是 group 0 ⇒ RGB 和 segmentation 里**都不可见**。
- **`CustomMaterial(texture="<绝对路径>")` 会 assert 失败** —— `texture` 传字符串时必须是
  robosuite 内置 `ALL_TEXTURES` 的名字。绕过方式（实测可用）：
  ```python
  mat = CustomMaterial(texture=None, tex_name="wamtex", mat_name="wammat",
                       tex_attrib={"type": "2d"}, shared=True)
  mat.tex_attrib["file"] = "/abs/path/patch.png"     # 绝对路径完全支持
  ```
  走 `set_xml_processor` 注入时**必须自己给绝对路径**（已经过了 robosuite 把相对路径
  绝对化的时机），且 LIBERO 场景 XML 里没有 `<compiler texturedir=...>`。
- **编译后无法新增 geom**：`sim.model.ngeom` 只读，mujoco 没有 `mj_addGeom`。
  必须在 XML 编译前注入。
- **开 segmentation 会改变 RGB**：robosuite 为了不让 site 出现在 mask 里，
  会把 `sim.model.site_size[:, :] = 1.0e-8`（`robosuite/environments/robot_env.py:355-357`）。
  LIBERO 的目标区域标记就是 box site，于是 RGB 里它们**消失了**（实测 mean|Δ| ≈ 2.58/255）。
  ⇒ **不要在喂给策略的那个 env 上开 segmentation**。用两个 env（同 seed、同 init state），
  一个出 RGB 给模型、一个出 mask，或每次 reset 后把 `site_size` 改回去。
- **桌面高度**：`libero_spatial` / `libero_goal` 用 `Libero_Tabletop_Manipulation`，
  `table_full_size=(1.0, 1.2, 0.05)`、`table_offset=(0,0,0.90)`
  ⇒ **桌面世界坐标 z = 0.900**，x∈[−0.5, 0.5]、y∈[−0.6, 0.6]。
  贴合放置：`z = 0.900 + half_thickness + eps`。
  ⚠️ **`libero_object` 用的是 `Libero_Floor_Manipulation`（EmptyArena，没有桌子）**，
  patch 得贴地面。别假设所有 suite 都有桌面。
  运行时稳妥取法：`env.env.workspace_offset[2]`。
- **obs 图像是 bottom-up 的**（robosuite `IMAGE_CONVENTION="opengl"` ⇒ `[::convention]` 是 no-op）。
  与 `project_points_from_world_to_camera` 的 top-down 行号换算是 `obs_row = H - 1 - proj_row`。
- **`.pruned_init` 在 torch 2.7.1 上加载会炸**：LIBERO 内部 `torch.load(path)` 在
  torch ≥ 2.6 默认 `weights_only=True`，遇到 pickled numpy 数组会
  `UnpicklingError: Unsupported global: numpy.core.multiarray._reconstruct`。
  不改 LIBERO 的修法：
  ```python
  import numpy.core.multiarray, torch
  torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
  ```
- `benchmark.LIBERO_100` 是坏的（`KeyError`），用 `libero_90` + `libero_10`。
- `env.sim` 有，但 `env.model` / `env.set_xml_processor` **没有** —— 走 `env.env.*`。
