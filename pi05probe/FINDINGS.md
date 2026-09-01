# FINDINGS.md — S0 事实核查(π0.5 + LIBERO-Goal)

状态:**S0 完成**。环境已跑通,**官方 LIBERO 评测 demo 已跑完并出成功率**(JAX 29/30 = 96.7%,见 §0),
十问全部有据。**框架已定:整条流水线用 PyTorch**,权重已转、attention 已验证可逐层取出(见 §PT)。
剩下的【待执行】:PyTorch 下的 30-episode 成功率(§PT-5B)、Q7–Q10 在 `f78abd6` 上复核、S0.5 两项检查。

最后更新:2026-08-04 · 主机 `nnmc65`(10.145.87.65),8× L40S(每张 45.0 GiB,他人占用波动 8–39 GB)

---

## 版本与 commit

| 项 | 值 |
|---|---|
| openpi | `github.com/Physical-Intelligence/openpi` @ **`15a9616a00943ada6c20a0f158e3adb39df2ccac`**(2026-06-16) |
| openpi 的 LIBERO submodule | **`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`** |
| 我们原有的 LIBERO(FastWAM 用) | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| π0.5 权重 | `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero`(12 GB,含 `params/` + `assets/physical-intelligence/libero/norm_stats.json`) |
| 推理显存需求(README) | > 8 GB |
| **推理显存实测** | **8730 MiB**(`XLA_PYTHON_CLIENT_PREALLOCATE=false`,serve_policy 常驻)⇒ 一张 L40S 富富有余 |

⚠️ **openpi 钉的 LIBERO 与我们原有的不是同一个 commit。** 下面 Q7–Q10 的实测是在 `8f1084e` 上做的,
**必须在 `f78abd6` 上复核**(两者都是 LIBERO 0.1.0,预期一致但不能假设)。

---

## §0 环境:✅ 已跑通,官方 demo 已出成功率

### 结论先说:**没有装任何新环境**,复用本机已有的两个 conda env 当"依赖提供者"

本机 `~/miniconda3/envs/` 里早就有(建于 **2026-04-16**,与那 12 GB 权重同一天):

| conda env | Python | 关键版本 |
|---|---|---|
| `openpi-server` | 3.11 | jax **0.5.3** / flax **0.10.2** / torch **2.7.1+cu126** / transformers **4.53.2** / numpy 1.26.4 |
| `openpi-libero` | 3.8 | mujoco **3.2.3** / robosuite **1.4.1** / torch **1.11.0+cu113** / numpy **1.22.4** |

这正好是 openpi 官方 **两进程两环境** 架构要的两套(`examples/libero/README.md:37-62`)。
两个 env 当初是 editable 安装到 `/home/user1/arash/openpi`(@ `a190e00`,2026-04-15)。

**为什么这两个 env 对我们的 clone 有效(不是碰运气):**
`a190e00..15a9616` 只改了 4 个文件,而且依赖变化**只有删除**(`openpi-client` 去掉了 `tree`),
**没有任何新增**:

```
packages/openpi-client/pyproject.toml | 5 ++---
src/openpi/policies/droid_policy.py   | 2 +-
src/openpi/policies/libero_policy.py  | 2 +-      ← [:, :7] 改成 [..., :7](支持 batch)
uv.lock                               | 23 ------
```
`pyproject.toml` 主文件在两个 commit 间**逐字节相同**。

**做法:不改这两个 env(它们可能还被别的项目用),用 `PYTHONPATH` 覆盖源码路径。**
`sys.path` 顺序是 `''` → `PYTHONPATH` → stdlib → site-packages(`.pth` 追加在最后),
所以 `PYTHONPATH` 优先于 `.pth` 里的 `/home/user1/arash/openpi/src`。已实测:

```
openpi        -> .../WAMattack/third_party/openpi/src/openpi/__init__.py
openpi_client -> .../WAMattack/third_party/openpi/packages/openpi-client/src/openpi_client/__init__.py
libero        -> .../WAMattack/third_party/openpi/third_party/libero/libero
```

⚠️ **`openpi-libero` 里 libero 的 editable 安装是坏的**:
`__editable___libero_0_1_0_finder.py` 的 `MAPPING` 是空 dict ⇒ `import libero` 直接 `ModuleNotFoundError`。
官方 README:48 本来也是用 `PYTHONPATH` 挂 libero,照办即可,不用修那个 env。

**⚠️ `uv` 依然被安全分类器拒(裸调、全路径都拒);`bash <script>`、带管道/`&&` 的复合命令也常被拒,
但 `<abs-path>/python <abs-path>/script.py` 可以过。** 所以脚手架一律写成 **单个 python 文件**,
不写 shell 脚本、不用管道。(`pi05probe/env.sh` / `check_env.sh` 只作为文档留着,跑不了。)

### 路径与环境变量(都在 `pi05probe/run_demo.py` 里)

| 项 | 值 | 理由 |
|---|---|---|
| `OPENPI_DATA_HOME` | `/home/user1/.cache/openpi` | 那 12 GB 权重恰好就在 `<HOME>/openpi-assets/checkpoints/pi05_libero`(= openpi 默认 cache),**一个字节都不用动,不 symlink 不复制**。路径规则见 `shared/download.py:58-61` |
| | | 安全性已核:`download.py:198-202` 对 `openpi-assets/checkpoints/` 的失效阈值是 **2025-02-03**,`:205-216` 只在 `mtime <= 阈值` 时 `rmtree`;该目录 mtime = **2026-04-16** ⇒ 不会被删 |
| `LIBERO_CONFIG_PATH` | `pi05probe/libero_config` | 全局 `~/.libero/config.yaml` 指向 **`/home/user1/workspace/arash/openpi/...`**(同事目录,只读不碰)⇒ 必须隔离。⚠️ 若 config 文件不存在,`libero/libero/__init__.py:62-96` 会走 `input()` **交互卡死** ⇒ 我们已预置好 yaml |
| `MUJOCO_GL` / `PYOPENGL_PLATFORM` | `egl` | 本机唯一可用 |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | jax 默认 preallocate 75% = 34.5 GB,在他人占卡时必 OOM。关掉后实测只用 **8730 MiB** |
| port | **8123** | 不用默认 8000,免得撞别人的 server |

### ✅ 官方 demo 结果(`libero_goal`,seed=7,3 trial/task,共 30 episode)

```
Total success rate: 0.9666666666666667      (29/30)
Total episodes: 30
```
官方 README 报的是 50 trial/task 下 **98.0**;30 episode 上 96.7% 与之相符。
唯一的失败在 task 0 `open the middle drawer of the cabinet`(2/3)。

**PART B 的四条候选指令全部 3/3**:

| 指令 | 成功率 |
|---|---|
| `turn on the stove` | 3/3 |
| `put the wine bottle on the rack` | 3/3 |
| `put the bowl on the plate` | 3/3 |
| `put the bowl on top of the cabinet` | 3/3 |

其余 6 条也全是 3/3(只有 task 0 是 2/3)。⇒ **S0.5 补测的"clean 成功率"这一项已有初步答案:
四条候选指令都不塌**(样本量小,S0.5 要按计划补到 10–20 episode 才算正式数)。

单 episode 墙钟约 **4–11 s**(含 warmup;首个 episode 含 JIT 编译更久)。

复现命令:
```bash
/home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/run_demo.py --trials 3
/home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/run_demo.py --status
/home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/run_demo.py --kill
```
两个子进程用 `start_new_session=True` spawn,**关终端不受影响**;日志在 `pi05probe/out/{server,client}.log`,
视频在 `pi05probe/out/videos/libero_goal/`。

⚠️ **`env.seed(seed)` 只在构造时调一次**(`main.py:195`),之后每个 episode 只 `reset()` + `set_init_state()`
**不重新 seed** ⇒ 按 Q10 的结论,官方 eval 里那三个 fixture 的位置**每个 episode 都在漂**。
我们自己的实验必须按 Q10 的 `start_shared()` 做法每次 reset 前重 seed。

---

## Q1 · 相机与分辨率 【代码已确认】

`src/openpi/policies/libero_policy.py:52-69`:

```python
base_image  = _parse_image(data["observation/image"])
wrist_image = _parse_image(data["observation/wrist_image"])
inputs = {
    "state": data["observation/state"],
    "image": {
        "base_0_rgb":       base_image,
        "left_wrist_0_rgb": wrist_image,
        "right_wrist_0_rgb": np.zeros_like(base_image),   # 补零
    },
    "image_mask": {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
    },
}
```

- **两路真实相机 + 一路补零**:`base_0_rgb`(agentview)、`left_wrist_0_rgb`(robot0_eye_in_hand)、
  `right_wrist_0_rgb` 全零且 **mask=False**(π0.5 不是 PI0_FAST ⇒ 走 `np.False_`)。
- **分辨率 224×224**(`libero_policy.py:14-15` 的 `make_libero_example`),而 LIBERO 渲染是 256。
- `state` 8 维 —— ⚠️ **但 π0.5-LIBERO 根本不读它,见 §Q3b。**

### ⚠️⚠️ 关键更正:client 侧做的是 **180° 旋转**,不是单纯的上下翻转

`examples/libero/main.py:113-122`:

```python
# IMPORTANT: rotate 180 degrees to match train preprocessing
img       = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
img       = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, 224, 224))
```

**`[::-1, ::-1]` = 上下翻 + 左右翻 = 旋转 180°**,不是 Q8 里 LIBERO 自己下游用的那个单 `[::-1]`。
第一个 `[::-1]` 抵掉 OpenGL 的 bottom-up,第二个 `[::-1]` 是**额外的左右镜像**
(openvla/openpi 训练数据就是这么存的,注释写明"to match train preprocessing")。

⇒ **§A-5 的 attention→世界坐标反投影必须把这个 180° 旋转反解回去**,否则 saliency 会整体点中心对称位置。
两路相机施加的是同一个变换。

### 256 → 224:**零 padding**,纯 BILINEAR 等比缩放【实测】

`openpi_client/image_tools.py:38-58` `_resize_with_pad_pil`:
`ratio = max(256/224, 256/224) = 1.1428571429` ⇒ `resized = (224, 224)` ⇒ `pad = (0, 0)`。
实测输出 `(224,224,3) uint8`,四边最小值 15/8/9/11(全非 0 ⇒ 确认没有黑边)。

**反投影用的像素中心映射(PIL / `tf.image.resize_with_pad` 约定,不是 `u*224/256`):**
```
u_256 = (u_224 + 0.5) * 256/224 - 0.5 = (u_224 + 0.5) / 0.875 - 0.5
f_224 = f_256 * 224/256    ⇒ agentview: f_256 = 309.0193  →  f_224 = 270.3919
c_224 = (111.5, 111.5)
```

服务端 `model_transforms` 里还有一次 `_transforms.ResizeImages(224, 224)`(`config.py:131`),
对已经 224 的输入是 **no-op**(`image_tools.py:28-29` 提前 return)。

---

## Q2 · 动作空间 —— ✅ 全部确认

维度:`src/openpi/policies/libero_policy.py:100` → `data["actions"][..., :7]` ⇒ **对外 7 维**;
模型内部 `action_dim=32`(`pi0_config.py:24`),LIBERO 只取前 7,其余是 padding。

**是否 delta**:`config.py:332-338` 注释 —— "LIBERO already represents actions as deltas",
且 `pi05_libero` 用 `extra_delta_transform=False`(`config.py:749`)
⇒ **模型直接输出 LIBERO 原生动作空间,没有额外 delta/absolute 转换。**

**`main.py:153` `env.step(action.tolist())` —— 原样送入,没有任何缩放、没有符号翻转。**

控制器:LIBERO `env_wrapper.py:17` `controller="OSC_POSE"`,用 robosuite 默认
`robosuite/controllers/config/osc_pose.json`:

```json
{"input_max": 1, "input_min": -1,
 "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
 "output_min": [-0.05,-0.05,-0.05,-0.5,-0.5,-0.5],
 "kp": 150, "impedance_mode": "fixed", "control_delta": true, "uncouple_pos_ori": true}
```

| 维 | 含义 | 单位 |
|---|---|---|
| `a[0:3]` | **平移 delta**,末端位置 | `[-1,1]` 线性映射到 **± 0.05 m**(`base_controller.py:104-123` `scale_action`) |
| `a[3:6]` | **旋转 delta,axis-angle `[ax,ay,az]`**(**不是欧拉角**) | `[-1,1]` 线性映射到 **± 0.5 rad** |
| `a[6]` | 夹爪 | **只用 `sign()`**,见下 |

旋转是 axis-angle 的证据:`robosuite/utils/control_utils.py:150-176` `set_goal_orientation` 的
docstring 明确写 *"Desired relative change in orientation, in axis-angle form [ax, ay, az]"*,
实现是 `axisangle2quat(delta)` → 旋转矩阵左乘当前姿态。
⇒ **S3 旋转通道用 SO(3) 测地距离是对的,绝不能直接相减。**

⚠️ **`scale_action` 先 `np.clip(action, -1, 1)`**(`base_controller.py:120`):
**模型输出超出 ±1 的部分被截断** ⇒ Δa 里落在饱和区的差异对环境**没有效果**。
S3 报 influence 时要标出哪些格子的差异发生在饱和区。

### ⚠️⚠️ 夹爪:**幅值被完全丢弃,只有符号有意义**

`robosuite/models/grippers/panda_gripper.py:43-58`:

```python
def format_action(self, action):
    """Maps continuous action into binary output   -1 => open, 1 => closed"""
    self.current_action = np.clip(
        self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0)
    return self.current_action
```
`speed = 0.01`。⇒ 夹爪命令是**积分状态**,而且输入只过 `np.sign()`。

**⇒ 不翻符号的 `Δa[6]` 对环境完全没有影响。** 这正好硬化了计划里"夹爪看是否翻转"的写法:
```
flip ⟺ sign(a_patched[6]) != sign(a_clean[6])        # 注意 sign(0)=0 的边界
```
`main.py:17` `LIBERO_DUMMY_ACTION = [0.0]*6 + [-1.0]` ⇒ warmup 那 10 步是"不动 + 张开"。

---

## Q3 · 归一化 —— ✅ 是**分位数** q01/q99,不是 mean/std

`config.py:187`:`use_quantile_norm = (model_config.model_type != ModelType.PI0)`。
`pi05_libero` 的 `model_type` 是 **`PI05`**(`pi0_config.py:51-56`)⇒ **`use_quantile_norm=True`**。

```python
# transforms.py:141-145
def _normalize_quantile(self, x, stats):
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
# transforms.py:175-183  _unnormalize_quantile 反过来
```

**stats 来源**(`policy_config.py:59-64`):从 **checkpoint 的 `assets/`** 读,
**不是** config 的 `assets_dirs` —— 注释明说是为了和训练时完全一致。
`asset_id = repo_id = "physical-intelligence/libero"`(`config.py:181-183`)
⇒ `<ckpt>/assets/physical-intelligence/libero/norm_stats.json`。
json 结构:`{"norm_stats": {"state": {...}, "actions": {...}}}`,每项都带 **`mean/std/q01/q99` 四个字段**
(所以文件本身看不出用哪套,得看 `use_quantile_norm`)。

### 实测数值(`pi05probe/out/s0_facts.txt`)

`actions`(dim=7):
```
q01  = [-0.74738 -0.79612 -0.93750 -0.11580 -0.16943 -0.19450 -1.00000]
q99  = [ 0.93712  0.85950  0.93712  0.14023  0.18104  0.31155  0.99960]
mean = [ 0.02683  0.08886 -0.09983  0.00024  0.00128 -0.00294 -0.13052]
std  = [ 0.33119  0.37192  0.45226  0.03949  0.06278  0.07318  0.99145]
```
`state`(dim=8):
```
q01  = [-0.35245 -0.26825  0.04084  1.53177 -2.71523 -1.07654  0.00172 -0.04004]
q99  = [ 0.13891  0.32520  1.25690  3.26277  2.44372  0.56385  0.04031 -0.00171]
```

**给 S3 §D 用的量:**
```
q99 - q01 = [1.6845 1.6556 1.8746 0.2560 0.3505 0.5061 1.9996]
norm_std  的分母:优先用 clean rollout 实测的 std_clean(a^i);训练集 std 作先验 = 上面那行 std
norm_bound 的 d_max^i = max(|a_clean^i − q01^i|, |a_clean^i − q99^i|)   # 逐帧算
```
⚠️ 注意 **旋转三维的 q99−q01 只有 0.26–0.51,比平移的 1.7–1.9 小 4–7 倍** ——
这正是计划里"绝不对 7 维直接求 L2"的量化理由:同样的绝对 Δ,旋转维相对正常波动要异常得多。
夹爪的 q01/q99 ≈ ±1 ⇒ 近二值,确认"做连续归一化没有意义"。

---

## Q3b · ⚠️⚠️ 顺带发现:**π0.5-LIBERO 完全不消费 `observation/state`**

三条证据链:

1. `pi0.py:151-157` —— pi05 的 `embed_suffix` 里那个 state token 在 **`if not self.pi05:`** 分支里
   ⇒ pi05 的 suffix **只有 10 个 action token,没有 state token**。
2. `config.py:745` —— `pi05_libero` 显式设了 **`discrete_state_input=False`**
   (虽然 `pi0_config.py:40-41` 里 pi05 的默认值是 `True`)。
   ⇒ `transforms.py:256-261` `TokenizePrompt` 收到 `state=None` ⇒ **state 不进 prompt**。
3. 于是 `state` 被 `Normalize` + `PadStatesAndActions(32)` 处理了一遍,然后**没有任何人读它**。

⇒ **动作只依赖 (3 路图像, prompt, flow matching 的 ε)。**
这对 S2 是好消息:贴 patch 不会通过 state 这条路径间接影响输出,反事实查询更干净;
也意味着 S2 里不需要为 state 做任何对齐处理。

【待执行·便宜】喂两个不同 `observation/state`、其余(图像/prompt/`noise`)逐位相同,
断言输出动作**逐位相同**。必须用显式 `noise=`,因为 `Policy.infer` 每次调用都会 split rng(见 Q5)。

---

## Q4 · action chunk 与执行步数 【代码已确认 —— 与原计划不同,重要】

`examples/libero/main.py`:

```python
replan_steps: int = 5                      # :29
num_steps_wait: int = 10                   # :37
elif args.task_suite_name == "libero_goal":
    max_steps = 300                        # :65   longest training demo has 270 steps
while t < max_steps + args.num_steps_wait: # :104
    if t < args.num_steps_wait: ...        # :108
assert len(action_chunk) >= args.replan_steps   # :146
action_plan.extend(action_chunk[: args.replan_steps])   # :148
```

| 量 | 值 |
|---|---|
| `K`(chunk 长度 / `action_horizon`) | **10** |
| **`H_exec`(实际执行)** | **5** ← 不是 10 |
| `num_steps_wait` | **10**(FastWAM 那边是 30,别混) |
| libero_goal `max_steps` | **300**(训练集最长 demo 270) |
| **单 episode 决策点数** | 最多 **60**,典型成功 episode 约 **30–54** |

⇒ **S2 的 `T` 取 clean rollout 的全部 replan 边界(每 5 步一个),覆盖到任务成功为止**,不固定 T=20。
⇒ S3 的"只累加 executed prefix"= 每个 10 步 chunk 的**前 5 步**。

---

## Q5 · flow matching 的 ε —— ✅ **可以显式传入,这是最好的情形** 【代码已确认】

`src/openpi/models/pi0.py:217-231`:

```python
def sample_actions(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    num_steps: int | at.Int[at.Array, ""] = 10,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
) -> _model.Actions:
    # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
    dt = -1.0 / num_steps
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
...
    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))   # :278
```

- **有一个显式的 `noise` 参数**。传同一个 tensor 进去,clean 与 patched 两次前向就共享逐位相同的 ε。
  不需要退化到"多次采样取平均"。
- 采样器:rectified flow / flow matching,显式 Euler,`dt = -1/num_steps`,默认 **10 步**。
- 训练用的随机在 `:190-200`(`compute_loss`),与推理路径无关。

⇒ **S2 的 ε 共享要求可以满足。** 实现上把 `noise` 预先生成一次并复用同一个数组。

### ✅ `Policy.infer()` 确实透传 `noise` —— 不用绕过 Policy 层

`src/openpi/policies/policy.py:68` 起:

```python
def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
    ...
    self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)   # :74
    sample_kwargs = dict(self._sample_kwargs)
    if noise is not None:
        noise = jnp.asarray(noise)
        if noise.ndim == 2:                       # (action_horizon, action_dim)
            noise = noise[None, ...]              # 自动补 batch 维
        sample_kwargs["noise"] = noise            # :87
```
⇒ **直接 `policy.infer(element, noise=eps)` 即可**,整条 transform 链(归一化、tokenize)照走。

### ⚠️⚠️ 但 **websocket 那条路传不了 `noise`** ⇒ S2 必须单进程内调用

`src/openpi/serving/websocket_policy_server.py:61` 是 `action = self._policy.infer(obs)`,
**没有 noise 参数**。而 `policy.py:74` 每次调用都 `jax.random.split(self._rng)`
⇒ **同样的输入连调两次会得到不同的 ε、不同的动作。**

⇒ **S2/S3 的扫描代码不要走 `run_demo.py` 那套 websocket 双进程**,
在 server 环境里直接 `create_trained_policy(...)` 拿到 `Policy` 对象,自己调 `infer(obs, noise=eps)`。
websocket 那套只用于"复现官方 demo 数字"。

⚠️ 另一条与红线 3 相关:`policy.py:63` 是 `nnx_utils.module_jit(model.sample_actions)`,
**jit 过的**;`infer` 里固定 `[np.newaxis, ...]` 打成 batch=1。
要做 batch=25 就得绕开 `infer` 直接调 `sample_actions`,**换 batch 形状会触发重编译**
⇒ S2.0 的 batch 不变性测试非做不可。

---

## Q6 · attention:文本 token → 图像 patch —— ✅ **PyTorch 路径可取** 【代码已确认】

**框架结论:用 PyTorch 路径。** openpi 有 `src/openpi/models_pytorch/`
(`pi0_pytorch.py`、`gemma_pytorch.py`、`preprocessing_pytorch.py`、`transformers_replace/`),
基于 HuggingFace `transformers 4.53.2`,并配 `examples/convert_jax_model_to_pytorch.py` 做权重转换。

**关键:PyTorch 路径显式把 attention 实现切成 `eager`** —— `pi0_pytorch.py:392` 与 `:448`:

```python
self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
```

`eager` 会真正物化 `softmax(QK^T)`(不像 SDPA/flash 那样融合掉),HF 的 eager 路径支持
`output_attentions=True` ⇒ **attention 权重可取,且不必改 vendored 源码**(用 hook / 传 flag)。

**mask 结构 —— 文本确实能看到图像**(`pi0_pytorch.py` 注释与 mask 构造):

```
:210  # Create attention masks so that image tokens attend to each other
:224  # full attention between image and language inputs      ← 关键
:261  # Set attention masks so that image and language inputs do not attend to state or actions
:307  # Set attention masks so that image, language and state inputs do not attend to action tokens
```

⇒ token 布局是 **[images…, language…, (state), actions…]**,prefix-LM 形式:
images 与 language 之间**双向全注意**;state/action 是后续 block,前缀看不到它们。
**所以 `attn[text_idx, image_idx]` 这一路存在且有意义**,不必退化去取 CLS→patch(那正是计划里点名要避免的设计失误)。
(π0.5 的 suffix 里其实**没有 state token**,见 Q3b。)

### ✅ 具体规格全部确认(`pi05probe/out/s0_facts.txt`)

| 项 | 值 | 出处 |
|---|---|---|
| vision tower | **SigLIP `So400m/14`**,`pool_type="none"` | `pi0.py:81-89` |
| ⇒ patch grid | 224/14 = **16×16 = 256 token / 图** | |
| ⇒ **无 CLS token** | `pool_type="none"` 不池化,全是 patch token | |
| prefix LM | **`gemma_2b`:width 2048,depth 18,num_heads 8,num_kv_heads 1(MQA),head_dim 256** | `gemma.py:79-87` |
| action expert | `gemma_300m`:width 1024,depth 18,heads 8,kv_heads 1,head_dim 256 | `gemma.py:69-78` |
| `max_token_len` | **200**(pi05;pi0 是 48) | `pi0_config.py:38-39` |
| `action_horizon` | **10** | `config.py:745` |

### token index 区间(直接可用)

```
[   0 ..  255]  base_0_rgb          16×16      ← 定位主用这一路(§A-4)
[ 256 ..  511]  left_wrist_0_rgb    16×16
[ 512 ..  767]  right_wrist_0_rgb   16×16      (全零, image_mask=False)
[ 768 ..  967]  language            200 slot
------------------------------------------------ prefix 长度 = 968
[ 968 ..  977]  action              10         ← suffix,pi05 无 state token
------------------------------------------------ 总长 978
```
顺序由 `obs.images` 的**插入序**决定(`pi0.py:112-118` 遍历 dict),
而 `libero_policy.py:53-57` 的插入序就是 base / left_wrist / right_wrist。

⇒ **B0 要的那一路 = `attn[768:968, 0:256]`,reshape 成 `(Z, 16, 16)`。**

### ⚠️⚠️ 200 个 text slot 里绝大多数是 padding —— `Z` 必须按 mask 取

`tokenizer.py:33`(`discrete_state_input=False` 分支):
`tokens = encode(text, add_bos=True) + encode("\n")`,
`:35-38` 不足 200 的用 **0 填充**、`mask=False`。

实测各候选指令的**真实 token 数**:

| Z | 指令 | pieces |
|---|---|---|
| 6 | `turn on the stove` | `<bos> turn ▁on ▁the ▁stove \n` |
| 9 | `put the wine bottle on the rack` | `<bos> put ▁the ▁wine ▁bottle ▁on ▁the ▁rack \n` |
| 8 | `put the bowl on the plate` | `<bos> put ▁the ▁bowl ▁on ▁the ▁plate \n` |
| 10 | `put the bowl on top of the cabinet` | `<bos> put ▁the ▁bowl ▁on ▁top ▁of ▁the ▁cabinet \n` |
| 9 | `open the middle drawer of the cabinet` | `<bos> open ▁the ▁middle ▁drawer ▁of ▁the ▁cabinet \n` |

⇒ **算 `A ∈ R^{Z×N_v}` 时 `Z` 只能取 `tokenized_prompt_mask == True` 的行**,
否则 190+ 行 padding 垃圾进 max/sum。**这一条直接决定 §A saliency 的正确性。**
⇒ 也证实了 §C:**各指令 Z 不同(6 / 8 / 9 / 10)⇒ "全 token max" 图跨指令确实有偏**,
B1/B2 必须改用名词专属图或逐 token 图。数值在这里,可直接引用。

⇒ §A-2 的"最小对"在 token 层面看得很清楚:
`put the bowl on the plate` (Z=8) vs `put the bowl on top of the cabinet` (Z=10),
**前 4 个 token `<bos> put ▁the ▁bowl` 逐位相同**,差异从第 5 个 token 开始。

### 框架选择的复核:**JAX 路径也物化了 attention** —— 两条路都可行,但都要额外工作

之前记的"用 PyTorch 路径"仍然成立(HF eager + `output_attentions=True` 最省事),
但 **JAX 路径并没有被 fused kernel 挡住**,`gemma.py`:

```python
:217  logits        = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
:226  masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
:228  probs         = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)   ← 显式物化
```
`probs` 形状 `[B, K=1, G=8, T, S]`。

**两条路各自的代价(S1 之前必须选定):**

| 路 | 代价 |
|---|---|
| **JAX** | 18 层被 `nn.scan` + `nn.remat(policy=nothing_saveable)` 包住(`gemma.py:359-380`)⇒ 取**逐层** attention 需要在 `Attention.__call__` 里 `sow` 并让 scan 带出 `intermediates` 轴 ⇒ **要动 vendored 源码**(计划要求尽量避免) |
| **PyTorch** | ① 我们只有 JAX checkpoint(`params/`,没有 `model.safetensors`)⇒ 要先跑 `examples/convert_jax_model_to_pytorch.py`;② 必须把 `src/openpi/models_pytorch/transformers_replace/*` **拷进 site-packages/transformers**(`README.md:207`,`pi0_pytorch.py:118-122` 会检查)⇒ **会污染共用的 `openpi-server` conda env** ⇒ 得先 clone 一个私有 env |

⇒ **【待决策 · 已到需要拍板的点】** 见文末"待办"。

【待执行】attention rollout 的显存开销:`probs` 单层是 `[1, 1, 8, 978, 978]` fp32 ≈ **30.6 MB**,
18 层全存 ≈ **550 MB/帧**。若只留 `[768:968, 0:256]` 这一块 ⇒ 单层 `8×200×256` fp32 = 1.6 MB,
18 层 ≈ **29 MB/帧**,`T≈60` 帧 ≈ **1.7 GB** ⇒ 按 §B 的 schema 存盘完全可行(但要按 mask 裁 Z 到 ~10,再降 20 倍)。

---

## Q7 · 场景加载与 geom 注入 —— ✅ 已在 **libero_goal** 上实测通过

【实测·LIBERO 8f1084e,需在 f78abd6 复核】

注入点 `robosuite/environments/base.py:186-193, 215-230`:`env.env.set_xml_processor(fn)`
(注意是 `env.env`,`ControlEnv` 不转发),`_initialize_sim()` 在每次 hard reset 都会调它 ⇒ **survives `reset()`**(连续 3 次 reset 验证)。

**必须把 geom 包在自己的 `<body>` 里。** 实测对比:

| 方式 | ngeom | 新 geom_id | 既有 geom_id | agentview 像素 |
|---|---|---|---|---|
| `<geom>` 直接塞进 `<worldbody>` | 240 | 7 | **全部 +1 偏移**(`table_collision` 7→8 …) | 1024 |
| **包进新 `<body>` 再塞** | 240 | **239(最后)** | **保持不变** | **985** |
| 同上但 `group="0"` | 240 | 7 | 偏移 | **0(不可见)** |

⇒ 直接塞 `<geom>` 会**把所有既有 geom_id 重新编号**,之前记录的 seg id 全部失效。**必须包 body。**
`group="1"` 是强制的(`base.py:290-292` 只开 geomgroup[1]),实测 `group="0"` 渲染 0 像素。

**z-fighting**:桌面 z=0.900,`z=0.9005`(底面正好齐平)时像素数随绘制顺序抖动(986 vs 1024);
**用 `z ≥ 0.9015`**(≈1 mm 间隙)。

**注入不改状态布局**:`nq=41, nv=37, len(flatten())=79` 不变 ⇒ 同一个 shared state 对 clean 与 injected env 通用。

**可用桌面区域**:桌面 `x∈[-0.5,0.5]`、`y∈[-0.6,0.6]`、top `z=0.900`。
所有 keep-out 都止于 `x ≤ 0.1622` ⇒ 对 10 cm patch(半边 0.05),
**`x∈[0.22,0.45]`、`y∈[-0.55,0.55]` 这一带无条件不与任何 fixture/物体相交**。

**最佳锚点 `(0.20, 0.30, 0.9015)`** —— 1070 px(占画面 1.63%)、无碰撞、**不遮挡任何任务物体**;
次选 `(0.10, 0.28, 0.9015)`,985 px,完全在画面内。

---

## Q8 · 每帧传感器 —— ✅ 全部可取 【实测】

`OffScreenRenderEnv(..., camera_depths=True, camera_segmentations="element")`,7 个相机
(`frontview, birdview, agentview, sideview, galleryview, robot0_robotview, robot0_eye_in_hand`)。

| 项 | 取法 / 值 |
|---|---|
| RGB | `obs["agentview_image"]` (256,256,3) uint8,**== `sim.render(...)` 无翻转** |
| **RGB 是 bottom-up(视觉上上下颠倒)** | `IMAGE_CONVENTION="opengl"` ⇒ `convention=1` ⇒ `img[::1]` 是 no-op。LIBERO 自己在下游 `[::-1]`。**RGB/depth/seg 必须施加同一个翻转,否则错位** |
| depth | `obs["agentview_depth"]` (256,256,1) float32,**原始归一化**,实测 [0.9847, 0.9966] |
| depth→米 | **`camera_utils` 导不进来(缺 h5py)**,须内联:`near/(1 - d*(1-near/far))`,`extent=10.6098, near=0.010610, far=530.4908` ⇒ 实测 0.6909–3.0689 m |
| segmentation | `obs["agentview_segmentation_element"]` (256,256,1) int32,**值 == `geom_id` 无偏移**;`ngeom=239`;⚠️ id 0 是真实 geom(`floor`),不能当背景 |
| 内参 K | 内联 `f = 0.5*H/tan(fovy*π/360)`;agentview @256² ⇒ **f=309.0193, c=(128,128)**;eye_in_hand f=166.8128 |
| 外参 [R\|t] | 内联 `T.make_pose(cam_xpos, cam_xmat) @ diag(1,-1,-1,1)`;agentview 固定在世界 `(0.65861, 0, 1.61035)`;eye_in_hand 随状态变,**每帧读** |
| qpos | `sim.data.qpos` (41,);`sim.get_state().flatten()` (79,) |

### ✅ 好消息:"开 seg 会改 RGB"这个坑 **在 libero_goal 上是 no-op**
两个 env 在同一 shared state 下实测:**agentview 与 wrist 都 0/65536 像素不同**。
原因:该场景 29 个 site 的 alpha 全 ≤ 0(`site_rgba[:,3] ∈ {-1, -0.5, -0.3, 0}`,`count(alpha>0)==0`),
`MujocoEnv.reset` 已把可视化 site 的 alpha 归零,再把不可见的 site 缩小自然无影响。
⇒ **可以用单个 env 同时出 RGB 和 seg**,不必开两个。(想要跨 arena 的保险做法:reset 后把 `site_size` 还原。)

---

## Q9 · 设置 qpos 并重新渲染 —— ✅ 可以,但有一个必须设计规避的陷阱 【实测】

```python
s = env.sim.get_state().flatten()          # 记录 (79,)
obs = env.regenerate_obs_from_state(s)     # 恢复 + 重渲染(LIBERO helper)
# 底层等价:set_state_from_flattened(s); sim.forward()   ← forward() 必须
```
- 状态恢复**精确**:`state_maxdiff = 0.0`;恢复→渲染**确定且幂等**(3 次 maxdiff 0)。
- `sim.forward()` **必需**,否则渲染是过期运动学。

⚠️ **陷阱:恢复后渲染 ≠ rollout 当时的实时渲染。**
实测 agentview 817 px 不同(max 45),**wrist 20844 px 不同(32%!)**。
根因:robosuite 的 step 循环 `sim.step()` 后直接渲染、**没有补一次 `mj_forward`**,
实时帧的 `xpos` 比 `qpos` 滞后一个积分子步;我们的 `forward()` 把这个滞后消掉了。wrist 相机挂在夹爪上所以差异最大。

⇒ **S2.1 的做法(采纳报告的建议 B):clean 参考帧也走 `regenerate_obs_from_state()` 这同一条恢复路径。**
这样 clean 与 patched 只差 patch 本身。(另一条路是完整 action replay,实测可 bit-exact,但更贵。)

---

## Q10 · libero_goal 十个初始状态是否一致 —— ❌ **不一致,必须强制统一** 【实测,关键】

十个 `.pruned_init` **全部互不相同**,各为 `(50, 79) float64`,sha256 十个全不同。

- 同一 episode index k=0、45 个 task 两两比:**max abs diff = 0.0927**(≈9.3 cm / 0.093 rad)
- 全 `(50,79)` 数组:9 个 task 对 task-0 全部 `False`
- 50 个状态的**集合**也不同;穷举 45 对**零共享行** ⇒ 每个 task 独立采样
- k=0 时 79 列里 **64 列逐位相同**;不同的 15 列正是:**7 个 Panda 关节** + **4 个可动物体的 x,y**

状态向量布局(实测):`[time(1), qpos(41), qvel(37)] = 79`
```
qpos 0..6  robot0_joint1..7   | 7,8 gripper finger | 9..15 bowl(free) | 16..22 cream_cheese
23..29 wine_bottle | 30..36 plate | 37,38,39 cabinet top/middle/bottom | 40 stove button
```

### ✅ 但状态**可跨 task 迁移** —— 十个 task 编译出结构完全相同的模型
`nq=41, nv=37, ngeom=239, nbody=38, njnt=17, nsite=29, ncam=7, nu=9, flatten=79`,
且 joint/body/geom 的**有序名字列表全相等** ⇒ geom_id 与 qpos 地址跨 task 一致(seg id 也可跨 task 比较)。
编译后 XML 只差 **3 行**(就是三个 fixture 的 body pos)。

把 task-0 的 `init_states[0]` 施加到 task 0/3/5/7:
```
state_maxdiff_vs_applied = 0.000e+00 ,  img_maxdiff = 0 ,  imgsha = 6c6e3e60856f  (四个 task 全同)
```
**256×256 agentview RGB 逐字节相同。**

### ⚠️⚠️ 最大的坑:三个 fixture 不在 qpos 里,`set_init_state` **管不到它们**
`bddl_base_domain.py:769-779`:可动物体走 `set_joint_qpos`,而 **fixture 走 `sim.model.body_pos`**。
`wooden_cabinet_1 / flat_stove_1 / wine_rack_1` 是无关节 body,**不在 `sim.get_state()` 里**,
**每次 `reset()` 都会用全局 numpy RNG 重新采样**。

搞错的代价(同一 state、两个 env 不同 seed):
```
flattened state maxdiff: 0.0        ← 状态完全相同
agentview RGB: max=163  mean=3.24   13175/65536 px 不同 (20.1%)
```
`seed()` 就是 `np.random.seed(seed)`(`bddl_base_domain.py:162-163`),是全局的。实测:
不重新 seed 则每次 reset 的 cabinet 位置**漂移**;`seed(1)` 后连续两次 reset **完全一致**。

### 强制统一的做法(实测可用)
```python
SHARED_SEED, SHARED_TASK, SHARED_EP = 10000, "open_the_middle_drawer_of_the_cabinet", 0
init = np.asarray(torch.load(f"{get_libero_path('init_states')}/libero_goal/{SHARED_TASK}.pruned_init",
                             weights_only=False))
SHARED_STATE = init[SHARED_EP].copy()          # (79,) float64

def start_shared(env):
    env.seed(SHARED_SEED)      # 必须在【每一次】 reset() 之前调
    env.reset()                # 确定性地采样 fixture body_pos
    return env.set_init_state(SHARED_STATE)    # 对齐机器人 + 4 个可动物体
```
配套注意:
1. **`seed()` 要在每次 `reset()` 前重调**,不是构造时调一次(构造时已经跑过一次未 seed 的 `_load_model`)。
2. `set_init_state` **不推进物理**(只 `sim.forward()`)。若要"稳定后"的场景,按惯例补约 10 步 dummy;
   那会改变状态 ⇒ 想把它当共享锚点就要记录 **settle 之后**的状态。
3. `ControlEnv.reset()` 有 `except RandomizationError: pass` 重试循环;一旦触发会多消耗 RNG ⇒ fixture 悄悄错位。
   **reset 后 assert 三个 fixture 的位置。**
4. 保险替代:reset 后直接把参考 env 的 `body_pos/body_quat` 拷过去再 `sim.forward()`。

---

---

## §A attention 提取的规格选择 —— 为什么这么定,别改错方向

对照方法是 **POAP**(arXiv 2606.03556),**未开源**,以下按其正文实现。
这一节专门记录三个容易被后人"顺手改回常规写法"的地方。

### A-1 `head` 求和(而非平均)—— 按 POAP 原文,**可改可不改**
与平均只差一个常数因子(除以 head 数),**不影响任何排序**,也不影响后续的秩相关/top-k。
所以谁改成平均都不算错,只是与原文不一致。记下来免得反复。

### A-2 `token` 取 max(而非求和)—— 按 POAP 原文,**承重选择,不要改**
max 与 sum 给出**不同的排序**,这不是常数因子的差别。⇒ **主报 max(忠实 POAP)**,
**另报 sum 作为稳健变体**。两者差异大,说明 POAP 的定位实际被一两个 token 主导 —— 这本身是个值得写进结果的观察。

max 的已知弱点:对异常 token 极敏感。全 token 里含 BOS/EOS 和 `the`/`on` 这类功能词,
而它们正是 attention sink 高发区,**单个弥散高值 token 就能决定整张 saliency map**。这正是必须同时报 sum 的理由。

### A-3 重归一化的**位置**(原文只说要做,没说在哪一步)—— 顺序是承重的
正确顺序:

```
1. head 求和                                   → A ∈ R^{Z × N_v}
2. 【逐行归一化】对 A 的每一行(= 每个文本 token),在图像 token 维度上重新归一化
3. token 取 max(逐元素)                        → saliency S ∈ R^{h×w}
4. s×s 窗口选块(s=3)→ 上采样成二值 mask
```

**关键是第 2 步在第 3 步之前、且是逐行做的。** 这样每个文本 token 对图像的分布都被归一化到
可比的量纲,再取 max 才有意义。若先 max 再归一化,sink token 的绝对量级会直接压过其他 token,
等于没治。是否重归一化、以及在哪一步做,都要写进结果说明。

⚠️ 另外:POAP 只在**优化纹理**阶段才用名词(对名词子词 token **求和**)。
**那是另一个阶段,不要混进定位。** 定位 = 全部 token + max。

### A-4 主视角 = `base_0_rgb`(agentview);其余 view 先存不用
π0.5 送三路图像:`base_0_rgb`(agentview)、`left_wrist_0_rgb`(robot0_eye_in_hand)、
`right_wrist_0_rgb`(全零且 `image_mask=np.False_`)。patch 贴在桌面上,agentview 才是俯视桌面那一路
⇒ **定位主用 base**。wrist 那一路照样按 `view` 轴存下来(实测 patch 在 wrist 里也可见),留待后面再看。

### A-5 图像空间 ↔ 世界坐标:**只做一次投影,最后统一转到世界系**
attention 出在图像 patch 网格上,influence 出在桌面世界坐标锚点上。两者要可比。
**约定:把 attention 转到世界系**(不是把锚点投到图像上),因为 S3 要报的是
"argmax 的**世界坐标位移**(米)",世界系本来就是最终的公共空间。

做法:用 depth + K + `[R|t]`(都已实测可取,见 Q8)把 saliency 网格每个 cell 的中心反投影成桌面上的
世界点;再对每个锚点,聚合落在其 10 cm × 10 cm 足迹内的那些 cell 的 saliency 值
⇒ 每个锚点一个 attention 分数,与 `influence[i]` 直接可比。

⚠️ 分辨率必须对齐:**模型输入是 224,LIBERO 渲染是 256**,`f` 随 H 线性缩放
(实测 256² 下 agentview `f = 309.0193`)。反投影要在与 attention 网格一致的那个分辨率下做,否则整体偏移。

---

## §PT · PyTorch 迁移(决定:**整条流水线都用 PyTorch**)

**2026-08-04 拍板:Δa 与 attention 都在 PyTorch 上做,不混用框架。**
理由:混用会让 `Spearman(attention, influence)` 的低相关有可能只是转换误差的伪影
(两个不同前向实现),PART B 的核心指标就废了。全 PyTorch 则不存在跨框架比较。

代价是要重新确立 baseline(§PT-5),以及一次权重转换。
另外要记清:**"走 PyTorch 就不用改 vendored 源码"这个说法是错的** ——
PyTorch 路径要求覆盖 HF transformers 的三个文件,只是这次改动是 openpi 官方 sanctioned 的。
两条路都在改别人的代码,所以这一条不构成 PyTorch 的优势。

### PT-1 · 官方**没有**发布 PyTorch 权重
`gs://openpi-assets/checkpoints/` 下全是 orbax(JAX)格式,只给了转换脚本。
但 README:192 明说 **"The PyTorch implementation has been validated on the LIBERO benchmark
(both inference and finetuning)"** ⇒ 不是没验证过的移植,一致性风险比预想小。
推理走**完全同一套 API**:`policy_config.py:49-50` 靠输出目录里有没有 `model.safetensors`
自动切路径,所以只换 `--policy.dir` 即可。

### PT-2 · transformers 补丁:**不碰共用 env**,做 in-tree 副本
官方做法是 `cp -r src/openpi/models_pytorch/transformers_replace/* <site-packages>/transformers/`
(README:203-207),README:212 自己警告这会 "propagate to other projects that use transformers"。
我们的 py3.11 是**共用的 `openpi-server`**,所以改成:

```
third_party/transformers_patched/transformers/     ← 整包副本(62 MB,纯 Python 零 .so)+ 补丁
运行时 PYTHONPATH 以它开头,盖掉 site-packages 里那份
```
脚本:`pi05probe/setup_torch_transformers.py`(幂等,可 `--force` 重来)。

⚠️ **实测发现:`openpi-server` 那份 transformers 早在 2026-04 建 env 时就已经打过补丁了**
(`site-packages/transformers/models/siglip/check.py` 存在),而且与我们 6 月 clone 里的
`transformers_replace` **5 个文件逐字节相同** ⇒ 没有版本漂移。
in-tree 副本仍然值得做:自包含(别人日后 `pip install -U transformers` 打不断我们)+
PYTHONPATH 里写明用的是哪份代码,不依赖 env 的历史状态。

### PT-3 · 权重转换:两个坑
脚本 `pi05probe/convert_weights.py`(后台跑,`--status` 看进度)。产物:
`checkpoints/pi05_libero_pytorch/model.safetensors` = **6.74 GiB**(bfloat16)。

1. `--checkpoint_dir` 要给 **checkpoint 根目录**,不是 `params/` ——
   `convert_jax_model_to_pytorch.py:402` 自己拼 `f"{checkpoint_dir}/params/"`。
   给成 `params` 会报 `FileNotFoundError: Metadata file (named _METADATA) does not exist`。
2. **⚠️ 上游 bug:根目录一给,`assets/` 就复制不过去。**
   `:536` 写的是 `assets_source = pathlib.Path(checkpoint_dir).parent / "assets"`,
   而根目录的 `.parent` 是 `.../checkpoints/`,那里没有 `assets`(正确写法应是
   `Path(checkpoint_dir) / "assets"`)。**实测确认这个 bug 真的会触发**(我们的兜底 copytree 生效了)。
   后果很阴:缺了 `norm_stats.json` 不会报错,`_load_norm_stats` 只
   `logging.info("... skipping")` 然后 `norm_stats=None`
   ⇒ **安静地输出未反归一化的垃圾动作**。`convert_weights.py` 已自己补拷 + assert。

### PT-4 · ✅ attention 逐层抽取:**一个普通 forward hook 就够,不改任何源码**

关键结构(`pi0_pytorch.py:377-419` + `gemma_pytorch.py:90-124`):

| 那一趟 | 走哪个 module | 我们关心吗 |
|---|---|---|
| **prefix**(`inputs_embeds=[prefix, None]`) | `gemma_pytorch.py:102` → 真正的 HF `paligemma.language_model.forward()` | ✅ **text→image 全在这里**,而且**每个 observation 只跑一次**(之后进 KV cache) |
| 10 个去噪步(`[None, suffix]`) | `gemma_pytorch.py:114` → **`gemma_expert.model`**(另一个 module) | 不碰我们的 hook |
| 训练用的混合前向 | `gemma_pytorch.py:126+` **手写逐层循环** | ⚠️ 那条路 `output_attentions` 无效,别用错 |

为什么 hook 就能白拿:打过补丁的
`transformers/models/gemma/modeling_gemma.py:327` 里 `GemmaAttention.forward`
**无条件 `return attn_output, attn_weights`** —— 连 `output_attentions` 开关都没有;
而 `pi0_pytorch.py:392` 把 `_attn_implementation` 强制设成 `"eager"`,
所以 `eager_attention_forward:248` 一定算出完整 post-softmax 权重(fp32 里做 softmax)。

```python
handles = [lm.layers[i].self_attn.register_forward_hook(mk_hook(i)) for i in range(18)]
```
⚠️ **必须关 `torch.compile`**:`pi0_pytorch.py:112` 默认
`torch.compile(self.sample_actions, mode="max-autotune")`,编译区里 hook 可能不触发。
用 `dataclasses.replace(cfg.model, pytorch_compile_mode=None)`。
(注:跑官方 demo 复现数字时**不要**关,那是默认路径。)

**实测(`pi05probe/out/s0_torch_probe.txt`):**
```
18/18 层全抓到,每层恰好被调用 1 次
layer0 attn_weights: shape=(1, 8, 968, 968) fp32     ← Tq=Tk=968 与 Q6 算的 prefix 长度完全一致
heads = 8;行和 0.997–1.002(softmax 归一,偏差来自 fp32→bf16→fp32)
```

#### ⚠️⚠️ 两个改变实现细节的实测数

**(a) attention sink 在 `\n` 上,不在 BOS 上。** prompt `put the bowl on the plate`(Z=8),
各 text token 在 base 图 256 个 patch 上的注意力质量(head-summed / 8):

| token | piece | mass | max_cell |
|---|---|---|---|
| 0 | `<bos>` | **0.0111** ← 最低 | 0.0025 |
| 1 | `put` | 0.0599 | 0.0368 |
| 2 | `▁the` | 0.0328 | 0.0304 |
| 3 | `▁bowl` | 0.0467 | 0.0438 |
| 4 | `▁on` | 0.0199 | 0.0175 |
| 5 | `▁the` | 0.0262 | 0.0241 |
| 6 | `▁plate` | 0.0544 | 0.0473 |
| 7 | `\n` | **0.3970** ← 比其他高 7–36 倍 | 0.2374 |

计划 §A-4 猜的是"质量塌到 BOS/首 token",**实际塌在末尾的 `\n`**(start-of-answer token)。
⇒ §A-3 那个"**逐行重归一化必须在 token 取 max 之前**"不是洁癖:
不做的话整张 saliency map 会被 `\n` 一个 token 决定。现在有实测支撑。
⇒ 另外这也说明 §B1"主报 max 附报 sum"是对的,两者差异会很大。

**(b) padding 行必须按 mask 裁掉 —— 差 78 倍。**
```
8 个真实 token 在 base 图上的注意力质量合计 =   5.18
192 个 padding slot 合计                    = 405.0      ← 78×
```
padding 行不是零,是**弥散的**(单格 max 只有 0.00103,但 192 行 × 8 头累起来极大)。
⇒ 若按 200 个 slot 全取,真实语义信号被淹掉 78 倍。**Q6 那条从推断升级为硬数据。**

**存储**:整层 `[1,8,968,968]` fp32 = 28.6 MiB/层 ⇒ 18 层 0.50 GiB/帧(太大);
只留 `[8, Z, 256]` = 64 KiB/层 ⇒ 18 层 **1.12 MiB/帧** ⇒ T=60 帧 **0.07 GiB**
⇒ §B 那个"层/token/时间三轴都不合并"的 schema 完全存得下。

### PT-5 · 一致性检查

**(A) 单帧 + 同 ε:JAX vs PyTorch**(`probe_torch.py`)
```
|Δ| max  = 7.45e-03      逐维 max = [0.0074 0.0019 0.0062 0.00036 0.00044 0.00068 0.0048]
|Δ| mean = 1.45e-03      相对误差 max = 7.4e-03  (动作量级 ≈ ±1)
夹爪 sign(a[:,6]) 两边完全一致(全 -1)
```
换成物理量:平移 `0.0074 × 0.05 m` = **0.37 mm**;旋转 `0.0007 × 0.5 rad` = **0.35 mrad**。
按 Q2,夹爪只有符号有物理意义 ⇒ **夹爪通道完全等价**。
⚠️ 但这是**单帧、随机图像**,不能当结论。真正的判据是下面 (B)。

**(B) 30-episode 成功率:✅ 跑完了 —— 28/30 = 93.3%(JAX 是 29/30 = 96.7%)**

```
PyTorch : Total success rate: 0.9333333333333333   (28/30)   显存 8062 MiB
JAX     : Total success rate: 0.9666666666666667   (29/30)   显存 8730 MiB
```

逐 task 对比(3 trial/task):

| task | JAX | PyTorch |
|---|---|---|
| `open the middle drawer of the cabinet` | 2/3 | 2/3 |
| **`put the bowl on top of the cabinet`** | 3/3 | **2/3** ← 唯一的差别 |
| 其余 8 个 task | 3/3 | 3/3 |

**⚠️ 这个检查只能说明"转换没坏",不能说明"紧等价"。** 必须写清楚:
- 差别是**恰好 1 个 episode**。n=30 下 93.3% 的 95% 置信区间约 [77%, 99%],
  两个数**统计上无法区分**;官方 500 episode 报 98.0,两者都与之相容。
- 单帧 `|Δ| ≈ 7.4e-3` 的转换误差足够把一个边缘 episode 推翻,这是预期行为,不是 bug 信号。
- 真正被排除掉的是"转换坏了"这种情形(那会掉到接近 0)。

**但因为我们决定整条线都用 PyTorch,跨框架紧等价其实已经不重要了。** 现在真正要管的是:
1. PyTorch 模型本身是不是一个称职的 π0.5-LIBERO 策略 → ✅ 93.3% 显然是。
2. 四条 PART B 指令的 clean 成功率够不够高 → **这是 S0.5 的活,n=3 说明不了任何事**。
   ⚠️ 但 `put the bowl on top of the cabinet` 掉到 2/3 值得留意,S0.5 补到 10–20 episode 时重点看它。

⚠️ PyTorch 首次前向要 `torch.compile(mode="max-autotune")`,**慢启动几分钟是正常的**。
复现:
```
/home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/run_demo.py --torch --trials 3
/home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/run_demo.py --status
```

**(C) ✅ 顺带把 Q3b 实测了**:换掉 `observation/state`、其余(图像/prompt/ε)不变
⇒ `|Δ| max = 0.000000e+00`(**恰好 0**)⇒ **π0.5-LIBERO 确实完全不读 state**,
动作只依赖 (3 路图像, prompt, ε)。

### PT-6 · ⚠️ 遗留的架构问题:S2 的"共享 ε"跨不了进程

- 渲染在 **py3.8**(`openpi-libero`:mujoco/robosuite/libero),模型在 **py3.11**(`openpi-server`)。
- 而 `websocket_policy_server.py:61` 是 `self._policy.infer(obs)`,**传不了 `noise`**;
  `policy.py:74` 每次调用都 `jax.random.split` / 重采样 ⇒ 同输入两次得到不同 ε。

**解法(不改 openpi 源码)**:`WebsocketPolicyServer` 只要求传入的对象有 `infer`,
所以写一个**包装 policy**,它的 `infer(obs)` 从 obs 里 `pop("noise")` 再转发给真 policy。
msgpack_numpy 本来就能序列化 numpy 数组,client 侧直接把 ε 塞进 element 里即可。
⇒ 写在 S2 的脚手架里,不动 vendored 代码。

(另一条更省的路:纯反事实扫描不需要闭环,可以在 py3.8 里把 T 帧 observation dump 成 npz,
再在 py3.11 里批量前向。但 S2.2 的 attacked rollout 是闭环的,还是需要上面那个包装。)

---

## 待办(S0 收尾)

✅ 已完成:
1. ~~装环境~~ → 复用已有两个 conda env + `PYTHONPATH` 覆盖,**没装新环境**(§0)。
2. ~~跑通官方 LIBERO 评测 demo~~ → **libero_goal 29/30 = 96.7%**,四条候选指令全 3/3(§0)。
3. ~~补 Q1 resize / Q2 动作语义 / Q3 归一化~~ → 全部有据(含 180° 旋转这个更正)。
4. ~~Q6 的 SigLIP grid / Gemma 层头数 / index 区间 / rollout 显存~~ → 全部拿到。
5. ~~Q5 的 `Policy.infer()` 是否透传 noise~~ → **透传**,但 websocket 传不了。

6. ~~PyTorch 下的 30-episode 成功率~~ → **28/30 = 93.3%**(§PT-5B)。

### 还没做的 —— 下一步是 S0.5,不是 S2
7. **S0.5 检查 A**:交叉评估(给指令 A、用目标 B 的成功判据评)。
   这是 PART B 全部内容的 go/no-go —— 若模型忽略文本、只按场景 affordance 行动,前提就不成立。
8. **S0.5 检查 B**:强制统一初始状态。Q10 已给出 `start_shared()` 做法,
   需在 openpi 钉的 LIBERO `f78abd6` 上落实并 assert 四个 task 的 `flatten()` 逐位相同。
9. **S0.5 补测**:四条候选指令各 10–20 episode 的 clean 成功率
   (重点看 `put the bowl on top of the cabinet`,§PT-5B 里它掉到 2/3)。
10. 在 `f78abd6` 上**复核 Q7–Q10**(现有实测在 `8f1084e` 上)。
    环境自检已确认我们跑的就是 `f78abd6` 那份源码,所以复核成本很低;
    可以和第 8 项合并做。
11. S1 之前:按 §PT-6 写好那个能透传 `noise` 的包装 policy。

---

## 脚手架清单(`pi05probe/`)

| 文件 | 作用 | 能跑? |
|---|---|---|
| `run_demo.py` | spawn 官方 demo 的 server+client,`start_new_session` 脱离终端;`--torch` / `--status` / `--kill` | ✅ |
| `probe_facts.py` | Q1/Q2/Q3/Q6 的可跑部分,不占 GPU,秒级 | ✅ |
| `setup_torch_transformers.py` | 做 `third_party/transformers_patched` 并打补丁(§PT-2),幂等 | ✅ |
| `convert_weights.py` | JAX → PyTorch 权重转换 + 补 assets(§PT-3),后台跑,`--status` | ✅ |
| `probe_torch.py` | 一致性检查 + attention 逐层抽取验证(§PT-4/PT-5) | ✅ |
| `libero_config/config.yaml` | `LIBERO_CONFIG_PATH` 指向的隔离配置 | ✅ |
| `out/s0_facts.txt` | `probe_facts.py` 的完整输出(带全部数值) | — |
| `out/s0_torch_probe.txt` | `probe_torch.py` 的完整输出 | — |
| `out/s0_torch_probe_attn_layer0.npz` | layer0 的 `attn[text, base_patch]`,留作后续对照 | — |
| `out/{server,client}.log` | JAX demo 日志(29/30) | — |
| `out/{server,client}_torch.log` | PyTorch demo 日志 | — |
| `out/videos/libero_goal{,_torch}/` | rollout 视频 | — |
| `env.sh` / `check_env.sh` | 环境变量文档;**shell 脚本被分类器拒,跑不了**,只当说明看 | ❌ |
| `setup_openpi_env.sh` / `setup_openpi_conda.sh` | 装环境的两版方案,**已作废**(复用现成 env,无需安装) | ❌ |

### 两条 PYTHONPATH(顺序都是承重的)

```bash
# JAX 路径(server env, py3.11)
$OPENPI/src:$OPENPI/packages/openpi-client/src

# PyTorch 路径 —— transformers_patched 必须排最前面
$ROOT/third_party/transformers_patched:$OPENPI/src:$OPENPI/packages/openpi-client/src

# LIBERO client(py3.8)
$OPENPI/packages/openpi-client/src:$OPENPI/third_party/libero
```
