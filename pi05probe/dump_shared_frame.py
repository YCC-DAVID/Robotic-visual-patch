#!/usr/bin/env python3
"""PART B 第 1 步(= S0.5 检查 B + Q7–Q10 在 f78abd6 上复核):
**强制统一初始状态**,并把四条候选指令共用的那一帧 dump 成 npz。

为什么必须先做这个
------------------
B1 的前提是"同一场景、只换文本"。而 FINDINGS Q10 实测:
  · 十个 `.pruned_init` **全部互不相同**(同一 episode index 跨 task max diff 0.093);
  · 更坑的是三个 fixture(`wooden_cabinet_1 / flat_stove_1 / wine_rack_1`)**不在 qpos 里**,
    `set_init_state` 管不到,每次 `reset()` 都用**全局 numpy RNG** 重新采样
    (`bddl_base_domain.py:769-779` 走 `sim.model.body_pos`)。
    搞错的代价:状态 maxdiff = 0 但 agentview 有 20.1% 的像素不同。
⇒ 不统一的话,"attention 因为换指令而变化"里会混进"物体挪了位置"。

做法(Q10 已实测可用):**每一次 reset() 之前重新 seed**,再 set_init_state。
`env.seed()` 就是 `np.random.seed()`(`bddl_base_domain.py:162-163`),是全局的;
构造 env 时已经跑过一次未 seed 的 `_load_model`,所以构造时 seed 一次不够。

跑在 py3.8 的 openpi-libero 环境(mujoco/robosuite/libero 在那儿)。
模型在 py3.11,两边不同进程 ⇒ 用 npz 交接。
attention 出在 prefix 那一趟、与 flow matching 的 ε 无关,所以静态帧 dump 完全够用。

用法:
    /home/user1/miniconda3/envs/openpi-libero/bin/python pi05probe/dump_shared_frame.py
"""

import hashlib
import os
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
OUT = ROOT / "pi05probe" / "out"

# --- 环境全在脚本内设好:命令行只要 `<python> <这个文件>`,不带前缀 ---
for p in reversed([OPENPI / "packages" / "openpi-client" / "src", OPENPI / "third_party" / "libero"]):
    sys.path.insert(0, str(p))
os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / "pi05probe" / "libero_config")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYTHONNOUSERSITE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

# ---------------------------------------------------------------- 共享锚点的定义
# 状态取自四条候选指令中的一条(而不是 Q10 里用的 open_the_middle_drawer),
# 这样至少有一个 task 用的是它自己的自然分布;状态跨 task 可迁移已由 Q10 验证。
SHARED_SEED = 10000
STATE_SOURCE_TASK = "put_the_bowl_on_the_plate"
SHARED_EP = 0
NUM_STEPS_WAIT = 10                      # 与 examples/libero/main.py:37 一致
DUMMY_ACTION = [0.0] * 6 + [-1.0]        # main.py:17,"不动 + 张开"
RES = 256                                # LIBERO_ENV_RESOLUTION
RESIZE = 224                             # main.py:28

# PART B / B1 的四条指令(计划里已核对过 bddl 文件名)
TASKS = [
    "turn_on_the_stove",
    "put_the_wine_bottle_on_the_rack",
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_top_of_the_cabinet",
]
# ⚠️ mujoco 里的 body 名带后缀:实际是 wooden_cabinet_1_main / _base / _cabinet_top …
# `set_init_state` 管不到的那三个 fixture 走的是根 body(`_main`)的 sim.model.body_pos,
# 但我们把所有匹配前缀的 body 都记下来,免得漏掉某个也会动的子 body。
FIXTURE_PREFIXES = ["wooden_cabinet_1", "flat_stove_1", "wine_rack_1"]

_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x)).hexdigest()[:12]


def find_task(suite, stem):
    """按 bddl 文件名找 task。计划里明确:指令串以**文件名**为准。"""
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        if pathlib.Path(t.bddl_file).stem == stem:
            return i, t
    raise KeyError(f"libero_goal 里找不到 {stem}")


def make_env(bddl, with_geom):
    """with_geom=True 时额外开 depth + element segmentation。"""
    from libero.libero.envs import OffScreenRenderEnv
    kw = dict(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    if with_geom:
        kw.update(camera_depths=True, camera_segmentations="element")
    return OffScreenRenderEnv(**kw)


def start_shared(env, state):
    """Q10 的 start_shared:**每次 reset 前重新 seed**,否则 fixture 位置漂。"""
    env.seed(SHARED_SEED)
    env.reset()
    return env.set_init_state(state)


def fixture_bodies(env):
    m = env.env.sim.model
    names = [n for n in m.body_names
             if any(n.startswith(p) for p in FIXTURE_PREFIXES)]
    return sorted(names)


def fixture_pos(env):
    m = env.env.sim.model
    names = fixture_bodies(env)
    return names, np.stack([m.body_pos[m.body_name2id(n)].copy() for n in names])


def cam_intrinsics(env, cam, h):
    """内联算 K(Q8:camera_utils 导不进来,缺 h5py)。"""
    m = env.env.sim.model
    fovy = float(m.cam_fovy[m.camera_name2id(cam)])
    f = 0.5 * h / np.tan(fovy * np.pi / 360.0)
    return np.array([[f, 0, (h - 1) / 2.0], [0, f, (h - 1) / 2.0], [0, 0, 1.0]]), fovy


def cam_extrinsics(env, cam):
    """world<-camera 的 4x4。robosuite 的相机看 -z,故右乘 diag(1,-1,-1,1)。"""
    d, m = env.env.sim.data, env.env.sim.model
    cid = m.camera_name2id(cam)
    T = np.eye(4)
    T[:3, :3] = d.cam_xmat[cid].reshape(3, 3)
    T[:3, 3] = d.cam_xpos[cid]
    return T @ np.diag([1.0, -1.0, -1.0, 1.0])


def depth_to_meters(env, d):
    """Q8 的内联公式:near/(1 - d*(1-near/far))。"""
    m = env.env.sim.model
    extent = float(m.stat.extent)
    near, far = float(m.vis.map.znear) * extent, float(m.vis.map.zfar) * extent
    return near / (1.0 - d * (1.0 - near / far)), extent, near, far


def model_input(obs):
    """main.py:113-122 那一段。⚠️ 是 [::-1, ::-1](180° 旋转),不是单个 [::-1]。"""
    from openpi_client import image_tools
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wri = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return (
        image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE)),
        image_tools.convert_to_uint8(image_tools.resize_with_pad(wri, RESIZE, RESIZE)),
    )


def quat2axisangle(quat):
    """main.py:199-214 的拷贝(robosuite 原版)。"""
    import math
    q = np.array(quat, dtype=np.float64)
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def main():
    from libero.libero import benchmark, get_libero_path
    import libero, robosuite, mujoco

    say("=" * 100)
    say("环境自检")
    say("=" * 100)
    say(f"  libero    @ {libero.__path__[0]}")
    say(f"  robosuite {robosuite.__version__} | mujoco {mujoco.__version__}")
    assert "/WAMattack/" in libero.__path__[0]
    say(f"  LIBERO_CONFIG_PATH = {os.environ['LIBERO_CONFIG_PATH']}")
    say(f"  init_states 目录   = {get_libero_path('init_states')}")

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    say(f"  libero_goal n_tasks = {suite.n_tasks}")

    # ------------------------------------------------------------ 共享状态
    say("")
    say("=" * 100)
    say("1 · 取共享初始状态")
    say("=" * 100)
    src_id, src_task = find_task(suite, STATE_SOURCE_TASK)
    init_path = pathlib.Path(get_libero_path("init_states")) / "libero_goal" / f"{STATE_SOURCE_TASK}.pruned_init"
    # ⚠️ 两个环境的 torch 版本相反着来:
    #   py3.11 server env 是 torch 2.7.1 ⇒ 必须显式 weights_only=False
    #   py3.8  client env 是 torch 1.11.0 ⇒ **根本没有这个参数**,传了会 TypeError
    try:
        init = np.asarray(torch.load(str(init_path), weights_only=False))
    except TypeError:
        init = np.asarray(torch.load(str(init_path)))
    SHARED_STATE = init[SHARED_EP].copy()
    say(f"  来源 task[{src_id}] = {STATE_SOURCE_TASK}  episode {SHARED_EP}")
    say(f"  .pruned_init shape = {init.shape}  dtype={init.dtype}")
    say(f"  SHARED_STATE shape = {SHARED_STATE.shape}  sha={sha(SHARED_STATE)}")
    say(f"  SHARED_SEED = {SHARED_SEED}   warmup = {NUM_STEPS_WAIT} 步 dummy")

    # ------------------------------------------------------------ 逐 task
    say("")
    say("=" * 100)
    say("2 · 四条指令 × 同一初始状态 —— 逐位相同性检查(S0.5 检查 B)")
    say("=" * 100)
    recs = {}
    for stem in TASKS:
        tid, task = find_task(suite, stem)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        say("")
        say(f"  --- task[{tid}] {stem}")
        say(f"      task.language = {task.language!r}")
        lang_from_name = stem.replace("_", " ")
        if task.language != lang_from_name:
            say(f"      ⚠️ 与文件名推导不一致!文件名 ⇒ {lang_from_name!r}")

        env = make_env(bddl, with_geom=False)
        obs = start_shared(env, SHARED_STATE)

        # 【Q10 注 3】reset 里有 `except RandomizationError: pass` 重试循环,
        # 一旦触发会多消耗 RNG ⇒ fixture 悄悄错位。所以必须 assert。
        fx_names, fx = fixture_pos(env)

        # settle:官方 eval 的 10 步 warmup。set_init_state 不推进物理(Q10 注 2),
        # 想把"稳定后"当共享锚点,就必须记 settle 之后的状态。
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(DUMMY_ACTION)

        flat = env.env.sim.get_state().flatten()
        img224, wri224 = model_input(obs)
        state8 = np.concatenate([obs["robot0_eef_pos"],
                                 quat2axisangle(obs["robot0_eef_quat"]),
                                 obs["robot0_gripper_qpos"]])

        # 几何量:另开一个带 depth+seg 的 env,施加同一 flatten 状态后取。
        # 这同时**在 f78abd6 上复核 Q8 的"开 seg 不改 RGB"**。
        env_g = make_env(bddl, with_geom=True)
        env_g.seed(SHARED_SEED); env_g.reset()
        obs_g = env_g.regenerate_obs_from_state(flat)
        rgb_g = obs_g["agentview_image"]
        seg = obs_g["agentview_segmentation_element"]
        dep_raw = obs_g["agentview_depth"]
        dep_m, extent, near, far = depth_to_meters(env_g, dep_raw)
        K, fovy = cam_intrinsics(env_g, "agentview", RES)
        E = cam_extrinsics(env_g, "agentview")
        Kw, _ = cam_intrinsics(env_g, "robot0_eye_in_hand", RES)
        Ew = cam_extrinsics(env_g, "robot0_eye_in_hand")

        # ⚠️ 注意:这里比的是【同一 flatten 状态下】两个 env 的 agentview。
        # 但主 env 的那张是 rollout 实时渲染,几何 env 走的是 regenerate_obs_from_state,
        # Q9 已实测这两条路本身就不同(实时帧的 xpos 滞后一个积分子步)⇒ 差异不全是 seg 的锅。
        rgb_main = obs["agentview_image"]
        ndiff = int((rgb_main != rgb_g).sum())
        say(f"      ngeom={env.env.sim.model.ngeom}  nq={env.env.sim.model.nq}  "
            f"flatten={len(flat)}")
        say(f"      fixture bodies ({len(fx_names)}) = {fx_names}")
        say(f"      fixture body_pos sha = {sha(fx)}")
        say(f"      settle 后 flatten sha = {sha(flat)}")
        say(f"      img224 sha = {sha(img224)}   wrist224 sha = {sha(wri224)}")
        say(f"      seg: 唯一 id 数={len(np.unique(seg))}  depth(m) 范围="
            f"[{dep_m.min():.4f}, {dep_m.max():.4f}]")
        say(f"      主 env 实时帧 vs 几何 env 恢复帧:{ndiff}/{RES*RES*3} 字节不同"
            f"(Q9 已知:恢复路径 != 实时渲染,这里不是 seg 的锅)")

        recs[stem] = dict(
            task_id=tid, language=task.language, bddl=str(bddl),
            img224=img224, wrist224=wri224, state8=state8,
            rgb256=rgb_main, rgb256_geomenv=rgb_g,
            seg=seg, depth_raw=dep_raw, depth_m=dep_m,
            K_agentview=K, E_agentview=E, K_wrist=Kw, E_wrist=Ew,
            fovy_agentview=fovy, depth_extent=extent, depth_near=near, depth_far=far,
            flatten=flat, fixtures=fx, qpos=env.env.sim.data.qpos.copy(),
        )
        env.close(); env_g.close()

    # ------------------------------------------------------------ 判据
    say("")
    say("=" * 100)
    say("3 · 判据:四个 task 是否真的共用同一场景")
    say("=" * 100)
    ref = recs[TASKS[0]]
    ok = True
    for key, label in [("flatten", "sim.get_state().flatten()"),
                       ("fixtures", "三个 fixture 的 body_pos"),
                       ("rgb256", "agentview RGB 256²"),
                       ("img224", "喂模型的 base 图 224²"),
                       ("wrist224", "喂模型的 wrist 图 224²"),
                       ("state8", "observation/state (8,)")]:
        same, worst = True, 0.0
        for stem in TASKS[1:]:
            a, b = np.asarray(ref[key]), np.asarray(recs[stem][key])
            eq = np.array_equal(a, b)
            same &= eq
            worst = max(worst, float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))
        say(f"  {'✅' if same else '❌'} {label:32s} 逐位相同={same}  maxdiff={worst:.3e}")
        ok &= same

    say("")
    if ok:
        say("  ✅ S0.5 检查 B 通过:四条指令面对**逐位相同**的场景,唯一变量是文本。")
    else:
        say("  ❌ 没对齐 —— B1 的结论会混进物体位置差异,**不要继续**,先查 seed/warmup 路径。")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "shared_frame.npz", **{
        f"{stem}__{k}": v for stem, r in recs.items() for k, v in r.items()
        if isinstance(v, (np.ndarray, np.floating, float, int))
    }, shared_state=SHARED_STATE, shared_seed=SHARED_SEED,
        state_source=STATE_SOURCE_TASK, num_steps_wait=NUM_STEPS_WAIT,
        tasks=np.array(TASKS), languages=np.array([recs[s]["language"] for s in TASKS]))
    (OUT / "shared_frame.txt").write_text("\n".join(_lines) + "\n")
    say(f"[written] {OUT/'shared_frame.npz'}")
    say(f"[written] {OUT/'shared_frame.txt'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
