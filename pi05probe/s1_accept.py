#!/usr/bin/env python3
"""S1 验收:注入的 3D patch 是否 ① 合法放置且不破坏 geom_id ② 真被机械臂遮挡
③ 真有透视形变 ④ 纹理不被降采样抹平。四项全过才可进 S2。

计划把 ②③ 列为"重中之重":唯一能在烧算力前发现"geom 其实没被遮挡/不形变"的关卡
—— 这类错误不报错,只让热图看起来正常而结论全错。

全部纯渲染(mujoco,py3.8),不需要 policy server:
  ②③ 用已有的 clean rollout(traj_*.npz 每帧 flatten)回放进**注入了 patch 的 env**,
  patch 是静态 geom,机械臂按录好的 qpos 扫过 ⇒ 遮挡/形变都能看出来。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/s1_accept.py
"""
import os
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out"
FIND = OUT / "findings"

sys.path.insert(0, str(PROBE))                       # scene_patch
for p in reversed([OPENPI / "packages" / "openpi-client" / "src", OPENPI / "third_party" / "libero"]):
    sys.path.insert(0, str(p))
os.environ["LIBERO_CONFIG_PATH"] = str(PROBE / "libero_config")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYTHONNOUSERSITE"] = "1"

import numpy as np
import yaml
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scene_patch as sp

CAM = "agentview"
RES = 256
SHARED_TASK_BDDL = "put_the_bowl_on_the_plate"   # traj 就是这条


def build_env(bddl, world_pos, texture_path, seed):
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES,
                             camera_segmentations="element")
    if world_pos is not None:
        env.env.set_xml_processor(sp.make_xml_processor(CFG, world_pos, str(texture_path)))
    env.seed(seed)
    env.reset()
    return env


def bddl_path(stem):
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        if pathlib.Path(t.bddl_file).stem == stem:
            return pathlib.Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file
    raise AssertionError(stem)


def seg_of(obs):
    return obs[f"{CAM}_segmentation_element"][..., 0]


def world_to_px(P, K, E):
    """世界点 → agentview 像素 (u,v)。E=world<-camera(已折叠 diag),故用 inv(E)。"""
    Einv = np.linalg.inv(E)
    Pc = Einv @ np.append(np.asarray(P, float), 1.0)
    x, y, z = Pc[:3]
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return np.array([u, v])


def save_rgb(arr, path):
    Image.fromarray(np.asarray(arr, np.uint8), "RGB").save(path)


def main():
    global CFG
    CFG = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
    tex = CFG["patch"].get("texture") or (PROBE / "config" / "probe_texture.png")
    FIND.mkdir(parents=True, exist_ok=True)
    seed = int(CFG["shared_seed"])
    bddl = bddl_path(SHARED_TASK_BDDL)

    anchors = sp.make_anchors(CFG)
    legal = [a for a in anchors if a.legal]
    log = []
    def out(s=""):
        print(s, flush=True); log.append(s)

    out("=" * 92)
    out("S1 验收")
    out("=" * 92)
    out(f"锚点网格 {CFG['grid']['n_u']}×{CFG['grid']['n_v']}={len(anchors)}  合法={len(legal)}  纹理={tex}")

    # 回放轨迹(②③用)
    traj = np.load(OUT / f"traj_{SHARED_TASK_BDDL}.npz", allow_pickle=True)
    nfr = int(traj["n_frames"])
    flat = [traj[f"f{k:03d}__flatten"] for k in range(nfr)]
    eef = np.array([traj[f"f{k:03d}__state8"][:2] for k in range(nfr)])   # eef xy

    # ---------- ① 合法放置 + 不破坏 geom_id ----------
    out("\n" + "-" * 92)
    out("① 合法放置 / 注入不重编号既有 geom / patch 有独立 seg id")
    out("-" * 92)
    env0 = build_env(bddl, None, tex, seed)
    obs0 = env0.regenerate_obs_from_state(flat[0])
    ngeom0 = int(env0.sim.model.ngeom)
    tid0 = int(env0.sim.model.geom_name2id("table_collision"))
    save_rgb(obs0[f"{CAM}_image"], FIND / "s1_check1_clean.png")
    env0.close()

    a = legal[len(legal) // 2]
    env1 = build_env(bddl, a.world, tex, seed)
    obs1 = env1.regenerate_obs_from_state(flat[0])
    ngeom1 = int(env1.sim.model.ngeom)
    tid1 = int(env1.sim.model.geom_name2id("table_collision"))
    gid = sp.patch_geom_id(env1, CFG)
    vpx = sp.visible_px(obs1, CAM, gid)
    save_rgb(obs1[f"{CAM}_image"], FIND / "s1_check1_patched.png")
    seg1 = seg_of(obs1)
    save_rgb(np.stack([(seg1 == gid).astype(np.uint8) * 255] * 3, -1), FIND / "s1_check1_patchseg.png")
    out(f"  合法锚点[{a.index}] u={a.u:.3f} v={a.v:.3f} world=({a.world[0]:.3f},{a.world[1]:.3f},{a.world[2]:.4f})")
    out(f"  ngeom: clean={ngeom0}  patched={ngeom1}  (期望 +1)")
    out(f"  table_collision geom_id: clean={tid0}  patched={tid1}  (期望不变)")
    out(f"  patch geom_id={gid}  agentview 可见像素={vpx}")
    c1 = (ngeom1 == ngeom0 + 1) and (tid1 == tid0) and (vpx > 0)
    out(f"  {'✅' if c1 else '❌'} ① {'通过' if c1 else '未过'}")
    env1.close()

    # ---------- ② 遮挡真的发生 ----------
    out("\n" + "-" * 92)
    out("② 遮挡:把 patch 放在 eef 路径中点下方,回放整条 clean 轨迹,看可见像素是否随臂经过下降")
    out("-" * 92)
    # 放在 eef 路径中点(取 x,y 的中位)下方的桌面上
    pxy = np.median(eef, axis=0)
    z_top = CFG["plane"]["origin"][2] + CFG["patch"]["thickness"] / 2 + CFG["patch"]["normal_offset"]
    demo_pos = np.array([pxy[0], pxy[1], z_top])
    env2 = build_env(bddl, demo_pos, tex, seed)
    gid2 = sp.patch_geom_id(env2, CFG)
    vis, dist = [], []
    frames_rgb = []
    for k in range(nfr):
        obs = env2.regenerate_obs_from_state(flat[k])
        vis.append(sp.visible_px(obs, CAM, gid2))
        dist.append(float(np.linalg.norm(eef[k] - pxy)))
        frames_rgb.append(obs[f"{CAM}_image"])
    vis, dist = np.array(vis), np.array(dist)
    kmin, kmax = int(vis.argmin()), int(vis.argmax())
    out(f"  patch 放于 eef 路径中点下方 world=({demo_pos[0]:.3f},{demo_pos[1]:.3f},{demo_pos[2]:.4f})")
    out(f"  可见像素逐帧: {vis.tolist()}")
    out(f"  max={vis.max()}(帧{kmax}) min={vis.min()}(帧{kmin})  min/max={vis.min()/max(vis.max(),1):.3f}")
    out("  注:遮挡由**整条手臂**(连杆/前臂)在 agentview 斜视角下挡住 patch 造成,")
    out(f"     不只是夹爪 eef 在正上方 ⇒ 最遮挡帧的 eef 距离不必最小(帧{kmin} eef 距 {dist[kmin]:.3f} m)。")
    # 曲线图(ASCII 标签,避免 CJK 字体缺失)
    fig, ax1 = plt.subplots(figsize=(7, 3.2))
    ax1.plot(vis, "o-", color="tab:red", label="patch visible px")
    ax1.set_xlabel("replan frame"); ax1.set_ylabel("patch visible px", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(dist, "s--", color="tab:blue", label="eef->patch dist (m)")
    ax2.set_ylabel("eef->patch dist (m)", color="tab:blue")
    ax1.set_title("Check 2: patch occluded as arm sweeps over it")
    fig.tight_layout(); fig.savefig(FIND / "s1_check2_occlusion_curve.png", dpi=110); plt.close(fig)
    # 前后对比帧(可见最多 vs 完全遮挡)
    pair = np.concatenate([frames_rgb[kmax], frames_rgb[kmin]], axis=1)
    save_rgb(pair, FIND / "s1_check2_frames_maxmin.png")
    # 判据:可见像素随臂经过明显下降(这里直接到 0)。遮挡是整臂造成,不绑 eef 距离。
    c2 = vis.min() < 0.3 * vis.max()
    out(f"  {'✅' if c2 else '❌'} ② {'通过' if c2 else '未过'}(可见像素随臂经过明显下降,{vis.max()}→{vis.min()})")
    env2.close()

    # ---------- ③ 透视真的形变 ----------
    out("\n" + "-" * 92)
    out("③ 透视:patch 顶面 4 世界角投影到 agentview 应是**梯形**(非矩形);近/远锚点像素面积不同")
    out("-" * 92)
    sf = np.load(OUT / "shared_frame.npz", allow_pickle=True)
    K = sf[f"{SHARED_TASK_BDDL}__K_agentview"]; E = sf[f"{SHARED_TASK_BDDL}__E_agentview"]
    w, h = CFG["patch"]["size_wh"]
    # 近锚点(离相机近)与远锚点:agentview 在 world (0.659,0,1.61),取 x 最大/最小的合法锚
    near = max(legal, key=lambda a: a.world[0])    # x 大 => 离相机(x=0.66)近
    far = min(legal, key=lambda a: a.world[0])
    areas = {}
    quads = {}
    for tag, an in [("near", near), ("far", far)]:
        cx, cy, cz = an.world
        corners = np.array([[cx - w/2, cy - h/2, cz], [cx + w/2, cy - h/2, cz],
                            [cx + w/2, cy + h/2, cz], [cx - w/2, cy + h/2, cz]])
        px = np.array([world_to_px(c, K, E) for c in corners])
        quads[tag] = px
        # 多边形面积(shoelace)
        x_, y_ = px[:, 0], px[:, 1]
        areas[tag] = 0.5 * abs(np.dot(x_, np.roll(y_, 1)) - np.dot(y_, np.roll(x_, 1)))
    # 梯形判据:上边与下边像素长度不等(透视)
    def edge_len(q, i, j):
        return np.linalg.norm(q[i] - q[j])
    q = quads["near"]
    top_len = edge_len(q, 0, 1); bot_len = edge_len(q, 3, 2)
    out(f"  近锚点 world x={near.world[0]:.3f}  投影面积={areas['near']:.0f} px²")
    out(f"  远锚点 world x={far.world[0]:.3f}   投影面积={areas['far']:.0f} px²  (近应>远)")
    out(f"  近锚点投影四边形:上边={top_len:.2f}px 下边={bot_len:.2f}px  比值={top_len/bot_len:.3f}(=1 才是矩形)")
    # 画出投影四边形叠在 patched 渲染上
    envn = build_env(bddl, near.world, tex, seed)
    obsn = envn.regenerate_obs_from_state(flat[0])
    img = np.array(obsn[f"{CAM}_image"], np.uint8).copy()
    qq = quads["near"].astype(int)
    for i in range(4):
        p0, p1 = qq[i], qq[(i + 1) % 4]
        n = max(abs(p1 - p0).max(), 1)
        for t in np.linspace(0, 1, n * 2):
            u, v = (p0 + t * (p1 - p0)).astype(int)
            if 0 <= v < RES and 0 <= u < RES:
                img[v, u] = [0, 255, 255]
    save_rgb(img, FIND / "s1_check3_perspective_quad.png")
    envn.close()
    c3 = (areas["near"] > areas["far"]) and (abs(top_len / bot_len - 1.0) > 0.02)
    out(f"  {'✅' if c3 else '❌'} ③ {'通过' if c3 else '未过'}(近>远 且 上下边不等长=梯形)")

    # ---------- ④ 纹理不被抹平 ----------
    out("\n" + "-" * 92)
    out("④ 纹理:patch 放最远锚点,裁出放大,色块结构应仍在(非糊成单色)")
    out("-" * 92)
    envf = build_env(bddl, far.world, tex, seed)
    obsf = envf.regenerate_obs_from_state(flat[0])
    gidf = sp.patch_geom_id(envf, CFG)
    segf = seg_of(obsf)
    ys, xs = np.where(segf == gidf)
    rgbf = np.array(obsf[f"{CAM}_image"], np.uint8)
    if len(xs) > 0:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = rgbf[y0:y1, x0:x1]
        up = np.array(Image.fromarray(crop).resize((crop.shape[1] * 10, crop.shape[0] * 10), Image.NEAREST))
        save_rgb(up, FIND / "s1_check4_texture_crop.png")
        # patch 像素内的颜色多样性(抹平则接近 1 种)
        patch_px = rgbf[segf == gidf]
        uniq = len(np.unique(patch_px, axis=0))
        out(f"  最远锚点 x={far.world[0]:.3f}  可见像素={len(xs)}  裁剪 {crop.shape[:2]}  patch 内唯一色={uniq}")
        c4 = uniq >= 5
    else:
        out("  ❌ 最远锚点 patch 不可见"); c4 = False
    out(f"  {'✅' if c4 else '❌'} ④ {'通过' if c4 else '未过'}(纹理结构未被抹平)")
    envf.close()

    out("\n" + "=" * 92)
    allok = c1 and c2 and c3 and c4
    out(f"S1 验收总结:① {'✅' if c1 else '❌'}  ② {'✅' if c2 else '❌'}  "
        f"③ {'✅' if c3 else '❌'}  ④ {'✅' if c4 else '❌'}  ⇒ "
        + ("✅ 四项全过,可进 S2。" if allok else "❌ 有未过项,先修再进 S2。"))
    out(f"证据文件:{FIND}/s1_check*.png")
    (FIND / "s1_accept.txt").write_text("\n".join(log) + "\n")
    print(f"\nwrote {FIND}/s1_accept.txt")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
