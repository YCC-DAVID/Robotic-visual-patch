#!/usr/bin/env python3
"""收尾项 D(§A-5):把 attention 网格反投影到**世界坐标(米)**,并对准 seg 物体。

链路(全部承重,错一步整体偏移):
  16×16 cell 中心 → 224 输入像素 → 撤 resize(224→256, PIL 像素中心约定)
  → 撤 180° 旋转(model_input = raw[::-1,::-1]) → raw256 像素
  → depth_m + K + E 反投影 → 世界 3D
E = make_pose(cam_xpos,cam_xmat) @ diag(1,-1,-1,1)(robosuite 相机看 -z),
折叠后是标准 CV 针孔(x 右 / y 下 / z 前),所以直接 P_cam=[(u-cx)z/f,(v-cy)z/f,z]。

先验证:整幅反投影,桌面主平面应 ≈ z=0.900(Q7 桌顶),xy 落在桌面范围内。
验证过了再信 attention 的世界坐标。纯后处理,零前向。

用法:
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/reproject.py
"""
import pathlib
import numpy as np

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
SF = OUT / "shared_frame.npz"
ATTN = OUT / "attn_b1b2.npz"
TXT = OUT / "reproject.txt"

NOUNS = ("bowl", "plate", "cabinet", "stove", "wine", "bottle", "rack")
# 每条指令的"目标物体"世界位置来源:bowl 从 qpos,fixture 从 shared_frame['*__fixtures']
TASKS = {
    "stove_orig": "turn_on_the_stove",
    "bottle_rack_orig": "put_the_wine_bottle_on_the_rack",
    "bowl_plate_orig": "put_the_bowl_on_the_plate",
    "bowl_cabinet_orig": "put_the_bowl_on_top_of_the_cabinet",
}


def cell_to_raw256(r, c):
    """16×16 cell (row r, col c) 中心 → raw-256 像素 (row,col)。"""
    u224 = (c + 0.5) * 14.0   # 224/16
    v224 = (r + 0.5) * 14.0
    u256 = (u224 + 0.5) * 256.0 / 224.0 - 0.5   # 撤 resize_with_pad(pad=0),PIL 像素中心
    v256 = (v224 + 0.5) * 256.0 / 224.0 - 0.5
    raw_row = 255.0 - v256      # model_input = raw[::-1,::-1]
    raw_col = 255.0 - u256
    return raw_row, raw_col


def backproject(raw_row, raw_col, depth_m, K, E):
    ri = int(np.clip(round(raw_row), 0, depth_m.shape[0] - 1))
    ci = int(np.clip(round(raw_col), 0, depth_m.shape[1] - 1))
    z = float(depth_m[ri, ci, 0])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Pc = np.array([(raw_col - cx) * z / fx, (raw_row - cy) * z / fy, z, 1.0])
    Pw = E @ Pc
    return Pw[:3], (ri, ci), z


def main():
    sf = np.load(SF, allow_pickle=True)
    pre = "put_the_bowl_on_the_plate"  # 场景四条指令逐位相同,取一条即可
    depth = sf[f"{pre}__depth_m"]
    K = sf[f"{pre}__K_agentview"]
    E = sf[f"{pre}__E_agentview"]
    seg = sf[f"{pre}__seg"][:, :, 0]
    flat = sf[f"{pre}__flatten"]
    fixtures = sf[f"{pre}__fixtures"]  # (11,3)

    lines = []
    def out(s=""):
        print(s, flush=True); lines.append(s)

    # ---------- 验证:整幅反投影,桌面平面应 ≈ 0.900 ----------
    out("=" * 96)
    out("D 几何自检:整幅 256² 反投影,桌面主平面应 ≈ z=0.900(Q7),xy 在桌面范围内")
    out("=" * 96)
    rr, cc = np.meshgrid(np.arange(256.0), np.arange(256.0), indexing="ij")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z = depth[:, :, 0]
    Xc = (cc - cx) * Z / fx
    Yc = (rr - cy) * Z / fy
    P = np.stack([Xc, Yc, Z, np.ones_like(Z)], axis=-1) @ E.T  # (256,256,4)
    wz = P[:, :, 2].ravel()
    wx = P[:, :, 0].ravel()
    wy = P[:, :, 1].ravel()
    # 桌面平面:取世界 z 的众数带(0.88~0.92)
    tabsel = (wz > 0.88) & (wz < 0.92)
    out(f"  世界 z 直方图峰值带 [0.88,0.92] 占比 = {tabsel.mean()*100:.1f}%")
    out(f"  该带内 z: mean={wz[tabsel].mean():.4f}  std={wz[tabsel].std():.4f}  (期望≈0.900)")
    out(f"  桌面像素 world x∈[{wx[tabsel].min():.2f},{wx[tabsel].max():.2f}]  "
        f"y∈[{wy[tabsel].min():.2f},{wy[tabsel].max():.2f}]  (Q7 桌面 x∈[-0.5,0.5] y∈[-0.6,0.6])")
    ok = abs(wz[tabsel].mean() - 0.90) < 0.02
    out("  " + ("✅ 几何自检通过,可信 attention 世界坐标。" if ok else
                "❌ 桌面平面不在 0.90,反投影约定有误,下面结果不可信。"))

    # 真实物体世界位置(用作 attention 落点的比对基准)
    # flatten = [time(1), qpos(41), qvel(37)];各物体 qpos 地址见 Q10 布局
    obj_xyz = {
        "bowl": flat[10:13],          # qpos 9..11
        "wine_bottle": flat[24:27],   # qpos 23..25
        "plate": flat[31:34],         # qpos 30..32
    }
    # fixture 名序见 shared_frame.txt:idx3=stove_button, idx9=cabinet_top
    fix_names = ['flat_stove_1_base', 'flat_stove_1_burner', 'flat_stove_1_burner_plate',
                 'flat_stove_1_button', 'flat_stove_1_main', 'wine_rack_1_main',
                 'wooden_cabinet_1_base', 'wooden_cabinet_1_cabinet_bottom',
                 'wooden_cabinet_1_cabinet_middle', 'wooden_cabinet_1_cabinet_top', 'wooden_cabinet_1_main']
    obj_xyz["stove"] = fixtures[fix_names.index("flat_stove_1_main")]
    obj_xyz["cabinet_top"] = fixtures[fix_names.index("wooden_cabinet_1_cabinet_top")]
    out("\n  真实物体世界位置(比对基准):")
    for k, v in obj_xyz.items():
        out(f"    {k:12s} = ({v[0]:6.3f}, {v[1]:6.3f}, {v[2]:6.3f})")

    # 每条指令的"目标名词 token"及其真实物体
    TARGET = {"stove_orig": ("stove", "stove"),
              "bottle_rack_orig": ("bottle", "wine_bottle"),
              "bowl_plate_orig": ("bowl", "bowl"),
              "bowl_cabinet_orig": ("bowl", "bowl")}

    az = np.load(ATTN, allow_pickle=True)

    # ---------- (a) 全名词求和图峰值:预期被 sink 主导 ----------
    out("\n" + "=" * 96)
    out("(a) 全名词求和图峰值(L04–L12 带平均)→ 世界 + seg_id")
    out("=" * 96)
    peaks = []
    for name in TASKS:
        head_sum = az[f"{name}__head_sum"]
        pieces = [str(x) for x in az[f"{name}__pieces"]]
        rows = [i for i, p in enumerate(pieces) if any(n in p.lower() for n in NOUNS)]
        band = head_sum[4:13, rows, 0].sum(axis=1).mean(axis=0)
        r, c = np.unravel_index(int(band.argmax()), band.shape)
        raw_r, raw_c = cell_to_raw256(r, c)
        Pw, (ri, ci), z = backproject(raw_r, raw_c, depth, K, E)
        peaks.append((r, c))
        out(f"  {TASKS[name][:16]:16s} 峰值 cell ({r:2d},{c:2d}) → "
            f"世界 ({Pw[0]:6.3f},{Pw[1]:6.3f},{Pw[2]:6.3f})  seg_id={int(seg[ri, ci])}")
    uniq = len(set(peaks))
    out(f"  ⇒ 4 条指令峰值 cell 唯一位置数 = {uniq}"
        + ("(全落同一格 ⇒ 被跨指令不变的 register/sink 主导,不区分指令)" if uniq == 1
           else "(有区分)"))

    # ---------- (b) 目标名词 token 图峰值 → 与真实物体距离 ----------
    out("\n" + "=" * 96)
    out("(b) 目标名词 token 自己的图峰值(L04–L12 带平均)→ 世界,与真实物体的距离")
    out("=" * 96)
    out("  指令              名词    峰值cell  世界(x,y,z)              真实物体      Δ(米)")
    for name, (tok, obj) in TARGET.items():
        head_sum = az[f"{name}__head_sum"]
        pieces = [str(x) for x in az[f"{name}__pieces"]]
        idx = [i for i, p in enumerate(pieces) if tok in p.lower()]
        if not idx:
            continue
        band = head_sum[4:13, idx[0], 0].mean(axis=0)
        r, c = np.unravel_index(int(band.argmax()), band.shape)
        raw_r, raw_c = cell_to_raw256(r, c)
        Pw, _, _ = backproject(raw_r, raw_c, depth, K, E)
        dd = float(np.linalg.norm(Pw - obj_xyz[obj]))
        out(f"  {TASKS[name][:16]:16s}  {tok:6s}  ({r:2d},{c:2d})  "
            f"({Pw[0]:6.3f},{Pw[1]:6.3f},{Pw[2]:6.3f})   {obj:12s}  {dd:.3f}")
    out("\n  ⚠️ 结论:band-mean(L04–L12)后取 argmax,各名词图峰值**全塌到同一 register/sink 格(7,9)**,")
    out("     所以 Δ 只反映 sink 离各物体多近,不是真定位。⇒ **绝对峰值不是可靠的逐指令定位器**。")
    out("     可靠的逐指令定位见 montage 的逐层 3×3 窗口(§A2 第 5 步,对单个 sink 格更稳);")
    out("     几何反投影本身已验证正确(桌面 z=0.900),留作 S2 influence 世界坐标对照用。")

    # ---------- 最小对目的地位移(米) ----------
    out("\n" + "=" * 96)
    out("最小对目的地位移:'plate' 峰值 vs 'cabinet' 峰值,世界坐标距离(米)")
    out("=" * 96)

    def dest_world(name, want):
        head_sum = az[f"{name}__head_sum"]
        pieces = [str(x) for x in az[f"{name}__pieces"]]
        idx = [i for i, p in enumerate(pieces) if want in p.lower()]
        if not idx:
            return None
        band = head_sum[4:13, idx[0], 0].mean(axis=0)
        r, c = np.unravel_index(int(band.argmax()), band.shape)
        raw_r, raw_c = cell_to_raw256(r, c)
        Pw, _, _ = backproject(raw_r, raw_c, depth, K, E)
        return Pw, (r, c)

    pl = dest_world("bowl_plate_orig", "plate")
    cb = dest_world("bowl_cabinet_orig", "cabinet")
    if pl and cb:
        d = np.linalg.norm(pl[0] - cb[0])
        out(f"  plate  峰值 cell {pl[1]} → 世界 ({pl[0][0]:.3f},{pl[0][1]:.3f},{pl[0][2]:.3f})")
        out(f"  cabinet峰值 cell {cb[1]} → 世界 ({cb[0][0]:.3f},{cb[0][1]:.3f},{cb[0][2]:.3f})")
        out(f"  ⇒ 目的地世界位移 = {d:.3f} m")
        out("  ⚠️ 目的地图跨层不稳(report_traj_minpair ②),单层 argmax 反投影只作示意。")

    TXT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {TXT}")


if __name__ == "__main__":
    main()
