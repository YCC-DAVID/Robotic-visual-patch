#!/usr/bin/env python3
"""PART B 时间轴:对 rollout_dump.py 采的全路径帧算 attention,并给出 **B3 噪声地板**。

输入 out/traj_<task>.npz(每个 replan 边界一帧)。
输出:
    out/attn_traj_<task>.npz     attn[variant][t, layer, token, view, 16, 16](head 已求和)
    out/attn_traj_<task>.txt     B3 地板 + 跨帧稳定性

为什么 B3 地板是**最重要**的那个数
--------------------------------
单帧上我们量到"换措辞的名词图 Spearman 0.85–1.00"。但 0.95 到底算高还是算低?
没有参照就没法说。B3 给的参照是:**同一条指令、相邻两帧**之间 attention 差多少。
指令带来的差异必须**超过**这个地板才算数。

⚠️ 注意本脚本的帧全部来自**一条 rollout**(执行的是该 task 自己的指令),
   所以对其他指令来说这些帧是"离策略"的 —— 这跟 S2 的反事实查询设计一致
   (沿 clean 轨迹查询,不执行被扰动的动作),不是 bug。

只存 head 求和版(per_head 全轨迹会到 GB 级);逐 head 只对少数几帧另存。

用法(py3.11,不需要 env / 不需要 server):
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/probe_attention_traj.py
    ... --variants bowl_plate_group    # 默认:bowl_plate 组 6 个 + 其余 3 条原生
    ... --variants all                 # 全部 23 个(慢)
"""

import argparse
import dataclasses
import os
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
OUT = ROOT / "pi05probe" / "out"
PATCHED_TF = ROOT / "third_party" / "transformers_patched"
TORCH_CKPT = ROOT / "checkpoints" / "pi05_libero_pytorch"

for p in reversed([PATCHED_TF, OPENPI / "src", OPENPI / "packages" / "openpi-client" / "src"]):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
os.environ["OPENPI_DATA_HOME"] = "/home/user1/.cache/openpi"
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    _o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, check=True).stdout
    _r = sorted(((int(i), int(f)) for i, f in (l.split(",") for l in _o.strip().splitlines())),
                key=lambda r: -r[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_r[0][0])
    print(f"[env] GPU {_r[0][0]}(剩余 {_r[0][1]} MiB)", flush=True)

from instructions import GROUPS, META, roles  # noqa: E402

N_IMG, N_SIDE, N_LAYER, N_HEAD = 256, 16, 18, 8
VIEWS = ["base_0_rgb", "left_wrist_0_rgb"]
_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def rank(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d > 0 else float("nan")


def topk_iou(a, b, k=8):
    ia = set(np.argsort(np.asarray(a).ravel())[::-1][:k].tolist())
    ib = set(np.argsort(np.asarray(b).ravel())[::-1][:k].tolist())
    return len(ia & ib) / len(ia | ib)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="put_the_bowl_on_the_plate")
    ap.add_argument("--variants", default="bowl_plate_group", choices=["bowl_plate_group", "all"])
    ap.add_argument("--renorm", default="img512", choices=["img512", "base", "none"])
    args = ap.parse_args()

    traj = OUT / f"traj_{args.task}.npz"
    assert traj.exists(), f"先跑 rollout_dump.py 拿到 {traj}"
    tz = np.load(traj, allow_pickle=False)
    nF = int(tz["n_frames"])
    ks = tz["ks"].tolist()
    say("=" * 100)
    say(f"输入 {traj.name}:{nF} 帧  success={bool(tz['success'])}  "
        f"env_steps={int(tz['env_steps'])}  replan={int(tz['replan'])}")
    say(f"  执行的指令 = {str(tz['prompt'])!r}")

    if args.variants == "all":
        names = [n for n in META if n != "REPEAT_bowl_plate"]
    else:
        names = list(GROUPS["group_bowl_plate"]) + [
            n for n in META if META[n]["tag"] == "orig"
            and META[n]["group"] != "bowl_plate" and n != "REPEAT_bowl_plate"]
    say(f"  variants({len(names)}): {names}")
    say(f"  总前向次数 = {nF} × {len(names)} = {nF * len(names)}")

    import transformers
    assert str(PATCHED_TF) in transformers.__file__
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config("pi05_libero")
    cfg_nc = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model,
                                                                pytorch_compile_mode=None))
    policy = _policy_config.create_trained_policy(cfg_nc, TORCH_CKPT)
    layers = policy._model.paligemma_with_expert.paligemma.language_model.layers  # noqa: SLF001
    say(f"  policy 就绪(torch.compile 已关,否则 hook 可能不触发);layers={len(layers)}")

    TXT_LO = 3 * N_IMG
    VLO = {"base_0_rgb": 0, "left_wrist_0_rgb": N_IMG}

    # ---------------------------------------------------------------- 逐帧 × 逐 variant
    store = {}           # name -> (T, L, Z, V, 16, 16)
    pieces_of = {}
    for name in names:
        prompt = META[name]["prompt"]
        per_t = []
        for k in ks:
            img = tz[f"f{k:03d}__img224"]
            wri = tz[f"f{k:03d}__wrist224"]
            st = tz[f"f{k:03d}__state8"]
            cap = {}

            def mk(i):
                def hook(_m, _i, out):
                    cap[i] = out[1].detach().float().cpu()
                return hook

            hs = [layers[i].self_attn.register_forward_hook(mk(i)) for i in range(N_LAYER)]
            try:
                policy.infer({"observation/image": img, "observation/wrist_image": wri,
                              "observation/state": st, "prompt": prompt})
            finally:
                for h in hs:
                    h.remove()
            full = np.stack([cap[l][0].numpy() for l in range(N_LAYER)])   # (L,H,968,968)
            Z = full.shape[2] - TXT_LO
            # 有效 token 数:靠 tokenizer 拿,别用 968-768=200
            if name not in pieces_of:
                from openpi.models import tokenizer as _tok
                tk = _tok.PaligemmaTokenizer(cfg.model.max_token_len)
                ids, msk = tk.tokenize(prompt)
                pieces_of[name] = [tk._tokenizer.id_to_piece(int(i))   # noqa: SLF001
                                   for i in ids[:int(msk.sum())]]
            Z = len(pieces_of[name])
            blk = np.stack([full[:, :, TXT_LO:TXT_LO + Z, VLO[v]:VLO[v] + N_IMG]
                            .reshape(N_LAYER, N_HEAD, Z, N_SIDE, N_SIDE)
                            for v in VIEWS], axis=3).sum(axis=1)          # head 求和 → (L,Z,V,h,w)
            per_t.append(blk.astype(np.float32))
        store[name] = np.stack(per_t)                                      # (T,L,Z,V,h,w)
        say(f"  {name:24s} → {store[name].shape}")

    # ---------------------------------------------------------------- B3 噪声地板
    say("")
    say("=" * 100)
    say("B3 · 噪声地板:**同一条指令、相邻两帧**之间的 attention 差异")
    say("=" * 100)
    say("  这就是'多少算大'的参照。指令带来的差异必须超过它才算数。")
    say("")

    def rows(A, layer, vi):
        """A:(T,L,Z,V,h,w) 的单帧 → (Z,256),按 renorm 口径逐行归一化。"""
        pv = [A[layer, :, j].reshape(A.shape[1], -1).astype(np.float64) for j in range(len(VIEWS))]
        if args.renorm == "none":
            return pv[vi]
        den = pv[0].sum(1, keepdims=True) if args.renorm == "base" else \
            sum(p.sum(1, keepdims=True) for p in pv)
        return pv[vi] / np.clip(den, 1e-12, None)

    def noun_map(name, t, layer, vi=0):
        A = store[name][t]
        ni = roles(name, pieces_of[name])["noun"]
        return rows(A, layer, vi)[ni].sum(axis=0)

    ref = "bowl_plate_orig" if "bowl_plate_orig" in store else names[0]
    say(f"  用 {ref} 的名词图,主视角 base:")
    say("  layer | 相邻帧 Spearman  mean / min | 相邻帧 top8 IoU mean | 全轨迹两两 mean")
    floor = {}
    for l in range(N_LAYER):
        adj_s = [spearman(noun_map(ref, i, l), noun_map(ref, i + 1, l)) for i in range(nF - 1)]
        adj_i = [topk_iou(noun_map(ref, i, l), noun_map(ref, i + 1, l)) for i in range(nF - 1)]
        allp = [spearman(noun_map(ref, i, l), noun_map(ref, j, l))
                for i in range(0, nF, max(1, nF // 8)) for j in range(0, nF, max(1, nF // 8)) if i < j]
        floor[l] = dict(adj_mean=float(np.mean(adj_s)), adj_min=float(np.min(adj_s)),
                        iou_mean=float(np.mean(adj_i)), all_mean=float(np.mean(allp)))
        say(f"  {l:5d} | {np.mean(adj_s):10.4f} / {np.min(adj_s):8.4f} | "
            f"{np.mean(adj_i):20.3f} | {np.mean(allp):15.4f}")

    # ---------------------------------------------------------------- 与地板比
    say("")
    say("=" * 100)
    say("判读:换措辞 / 换任务 的差异 vs 相邻帧地板(全轨迹平均)")
    say("=" * 100)
    reph = [n for n in GROUPS["group_bowl_plate"] if n != ref and n in store]
    tasks = [n for n in store if META[n]["tag"] == "orig" and n != ref]
    say(f"  换措辞组: {reph}")
    say(f"  换任务组: {tasks}   (⚠️ 名词集不同,只能比各自【全部名词求和】图)")
    say("")
    say("  layer | 地板(相邻帧) | 换措辞 mean | 换任务 mean | 措辞是否>地板 | 任务是否<地板")

    def allnoun(name, t, layer, vi=0):
        ni = roles(name, pieces_of[name])["noun"]
        return rows(store[name][t], layer, vi)[ni].sum(axis=0)

    for l in range(N_LAYER):
        fl = floor[l]["adj_mean"]
        s_re = float(np.mean([[spearman(allnoun(ref, t, l), allnoun(n, t, l))
                               for t in range(nF)] for n in reph]))
        s_tk = float(np.mean([[spearman(allnoun(ref, t, l), allnoun(n, t, l))
                               for t in range(nF)] for n in tasks]))
        say(f"  {l:5d} | {fl:12.4f} | {s_re:11.4f} | {s_tk:11.4f} | "
            f"{'✅' if s_re > fl else '❌':^13s} | {'✅' if s_tk < fl else '❌':^13s}")
    say("")
    say("  读法:理想是【换措辞的相似度 > 相邻帧地板】(改写比换一帧还稳)")
    say("        且【换任务的相似度 < 相邻帧地板】(换任务比换一帧影响更大)。")
    say("        两列都 ✅ 的层才真正支持'attention 编码任务语义、且对改写鲁棒'。")

    outp = OUT / f"attn_traj_{args.task}.npz"
    np.savez_compressed(outp, **{f"{n}__attn": v for n, v in store.items()},
                        **{f"{n}__pieces": np.array(p) for n, p in pieces_of.items()},
                        ks=np.array(ks), ts=tz["ts"], views=np.array(VIEWS),
                        renorm=np.array(args.renorm), variant_names=np.array(names))
    (OUT / f"attn_traj_{args.task}.txt").write_text("\n".join(_lines) + "\n")
    say("")
    say(f"[written] {outp}  ({outp.stat().st_size / 2**20:.1f} MiB)")
    say(f"[written] {OUT / f'attn_traj_{args.task}.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
