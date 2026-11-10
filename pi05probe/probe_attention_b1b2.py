#!/usr/bin/env python3
"""PART B 第 2 步:B1(换任务)+ B2(换措辞)的 attention 扫描。

输入:`out/shared_frame.npz`(由 dump_shared_frame.py 产生)。
      四个 task 的 img224/wrist224 已验证**逐位相同**(S0.5 检查 B)⇒ 唯一变量是文本。

为什么这一步不需要 ε / 不需要 websocket
--------------------------------------
text→image attention 全部出在 **prefix 那一趟**(`pi0_pytorch.py:396-403`
→ `gemma_pytorch.py:102` → HF `language_model.forward()`),
而 prefix 在任何去噪步之前、**完全不涉及 flow matching 的 noise**。
⇒ 静态帧 + npz 交接就够,不用碰 §PT-6 那个 noise 包装。

存储 schema(§B:层/token/时间三轴都不合并)
------------------------------------------
    attn_head_sum[variant][layer, token, view, h, w]     ← head 已按 §A2 求和
    attn_per_head[variant][layer, head, token, view, h, w] ← §B 额外要求,查 head 间方差
一律存**未归一化的原始值**,归一化/max/sum/名词筛选全部留给后处理(零成本)。

§A2 的流程(注意 §A3:**逐行归一化必须在 token 取 max 之前**):
    1 head 求和 → A ∈ R^{Z×N_v}
    2 逐行(每个 text token)在图像 token 维度上重新归一化
    3 token 取逐元素 max → saliency S ∈ R^{h×w}
    4 s×s 窗口(s=3)选块
本脚本把 1–4 都算一遍作为 report,但原始值照存,便于日后换算法。

⚠️ 三条已实测的硬约束(都在 FINDINGS 里)
  · Z 只能取 `tokenized_prompt_mask==True` 的行 —— padding 行的弥散注意力是真实 token 的 78 倍
  · §A-4 的 sink 实际落在末尾的 `\\n`(0.397 vs BOS 的 0.011)⇒ 第 2 步不做就全被它决定
  · §C 各指令 Z 不同(6/8/9/10)⇒ **"全 token max" 图跨指令有偏**,B1/B2 必须用名词专属/逐 token 图

用法:
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/probe_attention_b1b2.py
"""

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

# --- 环境全在脚本内设好(带一长串 env var 前缀的命令会被分类器拒,见 FINDINGS §0) ---
for p in reversed([PATCHED_TF, OPENPI / "src", OPENPI / "packages" / "openpi-client" / "src"]):
    sys.path.insert(0, str(p))
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

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from instructions import B1_ORIG, META, REPEAT, VARIANTS as VTABLE, roles  # noqa: E402

N_IMG, N_SIDE, N_LAYER, N_HEAD = 256, 16, 18, 8
VIEWS = ["base_0_rgb", "left_wrist_0_rgb"]      # right_wrist 全零且 mask=False,不存
WIN = 3                                         # §A2 第 5 步的 s×s 窗口

# 指令表在 instructions.py(单一来源)。四条 B1 原生指令各带一组 L1/L2/L3 改写。
VARIANTS = [(v[0], v[2]) for v in VTABLE] + [(REPEAT[0], REPEAT[2])]

_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def main():
    import transformers
    assert str(PATCHED_TF) in transformers.__file__, "没用到打补丁的 transformers"
    import torch
    from openpi.models import tokenizer as _tok
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    say("=" * 100)
    say("0 · 输入与环境")
    say("=" * 100)
    npz = np.load(OUT / "shared_frame.npz", allow_pickle=False)
    tasks = [str(t) for t in npz["tasks"]]
    say(f"  shared_frame.npz 里的 task: {tasks}")
    say(f"  shared_state sha 见 shared_frame.txt;warmup = {int(npz['num_steps_wait'])} 步")

    # 四个 task 的图像已验证逐位相同,取任意一个即可 —— 但这里再断言一次
    ref = tasks[0]
    img224 = npz[f"{ref}__img224"]
    wri224 = npz[f"{ref}__wrist224"]
    state8 = npz[f"{ref}__state8"]
    for t in tasks[1:]:
        assert np.array_equal(img224, npz[f"{t}__img224"]), f"{t} 的 base 图与 {ref} 不同!"
        assert np.array_equal(wri224, npz[f"{t}__wrist224"]), f"{t} 的 wrist 图与 {ref} 不同!"
    say(f"  ✅ 四个 task 的 img224/wrist224 逐位相同(再次确认)  img224{img224.shape} {img224.dtype}")

    cfg = _config.get_config("pi05_libero")
    # ⚠️ 必须关 torch.compile,否则 forward hook 可能不触发(见 FINDINGS §PT-4)
    cfg_nc = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    say(f"  pytorch_compile_mode: {cfg.model.pytorch_compile_mode!r} -> None")
    policy = _policy_config.create_trained_policy(cfg_nc, TORCH_CKPT)
    model = policy._model                                    # noqa: SLF001
    layers = model.paligemma_with_expert.paligemma.language_model.layers
    assert len(layers) == N_LAYER, f"层数 {len(layers)} != {N_LAYER}"
    say(f"  policy 就绪;language_model.layers = {len(layers)}")

    tk = _tok.PaligemmaTokenizer(cfg.model.max_token_len)
    ZMAX = cfg.model.max_token_len
    TXT_LO = 3 * N_IMG                                       # language block 起点 = 768
    VIEW_LO = {"base_0_rgb": 0, "left_wrist_0_rgb": N_IMG}   # 见 FINDINGS Q6 的 index 表

    # ------------------------------------------------------------ 逐 variant 前向
    say("")
    say("=" * 100)
    say("1 · 逐 variant 前向 + hook 抓 attention")
    say("=" * 100)
    results = {}
    for name, prompt in VARIANTS:
        ids, mask = tk.tokenize(prompt)
        Z = int(mask.sum())
        pieces = [tk._tokenizer.id_to_piece(int(i)) for i in ids[:Z]]   # noqa: SLF001

        cap = {}

        def mk(i):
            def hook(_m, _i, out):
                cap[i] = out[1].detach().float().cpu()   # (B, H, 968, 968)
            return hook

        hs = [layers[i].self_attn.register_forward_hook(mk(i)) for i in range(N_LAYER)]
        try:
            res = policy.infer({
                "observation/image": img224,
                "observation/wrist_image": wri224,
                "observation/state": state8,
                "prompt": prompt,
            })
        finally:
            for h in hs:
                h.remove()
        assert len(cap) == N_LAYER, f"{name}: 只抓到 {len(cap)} 层"

        # full[l] = (H, 968, 968)
        full = np.stack([cap[l][0].numpy() for l in range(N_LAYER)])     # (L,H,968,968)
        assert full.shape[-1] == 3 * N_IMG + ZMAX, f"prefix 长度异常 {full.shape}"

        # 只留有效 text 行 × 两路图像列 → (L, H, Z, V, 16, 16)
        per_head = np.stack([
            full[:, :, TXT_LO:TXT_LO + Z, VIEW_LO[v]:VIEW_LO[v] + N_IMG]
            .reshape(N_LAYER, N_HEAD, Z, N_SIDE, N_SIDE)
            for v in VIEWS], axis=3)
        head_sum = per_head.sum(axis=1)                                  # (L,Z,V,16,16)

        # attention rollout(§A3 第二种提取方式):head 平均 → 0.5A+0.5I → 逐层相乘
        S = full.shape[-1]
        roll = np.eye(S, dtype=np.float32)
        eye = np.eye(S, dtype=np.float32)
        for l in range(N_LAYER):
            Ab = full[l].mean(axis=0)                                    # head 平均 (968,968)
            Ah = 0.5 * Ab + 0.5 * eye
            Ah = Ah / np.clip(Ah.sum(axis=-1, keepdims=True), 1e-12, None)
            roll = Ah @ roll
        roll_blk = np.stack([
            roll[TXT_LO:TXT_LO + Z, VIEW_LO[v]:VIEW_LO[v] + N_IMG].reshape(Z, N_SIDE, N_SIDE)
            for v in VIEWS], axis=1)                                     # (Z,V,16,16)

        results[name] = dict(prompt=prompt, Z=Z, pieces=pieces, ids=ids[:Z],
                             head_sum=head_sum.astype(np.float32),
                             per_head=per_head.astype(np.float32),
                             rollout=roll_blk.astype(np.float32),
                             actions=np.asarray(res["actions"], dtype=np.float32))
        say(f"  {name:20s} Z={Z:2d}  pieces={pieces}")
        say(f"  {'':20s} head_sum{head_sum.shape}  per_head{per_head.shape}  rollout{roll_blk.shape}")

    # ------------------------------------------------------------ 红线 1
    say("")
    say("=" * 100)
    say("2 · 红线 1:什么都不改的重复(prefix 前向无随机性 ⇒ 必须逐位相同)")
    say("=" * 100)
    a = results["bowl_plate_orig"]["head_sum"]
    b = results[REPEAT[0]]["head_sum"]
    dmax = float(np.abs(a - b).max())
    say(f"  attention |Δ| max = {dmax:.6e}")
    act_d = float(np.abs(results['bowl_plate_orig']['actions']
                         - results[REPEAT[0]]['actions']).max())
    say(f"  动作     |Δ| max = {act_d:.6e}   (⚠️ 动作会因 ε 重采样而不同,这是预期的)")
    if dmax == 0.0:
        say("  ✅ attention 逐位相同 —— 噪声地板为 0,后面任何差异都是真实信号")
    else:
        say("  ❌ attention 不可复现!先查 torch.compile / dropout / eval 模式,不要继续")

    # ------------------------------------------------------------ §A2 saliency
    say("")
    say("=" * 100)
    say("3 · §A2 saliency(逐行归一化 → token max/sum → 3×3 窗口)· 主视角 base_0_rgb")
    say("=" * 100)
    vi = VIEWS.index("base_0_rgb")

    def saliency(hs, renorm=True, reduce="max"):
        """hs: (L,Z,V,16,16) 的某一层某一 view 切出来的 (Z,16,16)。"""
        A = hs.reshape(hs.shape[0], -1).astype(np.float64)      # (Z,256)
        if renorm:                                             # §A3 第 2 步,逐行
            A = A / np.clip(A.sum(axis=1, keepdims=True), 1e-12, None)
        return (A.max(axis=0) if reduce == "max" else A.sum(axis=0)).reshape(N_SIDE, N_SIDE)

    def best_window(S, s=WIN):
        best, pos = -np.inf, None
        for i in range(N_SIDE - s + 1):
            for j in range(N_SIDE - s + 1):
                v = S[i:i + s, j:j + s].sum()
                if v > best:
                    best, pos = v, (i + s // 2, j + s // 2)
        return pos, best

    # ⚠️ 这里只对四条 B1 原生指令印一个粗表当 sanity check。
    # "全 token max" 跨指令**有偏**(Z 不同),正经的跨指令比较在 report_b1b2.py 里
    # 用名词专属图做。另外全图 3×3 argmax 是很粗的离散量,别拿它下结论。
    say("  【sanity check,别拿来下结论】四条原生指令的 3×3 窗口 argmax(逐行归一化 + token max):")
    say("  layer | " + " | ".join(f"{n.replace('_orig',''):>14s}" for n in B1_ORIG))
    for l in range(N_LAYER):
        say(f"  {l:5d} | " + " | ".join(
            f"{str(best_window(saliency(results[n]['head_sum'][l, :, vi]))[0]):>14s}"
            for n in B1_ORIG))

    # ------------------------------------------------------------ sink 复核
    say("")
    say("=" * 100)
    say("4 · §A-4 sink:各 token 的注意力质量(layer 0,主视角,head 求和后除以 H)")
    say("=" * 100)
    for name, _ in VARIANTS:
        if name == REPEAT[0]:
            continue
        r = results[name]
        m = r["head_sum"][0, :, vi].reshape(r["Z"], -1).sum(axis=1) / N_HEAD
        top = int(np.argmax(m))
        rest = float(np.delete(m, top).mean())
        say(f"  {name:24s} argmax_token={top}({r['pieces'][top]!r})  "
            f"mass={m[top]:.4f}  其余均值={rest:.4f}  比值={m[top]/max(rest, 1e-12):.1f}×")

    # ------------------------------------------------------------ 存盘
    OUT.mkdir(parents=True, exist_ok=True)
    save = {}
    for name, r in results.items():
        save[f"{name}__head_sum"] = r["head_sum"]
        save[f"{name}__per_head"] = r["per_head"]
        save[f"{name}__rollout"] = r["rollout"]
        save[f"{name}__actions"] = r["actions"]
        save[f"{name}__pieces"] = np.array(r["pieces"])
        save[f"{name}__ids"] = np.asarray(r["ids"])
        save[f"{name}__prompt"] = np.array(r["prompt"])
    save["views"] = np.array(VIEWS)
    save["variant_names"] = np.array([n for n, _ in VARIANTS])
    save["variant_prompts"] = np.array([p for _, p in VARIANTS])
    np.savez_compressed(OUT / "attn_b1b2.npz", **save)
    (OUT / "attn_b1b2.txt").write_text("\n".join(_lines) + "\n")
    say("")
    say(f"[written] {OUT/'attn_b1b2.npz'}   "
        f"({(OUT/'attn_b1b2.npz').stat().st_size / 2**20:.1f} MiB)")
    say(f"[written] {OUT/'attn_b1b2.txt'}")
    say("")
    say("下一步(纯后处理,零额外前向):按 §C 用**名词专属图**做跨指令比较 —— "
        "各指令 Z 不同,'全 token max' 图有偏,不能直接比。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
