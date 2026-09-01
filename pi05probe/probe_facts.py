#!/usr/bin/env python3
"""S0 事实核查的可跑部分:Q1 resize / Q2 动作语义 / Q3 归一化 / Q6 token 布局。

不加载模型、不占 GPU,秒级跑完。
用法:
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/probe_facts.py
输出同时打到 stdout 和 pi05probe/out/s0_facts.txt。
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
OUT = ROOT / "pi05probe" / "out"
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI / "packages" / "openpi-client" / "src"))

CKPT = pathlib.Path("/home/user1/.cache/openpi/openpi-assets/checkpoints/pi05_libero")

_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _lines.append(s)


def arr(a, prec=5):
    return np.array2string(np.asarray(a, dtype=np.float64), precision=prec,
                           suppress_small=True, max_line_width=250)


# ---------------------------------------------------------------- Q1 · resize
say("=" * 100)
say("Q1 · 256 -> 224 的 resize:走 openpi_client.image_tools.resize_with_pad")
say("=" * 100)
from openpi_client import image_tools  # noqa: E402

rng = np.random.default_rng(0)
img256 = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
img224 = image_tools.resize_with_pad(img256, 224, 224)
say(f"  输入 {img256.shape} -> 输出 {img224.shape}  dtype={img224.dtype}")
# _resize_with_pad_pil 的算术:ratio = max(256/224, 256/224)
ratio = max(256 / 224, 256 / 224)
rh, rw = int(256 / ratio), int(256 / ratio)
pad_h, pad_w = max(0, int((224 - rh) / 2)), max(0, int((224 - rw) / 2))
say(f"  ratio={ratio:.10f}  resized=({rh},{rw})  pad=(h={pad_h}, w={pad_w})")
say(f"  ⇒ 【零 padding】,纯 PIL BILINEAR 各向同性缩放,尺度因子恰好 224/256 = {224/256}")
# 四条边是否真的没有黑边(pad 会留 0)
say(f"  四边最小值(pad 会是 0): top={img224[0].min()} bottom={img224[-1].min()} "
    f"left={img224[:,0].min()} right={img224[:,-1].min()}")
# 反投影用的像素中心对应关系
say("  像素中心映射(PIL/tf resize 约定,反投影必须用这个,不是 u*224/256):")
say("      u_256 = (u_224 + 0.5) * 256/224 - 0.5 = (u_224 + 0.5) / 0.875 - 0.5")
say("  ⇒ 内参随分辨率线性缩放:f_224 = f_256 * 224/256;agentview f_256=309.0193 ⇒ f_224 = "
    f"{309.0193 * 224 / 256:.4f}; c_224 = (111.5, 111.5)")

# 服务端还有一次 ResizeImages(224,224) —— 对 224 输入是 no-op
say("  服务端 model_transforms 里还有一次 _transforms.ResizeImages(224,224)"
    " (config.py:131),对已经 224 的输入是 no-op(image_tools.py:28-29 提前 return)")

# ---------------------------------------------------------------- Q3 · 归一化
say("")
say("=" * 100)
say("Q3 · 归一化:π0.5 走【分位数】(q01/q99),不是 mean/std")
say("=" * 100)
say("  config.py:187  use_quantile_norm = (model_config.model_type != ModelType.PI0)")
say("  pi05_libero 的 model_type 是 PI05 (pi0_config.py:51-56) ⇒ use_quantile_norm=True")
say("  transforms.py:141-145  _normalize_quantile:  (x - q01) / (q99 - q01 + 1e-6) * 2 - 1")
say("  transforms.py:175-183  _unnormalize_quantile: 反过来")

ns_path = CKPT / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
say(f"  stats 文件: {ns_path}")
say("  解析链: policy_config.py:59-64 load_norm_stats(checkpoint_dir/'assets', asset_id)")
say("          asset_id = repo_id = 'physical-intelligence/libero' (config.py:181-183)")
say("  ⚠️ 注意:norm stats 取自 **checkpoint 的 assets**,不是 config 的 assets_dirs"
    "(policy_config.py:60-61 明确说是为了和训练时一致)")

d = json.loads(ns_path.read_text())
say(f"  json 顶层 keys: {list(d.keys())}")
stats = d["norm_stats"]
for key in stats:
    say(f"  --- '{key}' 里有的字段: {list(stats[key].keys())}")

for key in ("state", "actions"):
    if key not in stats:
        continue
    s = stats[key]
    q01 = np.asarray(s["q01"], dtype=np.float64)
    q99 = np.asarray(s["q99"], dtype=np.float64)
    mean = np.asarray(s["mean"], dtype=np.float64)
    std = np.asarray(s["std"], dtype=np.float64)
    say("")
    say(f"  ######## {key}  (dim={q01.shape[0]}) ########")
    say(f"   q01  = {arr(q01[:8])}")
    say(f"   q99  = {arr(q99[:8])}")
    say(f"   mean = {arr(mean[:8])}")
    say(f"   std  = {arr(std[:8])}")
    say(f"   (只印前 8 维;数组本身长 {q01.shape[0]},其余是 padding)")

# S3 §D 要的 d_max:用 q01/q99 当 a_min/a_max
if "actions" in stats:
    s = stats["actions"]
    q01 = np.asarray(s["q01"], dtype=np.float64)[:7]
    q99 = np.asarray(s["q99"], dtype=np.float64)[:7]
    std = np.asarray(s["std"], dtype=np.float64)[:7]
    say("")
    say("  ⇒ 给 S3 §D 用的量(前 7 维 = LIBERO 真实动作维):")
    say(f"     q99 - q01 = {arr(q99 - q01)}")
    say(f"     std_clean(a^i) 的先验(训练集 std) = {arr(std)}")
    say("     norm_bound 的 d_max^i = max(|a_clean^i - q01^i|, |a_clean^i - q99^i|),逐帧算")

# ---------------------------------------------------------------- Q2 · 动作语义
say("")
say("=" * 100)
say("Q2 · 动作 7 维的含义与单位")
say("=" * 100)
say("  维度:libero_policy.py:100  data['actions'][..., :7]  ⇒ 对外 7 维")
say("        模型内部 action_dim=32(pi0_config.py:24),LIBERO 只取前 7,其余是 padding")
say("  是否 delta:config.py:332-338 —— 'LIBERO already represents actions as deltas',")
say("        且 pi05_libero 用 extra_delta_transform=False ⇒ 【模型直接输出 LIBERO 原生动作空间】,")
say("        没有额外的 delta/absolute 转换。")
say("  main.py:153  env.step(action.tolist()) —— 【原样送入,没有任何缩放/符号翻转】")
say("")
say("  控制器:LIBERO env_wrapper.py:17 controller='OSC_POSE',用 robosuite 默认 osc_pose.json:")
say("     input_min/max  = -1 / 1")
say("     output_max     = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]")
say("     output_min     = [-0.05,-0.05,-0.05,-0.5,-0.5,-0.5]")
say("     control_delta  = true    uncouple_pos_ori = true    impedance_mode = fixed  kp=150")
say("")
say("  ⇒ a[0:3] 平移 delta:∈[-1,1] 线性映射到 **±0.05 m**(base_controller.py:104-123 scale_action)")
say("  ⇒ a[3:6] 旋转 delta:∈[-1,1] 线性映射到 **±0.5 rad**,是 **axis-angle**(不是欧拉角)")
say("        证据:control_utils.py:150-176 set_goal_orientation 的 docstring 明确写")
say("        'Desired relative change in orientation, in axis-angle form [ax, ay, az]',")
say("        并 axisangle2quat(delta) 转成旋转矩阵左乘当前姿态。")
say("        ⇒ S3 的旋转通道用 SO(3) 测地距离是对的,不能直接减。")
say("  ⚠️ scale_action 先 np.clip(action, -1, 1):**模型输出超过 ±1 的部分被截断**,")
say("        Δa 里超出 ±1 的差异对环境无效果 ⇒ S3 报 influence 时要记这一点。")
say("")
say("  ⇒ a[6] 夹爪:panda_gripper.py:43-58 format_action")
say("        current_action = clip(current_action + [-1,1]*speed*np.sign(action), -1, 1),speed=0.01")
say("        注释:'-1 => open, 1 => closed'")
say("  ⚠️⚠️ **只用 np.sign(action),幅值被完全丢弃**,而且夹爪命令是【积分状态】。")
say("        ⇒ 夹爪通道唯一有物理意义的量就是 **sign(a[6]) 是否翻转**;")
say("          不翻符号的 Δa[6] 对环境**完全没有影响**。这正好印证计划里'夹爪看是否翻转'。")
say("        ⇒ 判据写死:flip ⟺ sign(a_patched[6]) != sign(a_clean[6]);注意 sign(0)=0 的边界。")
say("")
say("  main.py:17  LIBERO_DUMMY_ACTION = [0.0]*6 + [-1.0]  ⇒ warmup 的 10 步是'不动 + 张开'")

# ---------------------------------------------------------------- Q6 · token 布局
say("")
say("=" * 100)
say("Q6 · token 布局 / SigLIP grid / Gemma 层头数")
say("=" * 100)
import openpi.models.gemma as _gemma  # noqa: E402
from openpi.training import config as _config  # noqa: E402

cfg = _config.get_config("pi05_libero")
mc = cfg.model
pg = _gemma.get_config(mc.paligemma_variant)
ae = _gemma.get_config(mc.action_expert_variant)
say(f"  pi05={mc.pi05}  model_type={mc.model_type}  action_dim={mc.action_dim}  "
    f"action_horizon={mc.action_horizon}")
say(f"  max_token_len={mc.max_token_len}   discrete_state_input={mc.discrete_state_input}")
say(f"  paligemma_variant={mc.paligemma_variant!r}: width={pg.width} depth={pg.depth} "
    f"num_heads={pg.num_heads} num_kv_heads={pg.num_kv_heads} head_dim={pg.head_dim}")
say(f"  action_expert_variant={mc.action_expert_variant!r}: width={ae.width} depth={ae.depth} "
    f"num_heads={ae.num_heads} num_kv_heads={ae.num_kv_heads} head_dim={ae.head_dim}")
say("  vision tower: pi0.py:81-89  siglip.Module(variant='So400m/14', pool_type='none', num_classes=width)")
n_patch_side = 224 // 14
n_img_tok = n_patch_side * n_patch_side
say(f"  ⇒ SigLIP So400m/**14** @ 224 ⇒ grid {n_patch_side}×{n_patch_side} = {n_img_tok} patch token/图")
say("     pool_type='none' ⇒ 不池化,**没有 CLS token**,全是 patch token(正好,B0 要的就是 patch)")
say("")
say("  prefix token 顺序(pi0.py:112-133,按 obs.images 的插入序,libero_policy.py:53-57):")
o = 0
for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
    say(f"     [{o:4d} .. {o+n_img_tok-1:4d}]  {name}   ({n_patch_side}×{n_patch_side})")
    o += n_img_tok
say(f"     [{o:4d} .. {o+mc.max_token_len-1:4d}]  language (max_token_len={mc.max_token_len})")
say(f"  ⇒ prefix 长度 = 3×{n_img_tok} + {mc.max_token_len} = {3*n_img_tok + mc.max_token_len}")
say(f"  suffix:pi05 **没有 state token**(pi0.py:151-157 的 state_proj 在 `if not self.pi05` 里)")
say(f"         ⇒ suffix = {mc.action_horizon} 个 action token,总长 "
    f"{3*n_img_tok + mc.max_token_len + mc.action_horizon}")
say("")
say("  ⇒ 【B0 要的 attn[text_idx, image_idx] 区间】")
say(f"     text 行  : {3*n_img_tok} .. {3*n_img_tok + mc.max_token_len - 1}")
say(f"     base 图列: 0 .. {n_img_tok-1},reshape 成 ({n_patch_side}, {n_patch_side})")
say("")
say("  ⚠️⚠️ 【text token 有效行只有前几个,其余是 padding】")
say("     tokenizer.py:33  discrete_state_input=False ⇒ tokens = encode(text, add_bos=True) + encode('\\n')")
say("     tokenizer.py:35-38  不足 max_token_len 的用 **0 填充**,mask=False")
say(f"     ⇒ 200 个 slot 里绝大多数是 padding,attention 里被 mask 掉。")
say("     ⇒ 算 A ∈ R^{Z×N_v} 时 **Z 必须只取 tokenized_prompt_mask==True 的行**,")
say("       否则 191 行垃圾进 max/sum。这一条直接决定 §A 的 saliency 正确性。")

# 实际 token 数
try:
    from openpi.models import tokenizer as _tok
    tk = _tok.PaligemmaTokenizer(mc.max_token_len)
    say("")
    say("  四条候选指令的实际 token 数(PaligemmaTokenizer,state=None):")
    for p in ("turn on the stove",
              "put the wine bottle on the rack",
              "put the bowl on the plate",
              "put the bowl on top of the cabinet",
              "open the middle drawer of the cabinet"):
        tokens, mask = tk.tokenize(p)
        n = int(mask.sum())
        ids = tokens[:n].tolist()
        pieces = [tk._tokenizer.id_to_piece(i) for i in ids]  # noqa: SLF001
        say(f"     Z={n:2d}  {p!r}")
        say(f"            ids    = {ids}")
        say(f"            pieces = {pieces}")
    say("  ⚠️ 各指令 Z 不同 ⇒ 计划 §C 说的'全 token max 跨指令有偏'确有其事,数值在这里。")
except Exception as e:  # noqa: BLE001
    say(f"  [tokenizer 探测失败,可能需要联网下 paligemma_tokenizer.model]: {e!r}")

say("")
say("=" * 100)
say("⚠️ 顺带发现(重要,会影响 S2 的设计):π0.5-LIBERO **完全不消费 observation/state**")
say("=" * 100)
say("  1. pi05 的 embed_suffix 里 state token 在 `if not self.pi05:` 分支(pi0.py:151-157)⇒ 不加")
say("  2. pi05_libero 显式设了 discrete_state_input=False(config.py:745)")
say("     ⇒ TokenizePrompt 收到 state=None(transforms.py:256-261)⇒ state 不进 prompt")
say("  3. 于是 state 被 Normalize + PadStatesAndActions 处理过,然后**没人读**")
say("  ⇒ 动作只依赖 (3 路图像, prompt, flow matching 的 ε)。这让 S2 的反事实查询更干净:")
say("    换 patch 不会通过 state 这条路径间接影响输出。")
say("  ⇒ 【待实测确认】喂两个不同 state、其余逐位相同 ⇒ 输出动作应逐位相同。")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "s0_facts.txt").write_text("\n".join(_lines) + "\n")
say("")
say(f"[written] {OUT / 's0_facts.txt'}")
