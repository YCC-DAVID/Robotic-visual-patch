#!/usr/bin/env python3
"""PART B 的指令表 —— **单一来源**,被 probe_attention_b1b2 / make_attn_maps /
make_attn_montage 共同 import。分散定义会漂移(已经踩过一次)。

设计
----
四条 LIBERO 原生指令(B1,换任务)各自带一组改写(B2,换措辞),结构对称:
    orig + L1(只换动词) + L2(换句法) + L3(加框架词)

**两条硬约束**(来自计划,违反则归因失效):
  1. **绝不换名词**。bowl→dish、stove→cooktop 会引入词表覆盖问题,
     与句式改写混在一起无法归因。只动动词与句法。
  2. 每条改写句**必须先跑几条 rollout 验成功率**。LIBERO 指令模板化,
     模型可能对脱离模板的说法很脆。成功率塌了的句子**不能进对比** ——
     否则比的是"正常 vs 懵了",不是改写鲁棒性。
     ⚠️ 目前**没有任何改写句验过成功率**,这是 B2 最大的未决项。
     头号嫌疑是 L2 倒装(它的 attention 偏移也恰好最大 —— 可能是同一回事)。

`turn on the stove` 的动词是 phrasal verb "turn on",所以:
  · L1 只给两个(switch on / power on)—— 保持"动词+小品词"结构的自然替换就这些,
    硬凑第三个会变成不自然的英语,反而引入新的混淆变量。**不对称是刻意的,记在这里。**
  · L2 用小品词后移 "turn the stove on" —— 这是很自然的句法变体,且不碰名词。
"""

# group -> 该组共用的名词(按【词】写,不写 sentencepiece 的 piece 串)
GROUP_NOUNS = {
    "stove": ["stove"],
    "bottle_rack": ["wine", "bottle", "rack"],
    "bowl_plate": ["bowl", "plate"],
    "bowl_cabinet": ["bowl", "top", "cabinet"],
}

# (name, group, prompt, verb_words, layer_tag)
# layer_tag: orig / L1 / L2 / L3,便于后处理按改写层次分组
VARIANTS = [
    # ---------------- turn on the stove
    ("stove_orig",        "stove", "turn on the stove",              ["turn"],   "orig"),
    ("stove_L1_switch",   "stove", "switch on the stove",            ["switch"], "L1"),
    ("stove_L1_power",    "stove", "power on the stove",             ["power"],  "L1"),
    ("stove_L2_partmove", "stove", "turn the stove on",              ["turn"],   "L2"),
    ("stove_L3_please",   "stove", "please turn on the stove",       ["turn"],   "L3"),

    # ---------------- put the wine bottle on the rack
    ("bottle_rack_orig",      "bottle_rack", "put the wine bottle on the rack",        ["put"],   "orig"),
    ("bottle_rack_L1_place",  "bottle_rack", "place the wine bottle on the rack",      ["place"], "L1"),
    ("bottle_rack_L1_set",    "bottle_rack", "set the wine bottle on the rack",        ["set"],   "L1"),
    ("bottle_rack_L1_move",   "bottle_rack", "move the wine bottle on the rack",       ["move"],  "L1"),
    ("bottle_rack_L2_front",  "bottle_rack", "on the rack, put the wine bottle",       ["put"],   "L2"),
    ("bottle_rack_L3_please", "bottle_rack", "please put the wine bottle on the rack",  ["put"],   "L3"),

    # ---------------- put the bowl on the plate
    ("bowl_plate_orig",      "bowl_plate", "put the bowl on the plate",         ["put"],   "orig"),
    ("bowl_plate_L1_place",  "bowl_plate", "place the bowl on the plate",       ["place"], "L1"),
    ("bowl_plate_L1_set",    "bowl_plate", "set the bowl on the plate",         ["set"],   "L1"),
    ("bowl_plate_L1_move",   "bowl_plate", "move the bowl on the plate",        ["move"],  "L1"),
    ("bowl_plate_L2_front",  "bowl_plate", "on the plate, put the bowl",        ["put"],   "L2"),
    ("bowl_plate_L3_please", "bowl_plate", "please put the bowl on the plate",  ["put"],   "L3"),

    # ---------------- put the bowl on top of the cabinet
    ("bowl_cabinet_orig",      "bowl_cabinet", "put the bowl on top of the cabinet",        ["put"],   "orig"),
    ("bowl_cabinet_L1_place",  "bowl_cabinet", "place the bowl on top of the cabinet",      ["place"], "L1"),
    ("bowl_cabinet_L1_set",    "bowl_cabinet", "set the bowl on top of the cabinet",        ["set"],   "L1"),
    ("bowl_cabinet_L1_move",   "bowl_cabinet", "move the bowl on top of the cabinet",       ["move"],  "L1"),
    ("bowl_cabinet_L2_front",  "bowl_cabinet", "on top of the cabinet, put the bowl",       ["put"],   "L2"),
    ("bowl_cabinet_L3_please", "bowl_cabinet", "please put the bowl on top of the cabinet", ["put"],   "L3"),
]

# 红线 1:什么都不改的重复(prefix 前向无随机性 ⇒ attention 必须逐位相同)
REPEAT = ("REPEAT_bowl_plate", "bowl_plate", "put the bowl on the plate", ["put"], "orig")

# 只有这四条是 LIBERO 原生任务、有对应的 bddl / 成功判据
B1_ORIG = [v[0] for v in VARIANTS if v[4] == "orig"]

GROUPS = {}
for _n, _g, *_ in VARIANTS:
    GROUPS.setdefault(f"group_{_g}", []).append(_n)

META = {v[0]: dict(group=v[1], prompt=v[2], verbs=v[3], tag=v[4]) for v in VARIANTS}
META[REPEAT[0]] = dict(group=REPEAT[1], prompt=REPEAT[2], verbs=REPEAT[3], tag=REPEAT[4])


def match_words(pieces, words):
    """在 sentencepiece 的 pieces 序列里找出这些【词】占的 index。

    为什么不直接写 piece 串:同一个词在句首没有 `▁` 前缀(实测 'put'),
    在句中有(实测 '▁put');而且长词可能被切成多个 subword。
    ⇒ 按"去掉 ▁ 后小写"做贪心的连续拼接匹配,并**匹配不上就报错**,不静默漏掉。
    """
    norm = [p.replace("▁", "").lower() for p in pieces]
    out, unmatched = [], []
    for w in words:
        w = w.lower()
        hit = False
        for i in range(len(norm)):
            acc = ""
            for j in range(i, len(norm)):
                acc += norm[j]
                if acc == w:
                    out.extend(range(i, j + 1))
                    hit = True
                    break
                if not w.startswith(acc):
                    break
            if hit:
                break
        if not hit:
            unmatched.append(w)
    if unmatched:
        raise AssertionError(f"这些词在 tokens 里找不到: {unmatched}   pieces={pieces}")
    return sorted(set(out))


def roles(name, pieces):
    """→ dict(noun=[idx], verb=[idx], func=[idx]),func = 其余(含 <bos> 与 \\n,sink 在这儿)"""
    m = META[name]
    ni = match_words(pieces, GROUP_NOUNS[m["group"]])
    vi = match_words(pieces, m["verbs"])
    fi = [i for i in range(len(pieces)) if i not in ni and i not in vi]
    return dict(noun=ni, verb=vi, func=fi)


def disp_tok(p):
    """token 拿去画图前的清洗。

    ⚠️ PIL 默认位图字体里 `▁`(U+2581,sentencepiece 词首标记)和中文一样是方块 □;
    换行符也不能直接画。
    """
    return p.replace("▁", "").replace("\n", "\\n") or "?"
