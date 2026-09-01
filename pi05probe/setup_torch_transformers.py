#!/usr/bin/env python3
"""S0→S1 过渡步骤 1:准备打好 openpi 补丁的 transformers,**不碰共用的 conda env**。

背景
----
openpi 的 PyTorch 路径要求把 `src/openpi/models_pytorch/transformers_replace/*`
覆盖到已安装的 transformers 里(README.md:203-207),用来:
  1) 支持 AdaRMS  2) 精确控制激活精度  3) 允许 KV cache 只读不更新
`pi0_pytorch.py:118-125` 会主动检查,没打补丁直接抛 ValueError。

官方做法是 `cp -r ... .venv/lib/python3.11/site-packages/transformers/`,
但我们的 py3.11 环境是**共用的 conda env `openpi-server`**(别的项目也在用),
README:212 自己也警告这种覆盖会 "propagate to other projects that use transformers"。

⇒ 本脚本的做法:**把 transformers 整包复制进 WAMattack,在副本上打补丁**,
运行时用 `PYTHONPATH` 前置。transformers 4.53.2 是 62 MB **纯 Python**(零 .so),
所以整包复制完全可行。`openpi-server` 一个字节都不改。

⚠️ 实测发现(2026-08-03):**`openpi-server` 那份 transformers 早在 2026-04 建 env 时
就已经被打过补丁了**(`site-packages/transformers/models/siglip/check.py` 存在),
而且那份补丁与我们 6 月 clone 里的 `transformers_replace` **逐字节一致**(5 个文件全 "已相同")
⇒ 没有版本漂移,直接用共用 env 也能跑。
本脚本仍然做这份 in-tree 副本,理由是:
  1) 自包含 —— 别人日后在那个 env 里 `pip install -U transformers` 不会打断我们;
  2) 可复现 —— 我们的 PYTHONPATH 里写明了用的是哪份代码,不依赖 env 的历史状态。
代价只有 62 MB。

用法:
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/setup_torch_transformers.py
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/setup_torch_transformers.py --force
"""

import argparse
import filecmp
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
REPLACE_SRC = OPENPI / "src" / "openpi" / "models_pytorch" / "transformers_replace"

SERVER_PY = pathlib.Path("/home/user1/miniconda3/envs/openpi-server/bin/python")
SITE = pathlib.Path("/home/user1/miniconda3/envs/openpi-server/lib/python3.11/site-packages")
PRISTINE = SITE / "transformers"

DEST_ROOT = ROOT / "third_party" / "transformers_patched"   # ← 放到 PYTHONPATH 上的目录
DEST = DEST_ROOT / "transformers"                           # ← 包本身

EXPECT_VERSION = "4.53.2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="已存在也重新复制一遍")
    args = ap.parse_args()

    # ---------------------------------------------------------------- 0) 前置检查
    assert PRISTINE.is_dir(), f"找不到原始 transformers: {PRISTINE}"
    assert REPLACE_SRC.is_dir(), f"找不到 transformers_replace: {REPLACE_SRC}"
    so = list(PRISTINE.rglob("*.so")) + list(PRISTINE.rglob("*.pyd"))
    assert not so, f"transformers 里有编译扩展,整包复制这招不成立: {so[:3]}"
    print(f"[0] 原始 transformers: {PRISTINE}  (纯 Python,无 .so)")

    # ---------------------------------------------------------------- 1) 整包复制
    if DEST.exists() and not args.force:
        print(f"[1] {DEST} 已存在,跳过复制(要重来请加 --force)")
    else:
        if DEST.exists():
            print(f"[1] --force:先删掉 {DEST}")
            shutil.rmtree(DEST)
        DEST_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"[1] 复制 {PRISTINE} -> {DEST} …")
        shutil.copytree(PRISTINE, DEST)
        n = sum(1 for _ in DEST.rglob("*.py"))
        print(f"[1] 完成,{n} 个 .py 文件")

    # ---------------------------------------------------------------- 2) 打补丁
    # 逐个文件覆盖,并记录改了哪些、原文件是否真的不同(免得以为打了其实是同一份)
    print(f"[2] 覆盖 transformers_replace/* -> {DEST}")
    patched = []
    for src in sorted(REPLACE_SRC.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(REPLACE_SRC)
        dst = DEST / rel
        existed = dst.exists()
        same = existed and filecmp.cmp(src, dst, shallow=False)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        tag = "新增" if not existed else ("已相同" if same else "覆盖")
        patched.append((str(rel), tag))
        print(f"      [{tag}] {rel}")
    assert patched, "transformers_replace 里一个文件都没有?"

    # ---------------------------------------------------------------- 3) 验证
    # 关键:PYTHONPATH 前置后,transformers 必须解析到我们的副本,而且 check 能过。
    print("[3] 验证(用子进程,PYTHONPATH 前置)…")
    code = f"""
import transformers, pathlib
f = transformers.__file__
print("transformers.__file__ =", f)
print("transformers.__version__ =", transformers.__version__)
assert "{DEST_ROOT}" in f, "PYTHONPATH 没生效,还在用 conda env 里那份!"
assert transformers.__version__ == "{EXPECT_VERSION}", "版本不对"
from transformers.models.siglip import check
ok = check.check_whether_transformers_replace_is_installed_correctly()
print("check_whether_transformers_replace_is_installed_correctly() =", ok)
assert ok
# 共用 env 里那份的状态(只报告,不断言 —— 见下方说明)
p = pathlib.Path("{SITE}") / "transformers" / "models" / "siglip" / "check.py"
print("共用 env 是否早就打过补丁 (check.py 存在?):", p.exists())
print("PATCHED TRANSFORMERS OK")
"""
    env_pp = f"{DEST_ROOT}:{OPENPI/'src'}:{OPENPI/'packages'/'openpi-client'/'src'}"
    r = subprocess.run(
        [str(SERVER_PY), "-c", code],
        env={"PYTHONPATH": env_pp, "PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin",
             "HOME": "/home/user1"},
        capture_output=True, text=True,
    )
    print(r.stdout, end="")
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        print("[3] ❌ 验证失败")
        return 1

    print()
    print("=" * 90)
    print("✅ 完成。之后所有走 PyTorch 路径的调用,PYTHONPATH 必须【以这个为首】:")
    print(f"   {env_pp}")
    print("   顺序很重要:transformers_patched 要排在最前面,才能盖掉 site-packages 里那份。")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    sys.exit(main())
