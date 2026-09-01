#!/usr/bin/env python3
"""S0→S1 过渡步骤 2:把 JAX checkpoint 转成 PyTorch(`model.safetensors`)。

官方**没有**发布 PyTorch 格式的权重(`gs://openpi-assets/checkpoints/` 下全是 orbax),
只给了 `examples/convert_jax_model_to_pytorch.py`。

⚠️ 两个坑(都实测踩过):

1. `--checkpoint_dir` 要给 **checkpoint 根目录**(`.../pi05_libero`),不是 `params/` ——
   脚本第 402 行自己拼 `f"{checkpoint_dir}/params/"`。给成 `params` 会变成
   `params/params`,报 `FileNotFoundError: Metadata file (named _METADATA) does not exist`。

2. **⚠️ 上游 bug:根目录一给,`assets/` 就复制不过去。**
   第 536 行是 `assets_source = pathlib.Path(checkpoint_dir).parent / "assets"`,
   而根目录的 `.parent` 是 `.../checkpoints/`,那里没有 `assets`(正确写法应是
   `Path(checkpoint_dir) / "assets"`)。⇒ 转出来的 checkpoint **缺 norm_stats.json**。
   而 π0.5 的动作反归一化全靠它(见 FINDINGS Q3),缺了不会报错,
   `_load_norm_stats` 只 `logging.info("... skipping")` 然后 norm_stats=None
   ⇒ **安静地跑出未反归一化的垃圾动作**。
   ⇒ 本脚本在转换后**自己补一次 copytree**,并 assert 文件到位。

`policy_config.create_trained_policy` 靠输出目录里有没有 `model.safetensors`
自动切到 PyTorch 路径(policy_config.py:49-50),所以之后只要换路径,API 完全不变。

用法:
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/convert_weights.py
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/convert_weights.py --status
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
OUT = ROOT / "pi05probe" / "out"

SERVER_PY = "/home/user1/miniconda3/envs/openpi-server/bin/python"
PATCHED_TF = ROOT / "third_party" / "transformers_patched"

JAX_CKPT = pathlib.Path("/home/user1/.cache/openpi/openpi-assets/checkpoints/pi05_libero")
TORCH_CKPT = ROOT / "checkpoints" / "pi05_libero_pytorch"

LOG = OUT / "convert_weights.log"


def torch_pythonpath() -> str:
    """PyTorch 路径专用:打过补丁的 transformers 必须排最前面。"""
    return ":".join([
        str(PATCHED_TF),
        str(OPENPI / "src"),
        str(OPENPI / "packages" / "openpi-client" / "src"),
    ])


def do_status():
    st = TORCH_CKPT / "model.safetensors"
    print(f"输出目录: {TORCH_CKPT}")
    print(f"  model.safetensors 存在 = {st.exists()}"
          + (f"  ({st.stat().st_size / 2**30:.2f} GiB)" if st.exists() else ""))
    for sub in ("config.json", "assets/physical-intelligence/libero/norm_stats.json"):
        p = TORCH_CKPT / sub
        print(f"  {sub} 存在 = {p.exists()}")
    if LOG.exists():
        print(f"\n日志尾部 ({LOG}):")
        print("".join(LOG.read_text(errors="replace").splitlines(keepends=True)[-15:]))


def child_env():
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env["PYTHONPATH"] = torch_pythonpath()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["OPENPI_DATA_HOME"] = "/home/user1/.cache/openpi"
    # 转换只在 CPU 上组装权重再 save;不给 GPU,免得白占别人的卡
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_PLATFORMS"] = "cpu"
    return env


def run_worker(precision: str) -> int:
    """真正干活的一步:转换 + 补 assets + 自检。跑在脱离终端的子进程里。"""
    cmd = [
        SERVER_PY, "examples/convert_jax_model_to_pytorch.py",
        "--checkpoint_dir", str(JAX_CKPT),        # ← 根目录,脚本内部自己拼 /params/
        "--config_name", "pi05_libero",
        "--output_path", str(TORCH_CKPT),
        "--precision", precision,
    ]
    print("[worker] cmd:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(OPENPI), env=child_env())
    print(f"[worker] convert exit={r.returncode}", flush=True)
    if r.returncode != 0:
        return r.returncode

    st = TORCH_CKPT / "model.safetensors"
    assert st.exists(), f"转换声称成功但没有 {st}"
    print(f"[worker] model.safetensors = {st.stat().st_size / 2**30:.2f} GiB", flush=True)

    # 绕过上游那个 assets 路径 bug(见文件头说明 2)
    dst = TORCH_CKPT / "assets"
    if dst.exists():
        print(f"[worker] {dst} 已存在(上游这次居然拷对了?),跳过", flush=True)
    else:
        print(f"[worker] 补拷 assets: {JAX_CKPT/'assets'} -> {dst}", flush=True)
        shutil.copytree(JAX_CKPT / "assets", dst)

    ns = TORCH_CKPT / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    assert ns.exists(), f"norm_stats.json 没到位: {ns}"
    print(f"[worker] norm_stats.json OK ({ns.stat().st_size} bytes)", flush=True)
    print("[worker] ALL DONE", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--_worker", action="store_true", help="内部用:在子进程里真正干活")
    args = ap.parse_args()

    if args.status:
        return do_status()
    if args._worker:  # noqa: SLF001
        return run_worker(args.precision)

    assert (JAX_CKPT / "params").is_dir(), f"找不到 JAX params: {JAX_CKPT/'params'}"
    assert (JAX_CKPT / "assets").is_dir(), f"找不到 assets: {JAX_CKPT/'assets'}"
    assert (PATCHED_TF / "transformers" / "models" / "siglip" / "check.py").exists(), \
        "先跑 setup_torch_transformers.py"
    OUT.mkdir(parents=True, exist_ok=True)

    log = open(LOG, "wb")
    p = subprocess.Popen(
        [SERVER_PY, __file__, "--_worker", "--precision", args.precision],
        cwd=str(OPENPI), env=child_env(), stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,   # 脱离终端
    )
    print(f"[convert] worker pid={p.pid} → {LOG}")
    print("[convert] 关终端不受影响;进度看 --status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
