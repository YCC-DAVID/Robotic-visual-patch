#!/usr/bin/env bash
# 共用环境定义 —— 用 `source pi05probe/env.sh` 引入。
#
# 策略:**复用本机已有的两个 conda 环境作为"依赖提供者",源码全部指向我们自己的 clone。**
#
#   conda env  openpi-server (py3.11) —— jax 0.5.3 / flax 0.10.2 / torch 2.7.1+cu126 / transformers 4.53.2
#   conda env  openpi-libero (py3.8)  —— mujoco 3.2.3 / robosuite 1.4.1 / torch 1.11.0+cu113 / numpy 1.22.4
#
# 这两个 env 建于 2026-04-16,当时 editable 安装指向 /home/user1/arash/openpi(@a190e00)。
# 我们【不修改这两个 env】(它们可能还被别的项目用),而是用 PYTHONPATH 覆盖:
# sys.path 顺序是 ''、PYTHONPATH、stdlib、site-packages(.pth 追加在 site-packages 之后),
# ⇒ PYTHONPATH 里的路径优先级高于 .pth 里的 /home/user1/arash/openpi/src。
# 已实测验证(见 pi05probe/out/s0_env_check.txt)。
#
# 依赖集合的有效性依据:a190e00..15a9616 只改了 4 个文件,
# 且依赖变化【只有删除】(openpi-client 去掉了 `tree`),没有任何新增 ⇒ 4 月建的 env 完全够用。
#   packages/openpi-client/pyproject.toml | 5 ++---
#   src/openpi/policies/droid_policy.py   | 2 +-
#   src/openpi/policies/libero_policy.py  | 2 +-
#   uv.lock                               | 23 ------
#
# 另:openpi-libero 里 libero 的 editable 安装是【坏的】(__editable___libero_0_1_0_finder.py
# 里 MAPPING 是空 dict ⇒ `import libero` 直接 ModuleNotFoundError)。
# 官方 README:48 本来也是用 PYTHONPATH 挂 libero,我们照办。

ROOT=/home/user1/workspace/chence/WAMattack
OPENPI=$ROOT/third_party/openpi

# ---- 源码路径(全部在 WAMattack 内) ----
SERVER_PY=/home/user1/miniconda3/envs/openpi-server/bin/python
CLIENT_PY=/home/user1/miniconda3/envs/openpi-libero/bin/python

SERVER_PYTHONPATH="$OPENPI/src:$OPENPI/packages/openpi-client/src"
CLIENT_PYTHONPATH="$OPENPI/packages/openpi-client/src:$OPENPI/third_party/libero"

# ---- 权重:复用已有 12 GB,不重下、不复制、不 symlink ----
# 路径规则是 <OPENPI_DATA_HOME>/<netloc>/<path>  (src/openpi/shared/download.py:58-61)
# ⇒ gs://openpi-assets/checkpoints/pi05_libero 落在 <HOME>/openpi-assets/checkpoints/pi05_libero。
# 已有权重恰好就在 ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero(= openpi 的默认 cache),
# 所以直接指向它,一个字节都不用动。
#
# 安全性已核(重要:这是那 12 GB 的真身,删了要重下):
#   download.py:198-202 里 `openpi-assets/checkpoints/` 的失效阈值是 2025-02-03;
#   :205-216 只在 `mtime <= expire_time` 时 rmtree。该目录 mtime = 2026-04-16 ⇒ 更新 ⇒ 不会被删。
export OPENPI_DATA_HOME=/home/user1/.cache/openpi

# ---- LIBERO 路径隔离:不碰全局 ~/.libero/config.yaml(它指向同事的树) ----
export LIBERO_CONFIG_PATH=$ROOT/pi05probe/libero_config

# ---- 渲染 ----
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1        # 防 ~/.local site-packages 泄漏
export PYOPENGL_PLATFORM=egl

# ---- 显存:本机 8×L40S(46 GB/张)已被他人占 18–39 GB。
#      jax 默认 preallocate 75% = 34.5 GB 会 OOM ⇒ 关掉 preallocate。
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
