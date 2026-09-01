#!/usr/bin/env bash
# openpi 环境安装 —— 两个 venv,都在 WAMattack 内部。
#
# openpi 的 LIBERO example 是【两进程 + 两环境】架构(examples/libero/README.md):
#   - policy server : 主环境  third_party/openpi/.venv            (Python 3.11)
#   - LIBERO client : 独立环境 third_party/openpi/examples/libero/.venv (Python 3.8)
#   两者走 websocket 通信。
#
# 用法:
#   nohup bash pi05probe/setup_openpi_env.sh > pi05probe/setup_openpi.log 2>&1 &
#
# 装完后 tail 一眼日志末尾的 "=== ALL DONE ===" 和它上面几行的自检输出。

set -x
export PATH="/home/user1/.local/bin:$PATH"
export GIT_LFS_SKIP_SMUDGE=1

ROOT=/home/user1/workspace/chence/WAMattack
OPENPI=$ROOT/third_party/openpi

# 让 openpi 的所有下载/缓存都落在 WAMattack 内,而不是 ~/.cache/openpi
export OPENPI_DATA_HOME=$ROOT/checkpoints/openpi

cd "$OPENPI" || exit 1
uv --version || { echo "!!! uv not found on PATH"; exit 1; }

# ---------------------------------------------------------------- 0) 复用已有权重
# openpi 解析路径的规则是  <OPENPI_DATA_HOME>/<netloc>/<path>
# (src/openpi/shared/download.py:60),所以 gs://openpi-assets/checkpoints/pi05_libero
# 会被找到在 <OPENPI_DATA_HOME>/openpi-assets/checkpoints/pi05_libero。
# 已有的 12 GB 权重在 ~/.cache/openpi/... 下,symlink 过去即可,不重下、不复制。
# 安全性已核:该目录 mtime=2026-04-16,远晚于 openpi 的失效阈值 2025-02-03
# (download.py:201 + :205-216),所以不会触发 shutil.rmtree。
mkdir -p "$OPENPI_DATA_HOME/openpi-assets/checkpoints"
ln -sfn /home/user1/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
        "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"
ls -l "$OPENPI_DATA_HOME/openpi-assets/checkpoints/"
ls    "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/"

# ---------------------------------------------------------------- 1) 主环境(server)
uv sync
echo "=== uv sync exit=$? ==="
uv pip install -e .
echo "=== uv pip install -e . exit=$? ==="

./.venv/bin/python - <<'PY'
import jax, openpi
print("openpi import OK")
print("jax", jax.__version__)
print("jax devices:", jax.devices())
PY
echo "=== server env selfcheck exit=$? ==="

# ---------------------------------------------------------------- 2) client 环境(LIBERO)
# 按 examples/libero/README.md:43-48。注意是 Python 3.8 + cu113,与主环境完全隔离。
uv venv --python 3.8 examples/libero/.venv
uv pip sync --python examples/libero/.venv/bin/python \
    examples/libero/requirements.txt \
    third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match
echo "=== client uv pip sync exit=$? ==="
uv pip install --python examples/libero/.venv/bin/python -e packages/openpi-client
uv pip install --python examples/libero/.venv/bin/python -e third_party/libero
echo "=== client editable installs exit=$? ==="

./examples/libero/.venv/bin/python - <<'PY'
import mujoco, robosuite
print("client mujoco", mujoco.__version__, "robosuite", robosuite.__version__)
import libero, libero.libero as ll
print("libero at", libero.__path__[0])
PY
echo "=== client env selfcheck exit=$? ==="

echo "=== ALL DONE ==="
