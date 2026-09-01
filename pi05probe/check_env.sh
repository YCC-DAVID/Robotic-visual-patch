#!/usr/bin/env bash
# S0 环境自检:确认两个 conda env + PYTHONPATH 覆盖真的把源码指到了我们自己的 clone。
# 用法:bash pi05probe/check_env.sh 2>&1 | tee pi05probe/out/s0_env_check.txt
set -x
source /home/user1/workspace/chence/WAMattack/pi05probe/env.sh

echo "############ server env (py3.11) ############"
PYTHONPATH="$SERVER_PYTHONPATH" "$SERVER_PY" - <<'PY'
import openpi, openpi_client, jax, flax, torch, transformers, numpy
print("openpi        ->", openpi.__file__)
print("openpi_client ->", openpi_client.__file__)
print("jax", jax.__version__, "flax", flax.__version__, "torch", torch.__version__,
      "transformers", transformers.__version__, "numpy", numpy.__version__)
print("jax devices:", len(jax.devices()), jax.devices()[0])
assert "/WAMattack/" in openpi.__file__, "openpi 没指到我们的 clone!"
assert "/WAMattack/" in openpi_client.__file__, "openpi_client 没指到我们的 clone!"
print("SERVER SRC OK")
PY

echo "############ client env (py3.8) ############"
PYTHONPATH="$CLIENT_PYTHONPATH" "$CLIENT_PY" - <<'PY'
import os, mujoco, robosuite, torch, numpy
print("mujoco", mujoco.__version__, "robosuite", robosuite.__version__,
      "torch", torch.__version__, "numpy", numpy.__version__)
import openpi_client
print("openpi_client ->", openpi_client.__file__)
import libero, libero.libero as ll
print("libero        ->", libero.__path__[0])
print("LIBERO_CONFIG_PATH =", os.environ["LIBERO_CONFIG_PATH"])
for k in ("bddl_files", "init_states", "assets"):
    p = ll.get_libero_path(k)
    print(f"  {k:12s} -> {p}   exists={os.path.exists(p)}")
    assert "/WAMattack/" in p, f"{k} 指到了 WAMattack 外面!"
assert "/WAMattack/" in openpi_client.__file__
assert "/WAMattack/" in libero.__path__[0]

# 真开一个 env,确认 EGL 渲染能出图
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import pathlib
ts = benchmark.get_benchmark_dict()["libero_goal"]()
task = ts.get_task(0)
bddl = pathlib.Path(ll.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
env.seed(7); env.reset()
obs = env.set_init_state(ts.get_task_init_states(0)[0])
img = obs["agentview_image"]
print("task.language =", repr(task.language), "| bddl =", task.bddl_file)
print("agentview_image", img.shape, img.dtype, "min/max", img.min(), img.max())
print("ngeom =", env.env.sim.model.ngeom, "| nq =", env.env.sim.model.nq,
      "| flatten =", len(env.env.sim.get_state().flatten()))
env.close()
print("CLIENT SRC + EGL RENDER OK")
PY
echo "=== check_env done ==="
