#!/usr/bin/env python3
"""红线1:在烧算力前验证交叉评估的谓词机制(不需要 GPU/policy server,纯 mujoco)。

核心假设:libero_goal 每个 env 的 object_states_dict 含全部物体/区域,
所以能用 bowl_plate 这一个 env 评另外 3 条 task 的 goal ⇒ 不必开 4 个 step env。
本脚本证实:① 4 条 goal_state 都能在 bowl_plate env 上求值,无 KeyError;
② init 状态下 4 条 goal 都为 False(bowl 没在 plate/cabinet 上、stove 没开)。
"""
import os, pathlib, sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
for p in reversed([OPENPI / "packages" / "openpi-client" / "src", OPENPI / "third_party" / "libero"]):
    sys.path.insert(0, str(p))
os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / "pi05probe" / "libero_config")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYTHONNOUSERSITE"] = "1"

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

B1_TASKS = ["turn_on_the_stove", "put_the_wine_bottle_on_the_rack",
            "put_the_bowl_on_the_plate", "put_the_bowl_on_top_of_the_cabinet"]


def load_task(suite, stem):
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        if pathlib.Path(t.bddl_file).stem == stem:
            return i, t
    raise AssertionError(stem)


def main():
    suite = benchmark.get_benchmark_dict()["libero_goal"]()

    # 取 4 条 goal_state
    goals = {}
    bp_env = None
    for stem in B1_TASKS:
        tid, task = load_task(suite, stem)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.seed(7); env.reset()
        goals[stem] = env.env.parsed_problem["goal_state"]
        print(f"[{stem}] goal_state = {goals[stem]}")
        if stem == "put_the_bowl_on_the_plate":
            bp_env = env
        else:
            env.close()

    print("\nbowl_plate env 的 object_states_dict 键(应含全部物体/区域):")
    keys = sorted(bp_env.env.object_states_dict.keys())
    print("  " + ", ".join(keys))

    # 在 bowl_plate env 的 init 状态上评全部 4 条 goal
    init = np.array(torch.load(get_libero_path("init_states") + "/libero_goal/put_the_bowl_on_the_plate.pruned_init"))
    bp_env.seed(7); bp_env.reset(); bp_env.set_init_state(init[0])

    print("\ninit(ep0)状态下,用 bowl_plate env 评 4 条 goal(期望全 False):")
    all_ok = True
    for stem in B1_TASKS:
        try:
            v = all(bp_env.env._eval_predicate(s) for s in goals[stem])
            print(f"  {stem:34s} = {v}")
        except Exception as e:
            all_ok = False
            print(f"  {stem:34s} ❌ 求值失败: {type(e).__name__}: {e}")
    print("\n结论:" + ("✅ 4 条 goal 都能在单一 env 上求值,交叉评估机制成立。"
                       if all_ok else "❌ 有 goal 求值失败,需换成 4 个并行 env。"))
    bp_env.close()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
