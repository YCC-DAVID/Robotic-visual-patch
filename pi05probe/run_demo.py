#!/usr/bin/env python3
"""S0 第 2 步:跑 openpi 官方 LIBERO 评测 demo(π0.5 + pi05_libero checkpoint)。

官方架构是【两进程 + 两 Python 环境 + websocket】(examples/libero/README.md:37-62):
  policy server : py3.11  scripts/serve_policy.py --env LIBERO
  LIBERO client : py3.8   examples/libero/main.py
client 侧 `WebsocketClientPolicy` 连不上会每 5 s 重试(websocket_client_policy.py:34-44),
所以两边可以同时起,不需要等 server ready。

本脚本只负责 **spawn 后立刻退出**:两个子进程用 `start_new_session=True` 完全脱离终端,
关掉 ssh / terminal 不影响它们。日志落在 pi05probe/out/。

用法:
    python pi05probe/run_demo.py                       # 默认 libero_goal × 3 trial/task
    python pi05probe/run_demo.py --trials 50           # 复现官方 98.0 那一栏
    python pi05probe/run_demo.py --suite libero_spatial
    python pi05probe/run_demo.py --status              # 看进度
    python pi05probe/run_demo.py --kill                # 停掉
"""

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
OUT = ROOT / "pi05probe" / "out"

SERVER_PY = "/home/user1/miniconda3/envs/openpi-server/bin/python"
CLIENT_PY = "/home/user1/miniconda3/envs/openpi-libero/bin/python"

# 见 pi05probe/env.sh 的长注释:复用两个 conda env 的依赖,源码用 PYTHONPATH 指回我们自己的 clone。
SERVER_PYTHONPATH = f"{OPENPI/'src'}:{OPENPI/'packages'/'openpi-client'/'src'}"
CLIENT_PYTHONPATH = f"{OPENPI/'packages'/'openpi-client'/'src'}:{OPENPI/'third_party'/'libero'}"

# --torch 时:打过 openpi 补丁的 transformers 必须排在 PYTHONPATH 最前面,
# 才能盖掉 conda env site-packages 里那份(见 setup_torch_transformers.py)。
PATCHED_TF = ROOT / "third_party" / "transformers_patched"
TORCH_PYTHONPATH = f"{PATCHED_TF}:{SERVER_PYTHONPATH}"
TORCH_CKPT = ROOT / "checkpoints" / "pi05_libero_pytorch"

PIDFILE = OUT / "demo_pids.json"


def base_env():
    e = dict(os.environ)
    e.pop("PYTHONHOME", None)
    e["PYTHONNOUSERSITE"] = "1"
    e["PYTHONUNBUFFERED"] = "1"
    # 权重:直接指向已有的 12 GB cache(见 env.sh 注释,已核不会被 download.py 的失效检查删掉)
    e["OPENPI_DATA_HOME"] = "/home/user1/.cache/openpi"
    # LIBERO 路径隔离:全局 ~/.libero/config.yaml 指向同事的树,不能用也不改
    e["LIBERO_CONFIG_PATH"] = str(ROOT / "pi05probe" / "libero_config")
    return e


def server_env(gpu, torch_path=False):
    e = base_env()
    e["PYTHONPATH"] = TORCH_PYTHONPATH if torch_path else SERVER_PYTHONPATH
    e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # L40S 每张 46 GB 但已被他人占 18–39 GB;jax 默认 preallocate 75%(34.5 GB)会 OOM。
    # ⚠️ 即使走 PyTorch 路径也要设:openpi 到处用 jax.tree.map,jax 照样会初始化并抢显存。
    e["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    e["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.40"
    return e


def client_env():
    e = base_env()
    e["PYTHONPATH"] = CLIENT_PYTHONPATH
    e["MUJOCO_GL"] = "egl"
    e["PYOPENGL_PLATFORM"] = "egl"
    # client 只跑 mujoco,不需要 GPU 上的 torch;但 EGL 渲染需要能看到一张卡
    e["CUDA_VISIBLE_DEVICES"] = "0"
    return e


def pick_gpu():
    """挑剩余显存最多的一张卡。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = [(int(i), int(f)) for i, f in (l.split(",") for l in out.strip().splitlines())]
    rows.sort(key=lambda r: -r[1])
    return rows[0]


def spawn(cmd, env, logpath):
    log = open(logpath, "wb")
    p = subprocess.Popen(
        cmd, cwd=str(OPENPI), env=env, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,   # ← 脱离终端的关键
    )
    return p.pid


def do_status():
    if not PIDFILE.exists():
        print("no demo_pids.json — 没在跑过")
        return
    info = json.loads(PIDFILE.read_text())
    print(f"backend={info.get('backend', 'jax')}  suite={info['suite']}  "
          f"trials={info['trials']}  started={info['started']}")
    for name in ("server", "client"):
        pid = info[name]
        alive = pathlib.Path(f"/proc/{pid}").exists()
        print(f"{name:7s} pid={pid} alive={alive}")
    slog = info.get("server_log", str(OUT / "server.log"))
    clog = info.get("client_log", str(OUT / "client.log"))
    print(f"\nlogs:\n  {slog}\n  {clog}")
    p = pathlib.Path(clog)
    if p.exists():
        hits = [l for l in p.read_text(errors="replace").splitlines()
                if "successes:" in l or "Total success rate" in l]
        if hits:
            print(f"\n进度: {hits[-1]}")


def do_kill():
    if not PIDFILE.exists():
        print("no demo_pids.json")
        return
    info = json.loads(PIDFILE.read_text())
    for name in ("client", "server"):
        pid = info[name]
        if pid < 0:          # --server-only 时 client 没起
            continue
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            print(f"killed {name} pgid of pid {pid}")
        except ProcessLookupError:
            print(f"{name} pid {pid} already gone")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--trials", type=int, default=3, help="num_trials_per_task;官方是 50")
    ap.add_argument("--port", type=int, default=8123, help="别用 8000,免得撞上别人的 server")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--torch", action="store_true",
                    help="走 PyTorch 路径(转换后的 checkpoint),而不是默认的 JAX checkpoint")
    ap.add_argument("--server-only", action="store_true",
                    help="只起 policy server,不起官方 eval client(给 rollout_dump.py 用)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--kill", action="store_true")
    args = ap.parse_args()

    if args.status:
        return do_status()
    if args.kill:
        return do_kill()

    OUT.mkdir(parents=True, exist_ok=True)
    gpu, free = pick_gpu()
    print(f"[run_demo] GPU {gpu} 剩余 {free} MiB — server 用它")

    server_cmd = [SERVER_PY, "scripts/serve_policy.py", "--env", "LIBERO", "--port", str(args.port)]
    if args.torch:
        # policy_config.py:49-50 靠输出目录里有没有 model.safetensors 自动切到 PyTorch 路径,
        # 所以只要把 --policy.dir 指到转换后的 checkpoint,API 完全不变。
        st = TORCH_CKPT / "model.safetensors"
        assert st.exists(), f"还没转权重,先跑 convert_weights.py({st} 不存在)"
        server_cmd += ["policy:checkpoint", "--policy.config", "pi05_libero",
                       "--policy.dir", str(TORCH_CKPT)]
        print(f"[run_demo] PyTorch 路径,checkpoint = {TORCH_CKPT}")
    client_cmd = [
        CLIENT_PY, "examples/libero/main.py",
        "--args.host", "127.0.0.1",
        "--args.port", str(args.port),
        "--args.task-suite-name", args.suite,
        "--args.num-trials-per-task", str(args.trials),
        "--args.seed", str(args.seed),
        "--args.video-out-path", str(OUT / "videos" / (args.suite + ("_torch" if args.torch else ""))),
    ]

    tag = "_torch" if args.torch else ""
    slog, clog = OUT / f"server{tag}.log", OUT / f"client{tag}.log"
    spid = spawn(server_cmd, server_env(gpu, torch_path=args.torch), slog)
    print(f"[run_demo] server pid={spid} → {slog}")
    if args.server_only:
        cpid = -1
        print(f"[run_demo] --server-only:不起 client。端口 {args.port}。")
        print("[run_demo] ⚠️ 首次前向要 torch.compile(max-autotune),几分钟后才响应。")
    else:
        time.sleep(2)
        cpid = spawn(client_cmd, client_env(), clog)
        print(f"[run_demo] client pid={cpid} → {clog}")

    PIDFILE.write_text(json.dumps({
        "server": spid, "client": cpid, "gpu": gpu, "port": args.port,
        "suite": args.suite, "trials": args.trials, "seed": args.seed,
        "backend": "pytorch" if args.torch else "jax",
        "server_log": str(slog), "client_log": str(clog),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    print(f"[run_demo] 已写 {PIDFILE};两个进程都 start_new_session,关终端不受影响。")
    print(f"[run_demo] 总 episode 数 = 10 task × {args.trials} = {10*args.trials}")


if __name__ == "__main__":
    sys.exit(main())
