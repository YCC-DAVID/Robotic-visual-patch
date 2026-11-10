#!/usr/bin/env python3
"""一条命令跑完整个固定-ε rollout:挑空闲卡 → 起 server → 自检 → 跑 divergence_scan → 关 server。

为什么要它
--------
rollout 需要两个进程配合:policy server(openpi-server,py3.11)+ LIBERO 渲染客户端
(openpi-libero,py3.8),两边跨环境。手动起容易忘关 server、忘等编译、把 server 起在
别人占着的卡上。这个脚本把全过程串起来,并且**只挑真正空闲的卡,绝不碰别人的进程**。

设计要点
-------
- server 用 `start_new_session=True` 脱离进程组启动(spawn_server.py 同款),
  这样本脚本退出/被超时杀掉都不会连坐 server。跑完由本脚本显式 SIGTERM 关掉。
- 只在**本机**挑卡(在哪个节点跑就用哪个节点的卡)。轮询 nvidia-smi,
  出现 free ≥ --need GB 的卡才动手;没有就 sleep 重试。
- 渲染客户端用 MUJOCO_EGL_DEVICE_ID 把它那点 EGL footprint 也落到同一张空闲卡上。
- 全程日志写 out/autorun_rollout.log。

用法(在有空闲卡的节点上):
    /home/user1/miniconda3/envs/openpi-libero/bin/python pi05probe/autorun_rollout.py \
        --episodes 10 --need 9 --port 8137
"""
import argparse
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out"
SERVER_PY = "/home/user1/miniconda3/envs/openpi-server/bin/python"
LIBERO_PY = "/home/user1/miniconda3/envs/openpi-libero/bin/python"
LOG = OUT / "autorun_rollout.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def free_gpus(need_mib):
    """返回 (index, free_mib) 列表,只含 free ≥ need_mib 的卡,按 free 降序。

    ⚠️ **必须排除坏卡**。硬件故障的卡(ERR! 状态)会把整块显存报成"空闲"
    (因为谁也分配不上去),`memory.free` 看它反而最大 —— 本机 GPU6 就是这样,
    utilization.gpu / temperature.gpu 都返回 [N/A]。用 utilization 是不是数字来筛掉。
    """
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    rows = []
    for ln in q.stdout.strip().splitlines():
        parts = [c.strip() for c in ln.split(",")]
        i, fr, util, temp = parts[0], parts[1], parts[2], parts[3]
        # 坏卡:util / temp 是 [N/A] 或 ERR!,直接跳过
        if not util.isdigit() or not temp.isdigit() or not fr.isdigit():
            print(f"[skip] GPU{i} 状态异常(util={util!r} temp={temp!r} free={fr!r}),疑似坏卡,不用",
                  flush=True)
            continue
        if int(fr) >= need_mib:
            rows.append((int(i), int(fr)))
    return sorted(rows, key=lambda r: -r[1])


def wait_ready(port, log_path, timeout_s):
    """等 server 自检通过并开始监听。看它 log 里的关键行,同时试探端口。"""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if log_path.is_file():
            txt = log_path.read_text(errors="ignore")
            if "噪声地板 = 0" in txt and "监听" in txt:
                return True
            if "AssertionError" in txt or "Traceback" in txt:
                raise RuntimeError(f"server 启动失败,见 {log_path}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                if log_path.is_file() and "监听" in log_path.read_text(errors="ignore"):
                    return True
        except OSError:
            pass
        time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--need", type=float, default=9.0, help="需要的空闲显存 GB")
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--preset", default="fine")
    ap.add_argument("--texture", default=None)
    ap.add_argument("--poll", type=int, default=60, help="没空闲卡时的轮询间隔秒")
    ap.add_argument("--max-wait-min", type=float, default=600, help="等空闲卡的最长时间")
    ap.add_argument("--server-timeout", type=int, default=600, help="等 server 就绪的最长秒(含编译)")
    args = ap.parse_args()
    need_mib = int(args.need * 1024)

    log(f"=== autorun 开始 host={socket.gethostname()} 需要 {args.need}GB 空闲 "
        f"preset={args.preset} episodes={args.episodes} ===")

    # ---- 1) 等一张空闲卡 ----
    gpu = None
    t0 = time.time()
    while time.time() - t0 < args.max_wait_min * 60:
        cands = free_gpus(need_mib)
        if cands:
            gpu = cands[0][0]
            log(f"选中 GPU{gpu}(空闲 {cands[0][1]} MiB);全部候选 {cands}")
            break
        log(f"暂无 ≥{args.need}GB 空闲的卡,{args.poll}s 后重试")
        time.sleep(args.poll)
    if gpu is None:
        log("等不到空闲卡,放弃")
        return 1

    # ---- 2) 起固定-ε server(脱离进程组) ----
    slog = OUT / f"fixed_noise_server_{args.port}.log"
    if slog.exists():
        slog.unlink()
    senv = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONNOUSERSITE="1",
                XLA_PYTHON_CLIENT_PREALLOCATE="false")
    scmd = [SERVER_PY, str(PROBE / "serve_policy_fixed_noise.py"), "--port", str(args.port)]
    with open(slog, "w") as f:
        proc = subprocess.Popen(scmd, cwd=str(ROOT), env=senv, stdout=f,
                                stderr=subprocess.STDOUT, start_new_session=True)
    log(f"server pid={proc.pid} GPU{gpu} port={args.port} log={slog.name}")

    try:
        if not wait_ready(args.port, slog, args.server_timeout):
            log("server 超时未就绪,放弃并关闭"); return 1
        log("server 就绪,自检噪声地板 = 0 通过")

        # ---- 3) 跑 divergence_scan(渲染也落到同一张空闲卡) ----
        cenv = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
                    PYTHONNOUSERSITE="1")
        ccmd = [LIBERO_PY, str(PROBE / "divergence_scan.py"), "--preset", args.preset,
                "--episodes", str(args.episodes), "--port", str(args.port)]
        if args.texture:
            ccmd += ["--texture", args.texture]
        log(f"跑 divergence_scan:{' '.join(ccmd[1:])}")
        clog = OUT / "autorun_divergence.log"
        with open(clog, "w") as f:
            rc = subprocess.run(ccmd, cwd=str(ROOT), env=cenv, stdout=f,
                                stderr=subprocess.STDOUT).returncode
        log(f"divergence_scan 退出码 {rc},输出见 {clog.name}")
        tail = "\n".join(clog.read_text(errors="ignore").splitlines()[-30:])
        log("---- divergence_scan 末 30 行 ----\n" + tail)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            log(f"已关闭 server pid={proc.pid}")
        except ProcessLookupError:
            log("server 已不在")
    log("=== autorun 结束 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
