#!/usr/bin/env python3
"""把 serve_policy_scaled.py 以**脱离会话**的方式起起来,然后立刻退出。

为什么需要
--------
直接用后台 Bash 起 server,会被后台任务的超时(30 分钟)连坐杀掉 —— 之前四个 server
就是这么一起没的(退出码 144)。server 是常驻进程,不该被当成"会结束的任务"管理。
`start_new_session=True` 让它脱离进程组,超时和终端都影响不到它。

用法(py3.8 或任意 python 都行,它只负责 spawn):
    python pi05probe/spawn_server.py --gpu 0 --scale 0.0 --view base --port 8127
    python pi05probe/spawn_server.py --kill        # 关掉本脚本起过的全部 server
"""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out"
PIDS = OUT / "scaled_server_pids.json"
SERVER_PY = "/home/user1/miniconda3/envs/openpi-server/bin/python"


def load():
    return json.loads(PIDS.read_text()) if PIDS.is_file() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--scale", type=float)
    ap.add_argument("--view", default="wrist")
    ap.add_argument("--port", type=int)
    ap.add_argument("--kill", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    rec = load()
    if args.kill:
        for port, info in list(rec.items()):
            try:
                os.kill(info["pid"], signal.SIGTERM)
                print(f"[kill] port {port} pid {info['pid']}")
            except ProcessLookupError:
                print(f"[kill] port {port} pid {info['pid']} 已不在")
            rec.pop(port)
        PIDS.write_text(json.dumps(rec, indent=2))
        return 0
    if args.status:
        for port, info in rec.items():
            alive = pathlib.Path(f"/proc/{info['pid']}").exists()
            print(f"  port {port}  pid {info['pid']}  {info['view']}×{info['scale']}  "
                  f"{'运行中' if alive else '已退出'}")
        return 0

    assert None not in (args.gpu, args.scale, args.port), "起 server 需要 --gpu --scale --port"
    log = OUT / f"server_{args.view}{args.scale:g}_{args.port}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONNOUSERSITE="1")
    cmd = [SERVER_PY, str(PROBE / "serve_policy_scaled.py"),
           "--scale", str(args.scale), "--view", args.view, "--port", str(args.port)]
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)          # ← 关键:脱离进程组
    rec[str(args.port)] = dict(pid=p.pid, gpu=args.gpu, scale=args.scale, view=args.view,
                               log=str(log))
    PIDS.write_text(json.dumps(rec, indent=2))
    print(f"[spawn] pid={p.pid}  GPU{args.gpu}  {args.view}×{args.scale}  port {args.port}")
    print(f"[spawn] log → {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
