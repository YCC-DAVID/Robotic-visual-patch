"""Local, user-requested half-hour progress messages to this existing Codex thread."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def write(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--thread', required=True)
    parser.add_argument('--initial-report', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'progress_notifications'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = out / 'config.json'
        if config_path.exists():
            config = json.loads(config_path.read_text())
            assert config['thread'] == args.thread
        else:
            config = dict(enabled=True, thread=args.thread, interval_seconds=1800,
                          created_at=time.time(), run_root=str(root))
            write(config_path, config)
        if not config['enabled']:
            return
        state_path = out / 'status.json'
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        next_at = previous.get('next_report_at', time.time() if args.initial_report else time.time() + 1800)
        state = dict(stage='waiting', pid=os.getpid(), thread=args.thread,
                     interval_seconds=1800, next_report_at=next_at,
                     delivered=previous.get('delivered', 0))
        write(state_path, state)
        while True:
            config = json.loads(config_path.read_text())
            if not config['enabled']:
                state.update(stage='disabled', stopped_at=time.time())
                write(state_path, state)
                return
            if time.time() < next_at:
                time.sleep(min(30, next_at - time.time()))
                continue
            # Advance before delivery to avoid duplicate messages after a crash.
            next_at = time.time() + 1800
            state.update(stage='delivering', next_report_at=next_at, attempted_at=time.time())
            write(state_path, state)
            pipeline = root / 'pipeline_status.json'
            finished = pipeline.exists() and json.loads(pipeline.read_text()).get('complete', False)
            prompt = (
                '【用户授权的每30分钟实验进度汇报】请现在检查并向用户汇报 RoboTwin '
                'place_container_plate 的视觉编码器冻结/不冻结配对实验。不要切回 G1 simple-pick。'
                f'运行目录：{root}；报告：WAMattack/result/30_robotwin_vision_freeze.md。'
                '请读取最新状态文件、训练日志和实际进程，报告：1.当前任务目标；'
                '2.任务计划和各阶段完成进度（模型训练、patch训练、交叉测试）；'
                '3.实际任务状态及阻塞原因；4.基于实测速度的预计剩余时间，'
                '无法估计的阶段明确写未知，不编造总ETA。没有活进程不能说正在训练。'
                '汇报后继续推进此前已授权的实验；本条通知不代替检查、不授权绕过验证、'
                '不授权影响其他实验。通知调度器已存在，不要重复建立。'
                f'如用户要求停止汇报，将 {config_path} 的 enabled 设为 false。'
            )
            if finished:
                prompt += ' 实验状态已标为完成，请核实并给出最终结果；此调度器本次投递成功后自动结束。'
            try:
                result = subprocess.run(
                    ['/home/user1/.local/bin/codex', 'queue', '--thread', args.thread,
                     '--message', prompt], capture_output=True, text=True, timeout=60,
                    cwd=root.parents[2])
                record = dict(at=time.time(), returncode=result.returncode,
                              stdout=result.stdout[-4000:], stderr=result.stderr[-4000:])
            except Exception as exc:
                record = dict(at=time.time(), returncode=-1, error=repr(exc))
            with (out / 'delivery.jsonl').open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            success = record['returncode'] == 0
            state.update(stage='waiting' if success else 'delivery_failed',
                         last_delivery=record, delivered=state['delivered'] + int(success))
            if finished and success:
                config['enabled'] = False
                write(config_path, config)
                state.update(stage='complete', stopped_at=time.time())
                write(state_path, state)
                return
            write(state_path, state)


if __name__ == '__main__':
    main()
