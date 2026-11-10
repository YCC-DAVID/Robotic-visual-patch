"""Read-only experiment monitoring with durable delivery to the existing chat."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'out/stage_joint_20261026'
CODEX = '/home/user1/.local/bin/codex'
ZONE = ZoneInfo('America/Los_Angeles')
GOAL = ('验证单一 π0.5 在 place_container_plate 整个任务中的 patch 干扰能力；'
        '比较接近阶段训练的 E 与四阶段联合训练的 J-mean/J-balanced，'
        '分别测试全程暴露、分阶段开始持续暴露、单次推理暴露，并与 Clean/Random 配对比较。')


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def process(pid, root, script=None):
    try:
        proc = Path('/proc') / str(int(pid))
        args = proc.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
        stat = proc.joinpath('stat').read_text().split(') ', 1)[1].split()
        if stat[0] == 'Z' or str(root) not in args:
            return None
        if script and not any(Path(arg).name == script for arg in args):
            return None
        return dict(pid=int(pid), args=args, state=stat[0], start_ticks=stat[19])
    except (OSError, ValueError, TypeError):
        return None


def snapshot(root, now=None):
    now = time.time() if now is None else now
    status = read(root / 'status.json', {})
    summary = read(root / 'summary.json', {})
    progress = read(root / 'training_progress.json')
    phase = status.get('stage', 'missing_status')
    pipeline = process(status.get('pid'), root, 'stage_joint_pipeline.py')
    backend = []
    workers = []
    for path in Path('/proc').glob('[0-9]*'):
        entry = process(path.name, root)
        if entry is None:
            continue
        scripts = {Path(a).name for a in entry['args']}
        if 'stage_joint_backend.py' in scripts:
            backend.append(entry)
        elif scripts.intersection({'stage_joint_capture.py', 'stage_joint_train.py',
                                   'stage_joint_rollout.py', 'stage_joint_diagnostics.py',
                                   'stage_joint_report.py'}):
            workers.append(entry)
    counts = {split: len(list((root / 'bank' / split).glob('*.npz')))
              for split in ('train', 'validation')}
    records = []
    errors = []
    for folder in ('episodes', 'pilot'):
        for path in (root / folder).glob('*.json'):
            row = read(path, {})
            if folder == 'episodes':
                records.append(row)
            if row.get('error') or row.get('status') == 'runtime_error':
                errors.append(dict(file=str(path), error=row.get('error', 'runtime_error')))
    problems = []
    health = 'running'
    terminal = phase in ('complete', 'stopped_on_error')
    if phase == 'stopped_on_error':
        health = 'failed'
        problems.append(status.get('error', 'pipeline stopped_on_error'))
    elif phase == 'complete':
        health = 'complete'
    elif not pipeline:
        health = 'process_exited'
        problems.append('主流水线进程不存在或 PID 已被复用')
    if not terminal:
        if not backend:
            problems.append('独立 π0.5 模型服务进程缺失')
            if health == 'running':
                health = 'service_missing'
        if now - status.get('updated_unix', 0) > 900:
            problems.append('流水线状态超过 15 分钟未更新，疑似停滞')
            if health == 'running':
                health = 'possibly_stalled'
        expected_pid = status.get('child_pid', status.get('adopted_pid'))
        # A child can exit between its completion and the next pipeline tick.
        if expected_pid and not process(expected_pid, root) and now - status.get('updated_unix', now) > 45:
            problems.append(f'当前子进程 {expected_pid} 已退出，流水线尚未更新')
            if health == 'running':
                health = 'child_exited'
    if errors:
        problems.append(f'{len(errors)} 条运行错误；不计作攻击成功')
        if health in ('running', 'complete'):
            health = 'failed'
    # Parent heartbeat alone cannot establish progress. Track real output/log writes.
    evidence = []
    if phase == 'capture':
        evidence = list((root / 'bank/train').glob('*.npz')) + list((root / 'bank/validation').glob('*.npz'))
    elif phase == 'patch_training':
        evidence = [root / 'training_progress.json', root / 'training_state.npz']
    elif phase == 'formal_closed_loops':
        evidence = list((root / 'episodes').glob('*.json'))
    evidence += list((root / 'logs').glob('*.log')) if not terminal else []
    mtimes = [p.stat().st_mtime for p in evidence if p.exists()]
    last_output = max(mtimes) if mtimes else None
    if not terminal and last_output and now - last_output > 1800:
        problems.append('实验输出和日志超过 30 分钟未增长，疑似停滞')
        if health == 'running':
            health = 'possibly_stalled'
    valid = [r for r in records if r.get('status') in ('complete', 'no_opportunity')]
    results = [r for r in summary.get('rows', []) if r.get('n', 0) and r['condition'] != 'random']
    row = dict(checked_at=now, checked_at_local=datetime.fromtimestamp(now, ZONE).isoformat(timespec='seconds'),
                root=str(root), health=health, problems=problems, pipeline_status=status,
                pipeline=pipeline, backend=backend, workers=workers, calibration=counts,
                calibration_expected={'train': 1536, 'validation': 128}, training=progress,
                capture=read(root / 'capture_status.json'), pilot=read(root / 'audit/pilot.json'),
                pilot_records=len(list((root / 'pilot').glob('*.json'))),
                recorded=len(valid), expected=2400,
                actual_episodes=sum(r.get('status') == 'complete' for r in records),
                no_opportunities=sum(r.get('status') == 'no_opportunity' for r in records),
                native_clean_completed=summary.get('native_clean_completed', 0),
                native_clean_successes=summary.get('native_clean_successes', 0),
                paired_results=results, errors=errors,
                diagnostics=read(root / 'diagnostics/progress.json'),
                final_audit=read(root / 'audit/final.json'), last_output_unix=last_output,
                proxy=read(root / 'audit/proxy_review.json'))
    protocol = read(root / 'protocol.json', {})
    diagnostic_target = 0
    for seed in protocol.get('data', {}).get('validation_seeds', []):
        meta = read(root / 'trajectories' / f'seed{seed}' / 'complete.json', {})
        length = meta.get('total_frames', 0)
        if not length:
            continue
        selected = set(range(0, length, protocol['data'].get('validation_stride', 8))) | {length - 1}
        for lo, hi in meta.get('spans', {}).values():
            selected.update((lo, hi - 1))
        diagnostic_target += len(selected)
    diagnostic_done = 0
    saved_seeds = set()
    for path in (root / 'diagnostics').glob('seed*.json'):
        data = read(path, {})
        if data.get('complete'):
            saved_seeds.add(data['seed'])
            diagnostic_done += data.get('samples', 0)
    diagnostic_progress = row['diagnostics'] or {}
    if diagnostic_progress.get('seed') not in saved_seeds and not diagnostic_progress.get('complete'):
        # Each sampled state has four patched textures times three fixed noises.
        diagnostic_done += diagnostic_progress.get('rows', 0) // 12
    row.update(goal=GOAL, diagnostic_completed_states=diagnostic_done,
               diagnostic_expected_states=diagnostic_target,
               diagnostic_completed_trajectories=len(saved_seeds))
    row['plan'] = plan_progress(row)
    return row


def plan_progress(row):
    training = row['training'] or {}
    diagnostic = row['diagnostics'] or {}
    phase = row['pipeline_status'].get('stage', '')
    definitions = [
        ('标定采集', sum(row['calibration'].values()), 1664,
         all(row['calibration'][k] >= v for k, v in row['calibration_expected'].items()), phase == 'capture'),
        ('注入与回放预实验', row['pilot_records'], 12,
         bool((row['pilot'] or {}).get('passed')), phase.startswith('pilot_')),
        ('三版 patch 训练', training.get('iteration', 0), 500,
         bool(training.get('complete')), phase == 'patch_training'),
        ('独立轨迹实际渲染诊断', row['diagnostic_completed_states'], row['diagnostic_expected_states'],
         bool(diagnostic.get('complete')), phase in ('dense_true_render_diagnostics', 'diagnostic_figures')),
        ('正式闭环评测', row['recorded'], row['expected'],
         row['recorded'] >= row['expected'] and not row['errors'], phase == 'formal_closed_loops'),
        ('最终审计、图表与视频', int(phase == 'complete'), 1,
         phase == 'complete', phase in ('final_audit', 'figures_and_videos'))]
    return [dict(name=name, completed=done, total=total,
                 state='已完成' if complete else '进行中' if active else '待执行')
            for name, done, total, complete, active in definitions]


def estimates(row, history):
    """Extrapolate measured progress; never assign an invented rate to future work."""
    phase = row['pipeline_status'].get('stage')
    counts = {'capture': (sum(row['calibration'].values()), 1664),
              'patch_training': ((row['training'] or {}).get('iteration', 0), 500),
              'dense_true_render_diagnostics': (row['diagnostic_completed_states'], row['diagnostic_expected_states']),
              'formal_closed_loops': (row['recorded'], row['expected'])}
    completed, target = counts.get(phase, (0, 0))
    sample = dict(time=row['checked_at'], phase=phase, completed=completed)
    history = [h for h in history if 0 <= sample['time'] - h['time'] <= 7200]
    recent = [h for h in history if h['phase'] == phase and 60 <= sample['time'] - h['time'] <= 1800
              and h['completed'] < completed]
    eta = dict(stage_remaining_s=None, stage_range_s=None, total_remaining_s=None,
               basis='同一阶段最近30分钟实测吞吐外推；范围为估计值的0.5–2倍，并非统计置信区间。')
    if row['health'] == 'complete':
        eta.update(stage_remaining_s=0, stage_range_s=[0, 0], total_remaining_s=0,
                   note='全部完成。')
    elif row['health'] != 'running':
        eta['note'] = '存在异常，待恢复后重新估计剩余时间。'
    else:
        seconds = None
        if recent and target:
            previous = min(recent, key=lambda h: h['time'])
            rate = (completed - previous['completed']) / (sample['time'] - previous['time'])
            seconds = max(0, target - completed) / rate
            eta['sample_seconds'] = sample['time'] - previous['time']
        elif phase == 'patch_training' and completed:
            elapsed = (row['training'] or {}).get('elapsed_s_this_run', 0)
            if elapsed > 60:
                seconds = max(0, target - completed) * elapsed / completed
                eta['basis'] = '训练累计耗时外推；重启后优先使用同阶段近期实测吞吐。'
        if seconds is not None:
            eta.update(stage_remaining_s=seconds, stage_range_s=[seconds * .5, seconds * 2])
        if phase == 'formal_closed_loops' and seconds is not None:
            eta['note'] = '主要剩余闭环耗时见区间，另需最终审计、图表与视频；GPU并发及无注入机会比例会影响速度。'
        else:
            eta['note'] = '总体剩余时间暂不能可靠估计：正式闭环吞吐尚无依据；进入评测后更新。'
        if seconds is None:
            eta['note'] = '当前阶段测速样本不足。' + eta['note']
    history.append(sample)
    return eta, history[-250:]


def duration(seconds):
    if seconds < 60:
        return '不足1分钟'
    if seconds < 3600:
        return f'{seconds / 60:.0f}分钟'
    return f'{seconds / 3600:.1f}小时'


def message(row, kind):
    root = Path(row['root'])
    p = row['training'] or {}
    lines = [f'第33号 RoboTwin 全过程干扰实验：{kind}；{row["checked_at_local"]}。',
             f'当前任务目标：{row["goal"]}',
             f'任务计划完成进度：{sum(s["state"] == "已完成" for s in row["plan"])}/6 个阶段完成（各阶段工作量不同，不等同于总耗时比例）。',
             *[f'计划{s+1}：{p["name"]}，{p["completed"]}/{p["total"]}，{p["state"]}。' for s,p in enumerate(row['plan'])],
             f'状态 {row["health"]}；阶段 {row["pipeline_status"].get("stage")}；主进程 {row["pipeline_status"].get("pid")}。',
             f'标定训练 {row["calibration"]["train"]}/1536，验证 {row["calibration"]["validation"]}/128。',
             f'三版 patch 同步训练 {p.get("iteration", 0)}/500；预实验记录 {row["pilot_records"]}/12。',
             f'正式记录 {row["recorded"]}/2400，实际闭环 {row["actual_episodes"]}，无注入机会 {row["no_opportunities"]}。',
             f'已完成原生 clean {row["native_clean_completed"]}/100，其中成功 {row["native_clean_successes"]}。',
             f'错误或异常：{json.dumps(row["problems"], ensure_ascii=False)}。',
             f'实时快照：{root / "monitor/latest.json"}；当前报告：{HERE.parent / "result/33_robotwin_stage_joint.md"}。',
             '这是用户授权的监控汇报，请读取最新快照和 status.json，用中文向用户汇报当前进程、实验进度、实验结果、实验状态。',
             '本任务只做单一 π0.5 全过程/分阶段干扰；不要混入跨架构实验，不创建重复监控。',
             '已存在配对结果时给出 native_full 和各 delayed/pulse 阶段 ASR、有效样本数及相对 Random 增量；未测完说明是中间结果。',
             '无有效样本时明确尚无新 ASR；不要把训练 L0 当成闭环成功率。代理诊断分数不作为失败门槛。',
             '异常消息立即向用户说明异常和日志位置；疑似停滞不能声称确认失败。',
             '用户最新授权：遇到代码、依赖、环境、进程等工程问题，自行定位、修复、验证后恢复实验，不停留在重复告警。'
             '恢复时保留已完成数据、训练纹理和配对结果，从断点续跑；先核对已有进程，避免重复运行。'
             '如果发现原有方法本身有问题，或修复需要改变实验协议、训练目标、阶段定义、评测条件或统计口径，先说明问题并等待用户决定；不要擅自改变方法。'
             '每次修复、验证与恢复都记录到当前实验目录，并向用户说明。']
    eta = row.get('eta', {})
    if eta.get('stage_remaining_s') is not None:
        lines.insert(3, f'预估当前阶段剩余：约{duration(eta["stage_remaining_s"])}；粗略范围 {duration(eta["stage_range_s"][0])}–{duration(eta["stage_range_s"][1])}。')
    lines.insert(4, f'预估整体剩余时间：{eta.get("note", "当前测速样本不足，暂不能可靠估计。")}')
    lines.append('汇报必须包含：当前任务目标、任务计划各阶段完成进度、任务状态与当前进程、实验结果、预计当前阶段和整体剩余时间（说明估算依据或缺失依据）。不要漏掉目标和剩余时间，不把完成阶段数当作总工时百分比。')
    return '\n'.join(lines)


def queue(thread, text):
    result = subprocess.run([CODEX, 'queue', '--thread', thread, '--message', text],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def cleanup_recovered_backend(root):
    """Only stop the replacement service explicitly launched by this study."""
    receipt_path = root / 'monitor/backend_recovery.json'
    receipt = read(receipt_path, {})
    entry = process(receipt.get('pid'), root, 'stage_joint_backend.py')
    if entry and entry['start_ticks'] == receipt.get('start_ticks'):
        os.kill(entry['pid'], signal.SIGTERM)
        receipt.update(stopped_by_monitor=True, stopped_unix=time.time())
        write(receipt_path, receipt)


def run(args):
    dest = args.root / 'monitor'
    dest.mkdir(exist_ok=True)
    if args.once:
        print(json.dumps(snapshot(args.root), ensure_ascii=False, indent=2))
        return
    lock = (dest / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = read(dest / 'status.json', {})
    state = dict(pid=os.getpid(), thread=args.thread, interval_s=args.interval, poll_s=args.poll,
                 started_unix=time.time(), active=True, timezone=str(ZONE),
                 next_report=old.get('next_report', time.time()),
                 delivery='codex queue to this existing chat')
    if old.get('last_delivery'):
        state['last_delivery'] = old['last_delivery']
    last_alert = None
    last_alert_at = 0
    unhealthy_checks = 0
    while True:
        try:
            row = snapshot(args.root)
            row['eta'], history = estimates(row, read(dest / 'progress_history.json', []))
            write(dest / 'progress_history.json', history)
            unhealthy_checks = unhealthy_checks + 1 if row['health'] not in ('running', 'complete', 'failed') else 0
            if row['health'] == 'running':
                last_alert = None
            write(dest / 'latest.json', row)
            kind = None
            alert = (row['health'], tuple(row['problems']))
            if row['health'] in ('failed', 'complete') or unhealthy_checks >= 2:
                if alert != last_alert or time.time() - last_alert_at >= args.interval:
                    kind = '完成通知' if row['health'] == 'complete' else '即时异常告警'
            if kind is None and time.time() >= state['next_report']:
                kind = '每30分钟进度报告'
            if kind:
                receipt = queue(args.thread, message(row, kind))
                record = dict(sent_at=time.time(), kind=kind, health=row['health'], receipt=receipt)
                with (dest / 'deliveries.jsonl').open('a') as out:
                    out.write(json.dumps(record, ensure_ascii=False) + '\n')
                with (dest / 'reports.jsonl').open('a') as out:
                    out.write(json.dumps(dict(kind=kind, snapshot=row), ensure_ascii=False) + '\n')
                state['last_delivery'] = record
                if time.time() >= state['next_report']:
                    state['next_report'] = time.time() + args.interval
                state.pop('last_error', None)
                if row['health'] != 'running':
                    last_alert, last_alert_at = alert, time.time()
                if row['health'] == 'complete':
                    cleanup_recovered_backend(args.root)
                    state.update(active=False, finished_unix=time.time())
                    write(dest / 'status.json', state)
                    return
            state['last_check'] = time.time()
            write(dest / 'status.json', state)
        except Exception as error:
            state.update(last_error=repr(error), last_check=time.time())
            write(dest / 'status.json', state)
            print('[monitor error]', repr(error), flush=True)
            # A broken snapshot must itself be reported, not silently swallowed.
            if time.time() - state.get('last_monitor_error_alert', 0) >= args.interval:
                try:
                    receipt = queue(args.thread, f'全过程实验监视程序异常：{error!r}。请立即汇报并检查 {dest / "status.json"} 和 monitor.log；不要创建重复监控。')
                    state.update(last_monitor_error_alert=time.time(), monitor_error_receipt=receipt)
                    write(dest / 'status.json', state)
                except Exception as delivery_error:
                    print('[delivery error]', repr(delivery_error), flush=True)
        time.sleep(args.poll)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=DEFAULT)
    ap.add_argument('--thread', default=os.environ.get('CODEX_THREAD_ID'))
    ap.add_argument('--interval', type=int, default=1800)
    ap.add_argument('--poll', type=int, default=30)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--launch', action='store_true')
    args = ap.parse_args()
    args.root = args.root.resolve()
    if args.interval < 1 or args.poll < 1 or (not args.once and not args.thread):
        ap.error('positive interval/poll and an existing thread are required')
    if args.launch:
        dest = args.root / 'monitor'
        dest.mkdir(exist_ok=True)
        # Hold a separate launch lock until the child acquires its runtime lock.
        with (dest / 'launch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            old = read(dest / 'status.json', {})
            if process(old.get('pid'), args.root, Path(__file__).name):
                raise RuntimeError('This experiment already has a live monitor')
            argv = [sys.executable, '-u', str(Path(__file__).resolve()), '--root', str(args.root),
                    '--thread', args.thread, '--interval', str(args.interval), '--poll', str(args.poll)]
            with (dest / 'monitor.log').open('a') as log:
                child = subprocess.Popen(argv, cwd=HERE.parent, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(dest / 'launch.json', dict(pid=child.pid, argv=argv, started_unix=time.time()))
            print(json.dumps(dict(pid=child.pid, root=str(args.root))))
    else:
        run(args)
