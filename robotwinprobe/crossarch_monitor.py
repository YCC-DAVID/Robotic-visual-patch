"""Watch one existing experiment and queue progress/alerts to its Codex chat."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time
from zoneinfo import ZoneInfo

from crossarch_common import DEFAULT, ROOT, dump

CODEX_CLI = '/home/user1/.local/bin/codex'
ZONE = ZoneInfo('America/New_York')


def specialist_plan(root, protocol, progress, health):
    """Report separate work units and an empirical ETA, never overall update %."""
    models = ('pi', 'wam')
    phases = protocol['patch']['stages']
    steps = protocol['patch']['steps']
    patches = (progress or {}).get('patches', {})
    expected_per_model = len(protocol['evaluation']['seeds']) * len(protocol['evaluation']['conditions'])
    episodes = {model: [] for model in models}
    runtime_errors = []
    for model in models:
        for path in (root / 'episodes' / model).glob('*/seed*.json'):
            record = json.loads(path.read_text())
            if 'success' in record:
                episodes[model].append((path.stat().st_mtime, record))
            elif record.get('error'):
                runtime_errors.append(dict(path=str(path), error=record['error']))
    plan = dict(
        objective='同一RoboTwin任务下，训练π0.5和ImageWAM各4张阶段patch，共8张；'
                  '分别测全程和仅对应阶段曝光，定位阶段干扰效果。联合训练及跨模型迁移待本轮结果后再做。',
        patches_completed=sum(bool(p.get('complete')) for p in patches.values()),
        patches_expected=len(models) * len(phases),
        updates_completed=sum(p['iteration'] for p in patches.values()),
        updates_expected=len(models) * len(phases) * steps,
        closedloop_completed=sum(len(rows) for rows in episodes.values()),
        closedloop_expected=len(models) * expected_per_model,
        runtime_errors=runtime_errors,
        results_summary='已完成' if health == 'complete' else '待闭环完成后汇总ASR和Random配对对照')
    if health != 'running':
        plan['eta'] = dict(available=health == 'complete', remaining_training_s=0 if health == 'complete' else None,
            remaining_total_s=[0, 0] if health == 'complete' else None,
            basis='任务已完成' if health == 'complete' else '当前异常或疑似停滞，暂停给出完成时间估计')
        return plan
    training_s = 0.
    for phase in phases:
        remaining = []
        for model in models:
            patch = patches.get(model + '_' + phase, {})
            iterations = patch.get('iteration', 0)
            rates = [p['elapsed_s_this_run'] / p['iteration'] for p in patches.values()
                     if p['model'] == model and p.get('iteration', 0) > 0]
            if iterations < steps and not rates:
                plan['eta'] = dict(available=False, basis='尚无纹理更新耗时，暂不能估计全流程剩余时间')
                return plan
            remaining.append((steps - iterations) * median(rates) if iterations < steps else 0.)
        training_s += max(remaining)  # Models parallel; specialist phases sequential.
    smoke = sorted([(path.stat().st_mtime, json.loads(path.read_text()))
                    for path in (root / 'smoke').glob('*/*/seed*.json')])
    overheads = [now - previous - record['elapsed_s'] for (previous, _), (now, record) in zip(smoke, smoke[1:])
                 if 'elapsed_s' in record and 0 <= now - previous - record['elapsed_s'] < 120]
    overhead = median(overheads) if overheads else 10.
    low = high = training_s
    observed = 0
    for model in models:
        left = max(0, expected_per_model - len(episodes[model]))
        recent = [r['elapsed_s'] for _, r in sorted(episodes[model])[-20:] if r.get('elapsed_s', 0) > 0]
        observed += len(episodes[model])
        if len(recent) >= 4:
            per_low = .85 * median(recent) + overhead
            per_high = max(1.25 * median(recent), max(recent)) + overhead
        else:
            reference = [r['elapsed_s'] for _, r in smoke if r.get('model') == model and r.get('elapsed_s', 0) > 0]
            source = Path(protocol.get('source_run', str(root)))
            for path in (source / 'pilot' / model / 'clean').glob('seed*.json'):
                record = json.loads(path.read_text())
                if 'success' in record and record.get('elapsed_s', 0) > 0:
                    reference.append(record['elapsed_s'])
            reference += recent
            if not reference and left:
                plan['eta'] = dict(available=False, remaining_training_s=training_s,
                    basis='暂无闭环耗时参考；仅能估计纹理训练剩余时间')
                return plan
            per_low = .9 * median(reference) + overhead if reference else 0.
            per_high = 1.5 * max(reference) + overhead if reference else 0.
        low += left * per_low
        high += left * per_high
    plan['eta'] = dict(available=True, remaining_training_s=training_s,
        remaining_total_s=[low, high], formal_episodes_observed=observed,
        basis=('按各模型最近最多20次正式闭环耗时滚动估计；不足4次时参考流程测试及历史clean耗时'
               if observed else '正式闭环未开始，参考流程测试及历史clean耗时；含实测启动/轮询开销和初估余量'),
        limitation='经验估计范围，不是置信区间；按实际闭环速度更新，未包含异常中断时间')
    return plan


def eta_text(eta):
    train = eta.get('remaining_training_s')
    parts = []
    if train is not None:
        parts.append('纹理训练已完成' if train <= 0 else f'纹理训练约剩{max(1, round(train / 60))}分钟')
    bounds = eta.get('remaining_total_s')
    if bounds is not None:
        parts.append(f'整轮剩余约{bounds[0] / 3600:.1f}–{bounds[1] / 3600:.1f}小时（含未完成闭环）')
    parts.append(eta['basis'])
    if eta.get('limitation'):
        parts.append(eta['limitation'])
    return '；'.join(parts)


def snapshot(root):
    status = json.loads((root / 'status.json').read_text())
    pid = status.get('pid')
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        alive = any(x.endswith((b'/crossarch_pipeline.py', b'/crossarch_stage_pipeline.py')) for x in command)
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        alive = False
    training = root / 'training_progress.json'
    progress = json.loads(training.read_text()) if training.exists() else None
    traces = [json.loads(f.read_text()) for f in (root / 'trajectories').glob('seed*/complete.json')]
    counts = {split: len(list((root / 'bank' / split).glob('*.npz'))) for split in ('train', 'validation')}
    expected = {split: sum(len(t['selected']) * (4 if split == 'train' else 2)
                          for t in traces if t['split'] == split) for split in counts}
    phase = status['stage']
    health = 'running'
    if phase == 'complete':
        health = 'complete'
    elif phase == 'stopped_on_error':
        health = 'failed'
    elif not alive:
        health = 'process_exited'
    elif time.time() - status['updated'] > 900:
        health = 'possibly_stalled'
    protocol=json.loads((root/'protocol.json').read_text())
    specialists=protocol.get('experiment')=='stage_specialists'
    closedloop = {model: {condition: len(list((root / ('episodes' if specialists else 'closedloop') / model / condition).glob('seed*.json')))
                         for condition in protocol['evaluation']['conditions']} for model in ('pi', 'wam')}
    row = dict(checked_at=time.time(), checked_at_local=datetime.now(ZONE).isoformat(timespec='seconds'),
               health=health, pipeline_alive=alive, pipeline_status=status, traces_completed=len(traces),
               traces_expected=32, calibration_completed=counts, calibration_expected_from_traces=expected,
               training=progress, closedloop_record_counts=closedloop,
               post_records=len(list((root / 'post').glob('*.npz'))),run_root=str(root),
               experiment=protocol.get('experiment','crossarch_joint'))
    if specialists:
        row['task_plan'] = specialist_plan(root, protocol, progress, health)
    return row


def message(row, kind, root):
    status = row['pipeline_status']
    if status['stage'].startswith('trace_'):
        phase = '四阶段专家轨迹回放'
    elif status['stage'].startswith('capture_'):
        phase = '真实渲染标定数据采集'
    elif status['stage'].startswith('proxy_validation_'):
        phase = '可微代理保真检验'
    elif status['stage'] == 'patch_training':
        phase = 'Patch优化'
    elif status['stage'].startswith('eval_'):
        phase = '真实渲染闭环评估'
    else:
        phase = status['stage']
    train = row['training']
    specialists=row.get('experiment')=='stage_specialists'
    run_root=Path(row.get('run_root',root))
    train_description=(f'8张阶段patch更新总进度：{train["iteration"]}/{train["total"]}，已完成{train["patches_completed"]}/8张。' if specialists and train else
        f'三张patch更新进度：{train["iteration"]}/500。' if train else 'Patch纹理优化尚未开始。')
    text = [f'第31号RoboTwin跨架构patch实验：{kind}。检查时间 {row["checked_at_local"]}。',
            f'任务状态：{row["health"]}；阶段：{phase}（{status["stage"]}）；PID {status.get("pid")}。',
            f'四阶段轨迹：{row["traces_completed"]}/32。',
            f'标定数据已完成：训练{row["calibration_completed"]["train"]}，验证{row["calibration_completed"]["validation"]}；完整轨迹对应的目标条数见latest.json。',
            train_description,
            f'当前实验目录：{run_root}。闭环记录数：{json.dumps(row["closedloop_record_counts"],ensure_ascii=False)}。',
            f'错误：{status.get("error", "无已记录流水线错误")}。',
            f'实时快照：{root / "monitor/latest.json"}。主报告：{ROOT / "result/31_robotwin_crossarch_joint.md"}；分阶段报告：{ROOT / "result/32_robotwin_stage_specialists.md"}。',
            '这是用户已授权的定时监控消息。请核对最新status.json和latest.json后，用中文简洁汇报：'
            '当前任务目标、任务计划完成进度、任务状态和错误、预估剩余时间。'
            '采集/优化/闭环分开报告，未开始的阶段明确写未开始；训练更新百分比不得当作整轮完成百分比。'
            '剩余时间分别说明纹理训练和整轮闭环，并交代估计依据与不确定性。',
            '不要重复创建监控或改动实验协议。失败告警先报告错误及对应日志；疑似停滞不能写成已确认失败。']
    recovery = row.get('recovery_policy', {})
    if recovery.get('authorized'):
        text.append('用户最新授权的异常处理：环境、资源、进程或I/O等运行故障，核实原因和进程归属后尽量自行修复，'
                    '保留原始错误、日志及checkpoint，在原协议下从已有进度续跑；必要时只重启该实验自己拥有的进程，'
                    '无需再次等待用户确认。运行错误不得计作任务失败，修复及恢复结果及时汇报。'
                    '机制或方法问题须及时报告，不擅自调整目标、模型输入/渲染/梯度、阶段定义、EOT、种子、预算、统计口径或放宽检查；'
                    '不得通过跳过失败样本或重跑已成功回合改变实验结果。疑似停滞先排查，不直接认定失败或盲目重启。')
    else:
        text.append('未获运行故障恢复授权时，不重启实验。')
    plan = row.get('task_plan')
    if plan:
        expected = row['calibration_expected_from_traces']
        completed = row['calibration_completed']
        text[1:1] = [f'当前任务目标：{plan["objective"]}',
            f'任务计划完成进度：轨迹{row["traces_completed"]}/{row["traces_expected"]}；'
            f'训练标定{completed["train"]}/{expected["train"]}，验证标定{completed["validation"]}/{expected["validation"]}；'
            f'patch完成{plan["patches_completed"]}/{plan["patches_expected"]}张，更新{plan["updates_completed"]}/{plan["updates_expected"]}；'
            f'正式闭环{plan["closedloop_completed"]}/{plan["closedloop_expected"]}；结果汇总：{plan["results_summary"]}。',
            f'预估剩余时间：{eta_text(plan["eta"])}。']
        if row['health'] == 'failed':
            text[1:1] = [f'错误：{status.get("error", "见status.json")}；对应日志：{run_root / "logs"}。']
    return '\n'.join(text)


def queue(thread, text):
    result = subprocess.run([CODEX_CLI, 'queue', '--thread', thread, '--message', text],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def run(args):
    dest = args.root / 'monitor'
    dest.mkdir(exist_ok=True)
    lock = (dest / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    now = time.time()
    prior=json.loads((dest/'status.json').read_text()) if (dest/'status.json').exists() else {}
    monitor = dict(pid=os.getpid(), thread=args.thread, interval_s=args.interval, poll_s=args.poll,
                   started=now, next_report=prior.get('next_report',now+args.interval), active=not args.once,
                   timezone='America/New_York', delivery='codex queue to existing chat')
    dump(dest / 'status.json', monitor)
    last_alert = None
    last_alert_at = 0
    while True:
        try:
            target_path=dest/'target.json'
            target=Path(json.loads(target_path.read_text())['run_root']) if target_path.exists() else args.root
            row = snapshot(target)
            preferences_path = dest / 'reporting_preferences.json'
            if preferences_path.exists():
                preferences = json.loads(preferences_path.read_text())
                row['recovery_policy'] = preferences.get('runtime_recovery', {})
            dump(dest / 'latest.json', row)
            if args.once:
                print(json.dumps(row, ensure_ascii=False))
                return
            kind = None
            alert = None
            if row['health'] != 'running':
                status = row['pipeline_status']
                alert = (row['health'], status['stage'], status.get('error'), status.get('pid'))
                # A still-failing experiment is repeated at the normal reporting cadence.
                if alert != last_alert or time.time() - last_alert_at >= args.interval:
                    kind = '完成通知' if row['health'] == 'complete' else '即时异常告警'
            if kind is None and time.time() >= monitor['next_report']:
                kind = '每30分钟进度报告'
            if kind:
                receipt = queue(args.thread, message(row, kind, args.root))
                record = dict(sent_at=time.time(), kind=kind, health=row['health'], receipt=receipt)
                with (dest / 'deliveries.jsonl').open('a') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                monitor.update(last_delivery=record, next_report=time.time() + args.interval)
                monitor.pop('last_error', None)
                if alert is not None:
                    last_alert, last_alert_at = alert, time.time()
                if row['health'] == 'complete':
                    monitor['active'] = False
                    dump(dest / 'status.json', monitor)
                    return
            monitor['last_check'] = time.time()
            dump(dest / 'status.json', monitor)
        except Exception as error:
            monitor.update(last_error=repr(error), last_check=time.time())
            dump(dest / 'status.json', monitor)
            print('[monitor error]', repr(error), flush=True)
            if args.once:
                raise
        time.sleep(args.poll)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=DEFAULT)
    ap.add_argument('--thread', required=True)
    ap.add_argument('--interval', type=int, default=1800)
    ap.add_argument('--poll', type=int, default=30)
    ap.add_argument('--once', action='store_true')
    run(ap.parse_args())
