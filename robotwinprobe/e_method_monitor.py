"""Half-hour updates and immediate repair/next-method follow-ups for E study."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from stage_joint_monitor import process,queue,read,write,duration

HERE=Path(__file__).resolve().parent
DEFAULT=HERE/'out/e_method_20261027'
ZONE=ZoneInfo('America/New_York')


def snapshot(root):
    now=time.time();status=read(root/'status.json',{'stage':'prepared'})
    summary=read(root/'summary.json',{})
    stage=status['stage'];health='running'
    problems=[]
    if stage=='complete':health='complete'
    elif stage=='round_complete_below_target' or stage.endswith('_needs_method_iteration'):health='needs_method_iteration'
    elif stage=='stopped_on_error':health='failed';problems.append(status.get('error','pipeline error'))
    elif not any(process(status.get('pid'),root,script) for script in ('e_method_pipeline.py','e_method_refine.py')):
        health='process_exited';problems.append('主流水线不存在')
    elif now-status.get('updated_unix',0)>900:
        health='possibly_stalled';problems.append('流水线状态超过15分钟未更新')
    for split in ('development','test'):
        problems += [str(e) for e in summary.get('results',{}).get(split,{}).get('errors',[])]
    if problems and health=='running':health='failed'
    completed=0;total=0
    if stage=='policy_state_collection':completed=summary.get('collection_seeds_completed',0);total=24
    elif stage=='onpolicy_state_collection':completed=summary.get('onpolicy_collection_seeds_completed',0);total=24
    elif stage.startswith('train_'):
        completed=summary.get('training',{}).get(stage[6:],{}).get('iteration',0);total=500
    elif stage.endswith('_evaluation'):
        split=stage[:-11];completed=summary.get('results',{}).get(split,{}).get('records',0)
        total=len(status.get('conditions',[]))*(32 if split=='development' else 100)
    elif stage=='transfer_diagnostic':
        progress=read(root/'transfer_diagnostic/progress.json',{})
        completed=progress.get('completed_seeds',0);total=32
    elif stage=='training_gradient_conflict':
        progress=read(root/'group_round/progress.json',{})
        completed=progress.get('completed_states',0);total=progress.get('total_states',49)
    return dict(root=str(root),checked_at=now,checked_at_local=datetime.fromtimestamp(now,ZONE).isoformat(timespec='seconds'),
        health=health,problems=problems,status=status,summary=summary,completed=completed,total=total,
        gradient_audit=read(root/'audit/method_gradients.json'),acceptance=read(root/'acceptance.json'))


def update_estimate(row,history):
    now=row['checked_at'];stage=row['status']['stage']
    recent=[h for h in history if h['stage']==stage and 60<=now-h['time']<=1800 and h['completed']<row['completed']]
    estimate=None
    if row['health']=='running' and recent and row['total']:
        first=min(recent,key=lambda h:h['time'])
        rate=(row['completed']-first['completed'])/(now-first['time'])
        estimate=max(0,row['total']-row['completed'])/rate
    row['stage_remaining_s']=estimate
    history=[h for h in history if now-h['time']<7200]
    history.append(dict(time=now,stage=stage,completed=row['completed']))
    return history[-250:]


def message(row,kind):
    root=Path(row['root']);summary=row['summary'];status=row['status']
    training=summary.get('training',{});goal=row.get('acceptance')
    progress='；'.join(f'{name} {value["iteration"]}/500' for name,value in training.items()) or '未开始'
    lines=[f'第34号 RoboTwin E训练改进：{kind}；{row["checked_at_local"]}。',
        '当前任务目标：先把 E 在持续暴露下的真实闭环ASR做稳；冻结π0.5、place_container_plate、site07与10cm尺寸。',
        f'状态：{row["health"]}；阶段：{status["stage"]}；主PID：{status.get("pid")}。',
        '任务计划：策略自身早期状态采集 → 四种损失受控训练 → 开发集筛选；未达标则真实图像机制诊断与下一轮改进，达标后进行训练seed复验、独立测试与审计。',
        f'采集：{summary.get("collection_seeds_completed",0)}/24初态；训练：{progress}。']
    if summary.get('coverage_round'):
        lines.append('当前第五轮：保持原数据和额外500x8预算，将受扰槽位集中到非初始状态；随后补做最佳方法两份独立灰色父纹理加继续训练seed的开发复验，即使首份略低于目标也执行诊断；不降低门槛，独立测试保持封存。')
    elif summary.get('onpolicy_round'):
        lines.append('当前第四轮：冻结首轮纹理在原24训练初态采集approach受扰状态，再比较原数据继续训练与50%受扰状态混合继续训练；同父纹理、同额外500x8预算，原开发集/noise不变，独立测试保持封存。')
        lines.append(f'第四轮受扰训练状态采集：{summary.get("onpolicy_collection_seeds_completed",0)}/24初态；训练轨迹结果不作为闭环ASR。')
    elif summary.get('group_round') or status['stage']=='training_gradient_conflict':
        lines.append('当前第三轮：先诊断训练状态间的梯度冲突，再比较四组均衡采样PGD与冲突梯度投影；仍固定首步L0、同数据/预算及原开发初态，独立测试保持封存。')
    elif summary.get('refinement'):
        lines.append('当前第二轮：首步L0下比较操作侧夹爪目标与操作臂/夹爪固定尺度平衡目标；沿用原数据、预算和开发集，独立测试保持封存。')
    if status['stage']=='transfer_diagnostic':
        lines.append(f'开发集真实图像传递诊断：{row["completed"]}/{row["total"]}初态；固定初始状态与三份动作noise，不新增闭环ASR。')
    if status['stage']=='training_gradient_conflict':
        lines.append(f'训练状态梯度冲突诊断：{row["completed"]}/{row["total"]}状态，不新增闭环ASR。')
    for split,label in [('development','开发集'),('test','独立测试集')]:
        data=summary.get('results',{}).get(split,{})
        lines.append(f'{label}：{data.get("records",0)}条记录，clean成功{data.get("clean_successes",0)}/{data.get("clean_completed",0)}。')
        paired=[r for r in data.get('rows',[]) if r.get('n') and r.get('ASR') is not None and r['condition'] not in ('clean','random')]
        for result in paired:
            lines.append(f'{label} {result["condition"]}：ASR {result["failures"]}/{result["n"]}={result["ASR"]:.1%}；'
                f'同配对Random {result["random_ASR"]:.1%}；增量{100*result["excess_ASR"]:+.1f}个百分点。')
        if not paired:lines.append(f'{label}尚无可报告的有效闭环ASR。')
        elif split=='development':lines.append('以上为开发集方法筛选结果，不能据此宣称独立测试通过或训练seed稳定。')
    if row['stage_remaining_s'] is None:
        lines.append('当前阶段剩余时间：尚无足够同阶段吞吐数据；总体时间须等待各训练目标与闭环测速。')
    else:
        remaining=row['stage_remaining_s']
        lines.append(f'当前阶段预计还需{duration(remaining)}（粗略范围{duration(remaining*.5)}–{duration(remaining*2)}，按最近30分钟实测吞吐外推）；另有后续阶段，总体时间尚无完整估算。')
    lines += [f'异常：{json.dumps(row["problems"],ensure_ascii=False)}。',
        f'实时快照：{root / "monitor/latest.json"}；报告：{HERE.parent / "result/34_robotwin_e_method.md"}。',
        '这是用户授权的监控消息。读取最新状态，用中文汇报目标、各计划进度、进程状态、真实闭环ASR及相对Random增量、剩余时间。',
        '结果缺失时明确未测，不把离线loss当ASR。旧site07、上一轮E与新方法必须在同一开发初态/noise下比较。',
        '用户已授权工程bug自行修复验证并恢复实验；不需要再次征求恢复许可。保留所有完成产物，先检查进程防止重复运行。',
        '用户本轮要求完善E训练方法：若状态needs_method_iteration，报告受控结果与不足，分析瓶颈并继续设计和执行下一轮可审计训练改进；不要只是重复通知，不把一轮结束当成已达到目标。',
        '若存在transfer_diagnostic/summary.json，先读取真实渲染的首步速度、有效维度和完整动作响应诊断，再决定下一轮；仅初始状态诊断不能当整条轨迹因果证据。',
        '保持任务/模型/尺寸/位置与独立测试纪律；不得根据最终测试结果挑初态、改ASR分母或选择checkpoint。若需要改变这些范围，先向用户说明。']
    if row['health']=='complete':lines.append('验收通过并全部完成；报告最终三训练seed的独立测试结果和审计，剩余时间为0。')
    return '\n'.join(lines)


def run(args):
    dest=args.root/'monitor';dest.mkdir(exist_ok=True)
    if args.once:
        print(json.dumps(snapshot(args.root),ensure_ascii=False,indent=2));return
    with (dest/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=read(dest/'status.json',{})
        state=dict(pid=os.getpid(),thread=args.thread,active=True,interval_s=1800,poll_s=30,
            next_report=old.get('next_report',time.time()+1800),started_unix=time.time(),timezone=str(ZONE))
        last_alert=None;last_sent=0
        while True:
            try:
                target=read(dest/'target.json',{}).get('run_root',str(args.root))
                row=snapshot(Path(target))
                history=update_estimate(row,read(dest/'progress_history.json',[]))
                write(dest/'progress_history.json',history);write(dest/'latest.json',row)
                alert=(row['health'],tuple(row['problems']),row['status']['stage'])
                kind=None
                if row['health']=='running':last_alert=None
                elif alert!=last_alert or time.time()-last_sent>=1800:
                    kind='完成通知' if row['health']=='complete' else '方法改进需继续' if row['health']=='needs_method_iteration' else '即时异常告警'
                if kind is None and time.time()>=state['next_report']:kind='每30分钟进度报告'
                if kind:
                    receipt=queue(args.thread,message(row,kind))
                    delivery=dict(sent_at=time.time(),kind=kind,health=row['health'],receipt=receipt)
                    with (dest/'deliveries.jsonl').open('a') as out:out.write(json.dumps(delivery,ensure_ascii=False)+'\n')
                    state.update(last_delivery=delivery,next_report=time.time()+1800)
                    last_alert,last_sent=alert,time.time()
                    if row['health']=='complete':
                        state.update(active=False,finished_unix=time.time());write(dest/'status.json',state);return
                state.update(last_check=time.time());state.pop('last_error',None);write(dest/'status.json',state)
            except Exception as error:
                state.update(last_error=repr(error),last_check=time.time());write(dest/'status.json',state)
                print('[monitor error]',repr(error),flush=True)
                if time.time()-state.get('last_monitor_error_alert',0)>=1800:
                    try:
                        state['error_receipt']=queue(args.thread,f'E训练监视程序报错：{error!r}，请检查 {dest} 并自行修复；不要创建重复监控。')
                        state['last_monitor_error_alert']=time.time();write(dest/'status.json',state)
                    except Exception as delivery_error:print('[delivery error]',repr(delivery_error),flush=True)
            time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--thread',default=os.environ.get('CODEX_THREAD_ID'));parser.add_argument('--once',action='store_true')
    parser.add_argument('--launch',action='store_true');args=parser.parse_args();args.root=args.root.resolve()
    if not args.once and not args.thread:parser.error('Existing thread required')
    if args.launch:
        dest=args.root/'monitor';dest.mkdir(exist_ok=True)
        old=read(dest/'status.json',{})
        if process(old.get('pid'),args.root,'e_method_monitor.py'):raise RuntimeError('Monitor already running')
        with (dest/'monitor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(args.root),'--thread',args.thread],
                cwd=HERE.parent,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(dest/'launch.json',dict(pid=child.pid,started_unix=time.time()));print(child.pid)
    else:run(args)
