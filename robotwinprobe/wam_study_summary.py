"""One-shot post-run consolidation and owned-process cleanup; no new experiments."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

from crossarch_common import ROOT,dump,sha
from wam_steps_to_patch import DEFAULT,OBJECTIVES

REPORT=ROOT/'result/38_imagewam_render_attack_summary.md'
ZONE=ZoneInfo('America/New_York')


def complete_records(folder):
    return {r['seed']:r for path in folder.glob('seed*.json')
            if 'success' in (r:=json.loads(path.read_text())) and 'error' not in r}


def paired_results(clean,candidate):
    paired=sorted(set(clean)&set(candidate))
    eligible=[seed for seed in paired if clean[seed]['success']]
    failures=[seed for seed in eligible if not candidate[seed]['success']]
    return dict(completed=len(candidate),task_successes=sum(bool(r['success']) for r in candidate.values()),
        paired=len(paired),eligible=len(eligible),failures=len(failures),failure_seeds=failures,
        ASR=len(failures)/len(eligible) if eligible else None)


def owned_command(command,root):
    allowed={'wam_steps_to_patch.py','wam_flow_patch.py'}
    if not any(Path(part).name in allowed for part in command):return False
    return any(command[i]=='--root' and Path(command[i+1]).resolve()==root.resolve()
               for i in range(len(command)-1))


def owned_process(pid,root):
    if not pid:return None
    try:
        command=[part.decode() for part in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if part]
        if not owned_command(command,root):return None
        stat=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        if stat[0]=='Z':return None
        return dict(pid=pid,command=command,start_ticks=stat[19])
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def run_processes(root):
    status=json.loads((root/'status.json').read_text());pids={status.get('pid'),status.get('training_pid'),status.get('child_pid')}
    ready=root/'ready.json'
    if ready.exists():pids.add(json.loads(ready.read_text()).get('pid'))
    return [row for pid in pids if (row:=owned_process(pid,root))]


def cleanup(root,rows,action):
    for row in rows:
        now=owned_process(row['pid'],root)
        if now and now['start_ticks']==row['start_ticks']:
            try:os.kill(row['pid'],action)
            except ProcessLookupError:pass


def build(root,final=False):
    dest=root/'consolidation';dest.mkdir(exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text());status=json.loads((root/'status.json').read_text())
    previous=ROOT/'robotwinprobe/out/crossarch_joint_20261025/stage_specialists_20261026'
    model={name:complete_records(root/'episodes/wam'/name) for name in ['clean','random_full']+list(OBJECTIVES)}
    clean=model['clean'];results={name:paired_results(clean,model[name]) for name in ['random_full']+list(OBJECTIVES)}
    progress={name:json.loads(path.read_text()) for name in OBJECTIVES
              if (path:=root/'patches/wam'/name/'progress.json').exists()}
    validation=[json.loads(path.read_text()) for path in (root/'patch_validation').glob('*.json')]
    errors=[dict(path=str(path),error=r['error']) for path in (root/'episodes').glob('**/seed*.json')
            if 'error' in (r:=json.loads(path.read_text()))]
    if final:
        assert status['stage']=='complete' and not errors
        assert all(progress[name]['complete'] and progress[name]['iteration']==500 for name in OBJECTIVES)
        assert all(len(model[name])==10 for name in OBJECTIVES) and len(validation)==192
        assert not run_processes(root),'Wait for this run processes to finish before final consolidation'
    final_time=datetime.now(ZONE).isoformat(timespec='seconds')
    summary=dict(updated_at=final_time,final=final,primary_goal='A fixed patch affects ImageWAM execution through real SAPIEN images',
        task=protocol['task'],status=status['stage'],results=results,
        clean=dict(successes=sum(r['success'] for r in clean.values()),completed=len(clean)),training=progress,
        validation_completed=len(validation),errors=errors,stopped=final,
        internal_velocity_controls_are_not_patch_ASR=True,patch_minimum_steps=None)
    dump(dest/'summary.json',summary)
    import csv
    with (dest/'paired_seed_results.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['seed','clean_success','random_success','first_flow_success','final_action_success','first_flow_attack_on_clean_success','final_action_attack_on_clean_success'])
        for seed in protocol['evaluation']['seeds']:
            c=clean.get(seed)
            values=[model[name].get(seed) for name in ['random_full']+list(OBJECTIVES)]
            attacks=[bool(c['success'] and not model[name][seed]['success']) if c and seed in model[name] else '' for name in OBJECTIVES]
            writer.writerow([seed,c['success'] if c else '']+[r['success'] if r else '' for r in values]+attacks)
    val_rows=[]
    import numpy as np
    for texture in ['random']+list(OBJECTIVES):
        for path in ['proxy','render']:
            selected=[r for r in validation if r['texture']==texture and r['path']==path]
            if selected:val_rows.append(dict(texture=texture,path=path,n=len(selected),
                **{key:float(np.median([r[key] for r in selected])) for key in ['first_L0','final_action_MSE','joint_action_RMSE_rad']}))
    with (dest/'validation_summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['texture','path','n','first_L0','final_action_MSE','joint_action_RMSE_rad'])
        writer.writeheader();writer.writerows(val_rows)
    lines=['# ImageWAM真实渲染patch实验：整理汇总','',
        f'整理时间：{final_time}。'+('本轮已完成，已按用户要求停止新增实验；本报告只整理已有结果。' if final else '本轮仍在运行；此为收尾框架，未完成部分不作效果结论。'),'',
        '目标：固定patch实际放入RoboTwin场景，经SAPIEN三相机图像输入ImageWAM，检查真实动作与任务执行是否受到干扰，并与同条件Random比较。'
        '只有place_container_plate/demo_clean一个任务；四阶段是训练采样范围，不是跨任务通用性要求，也不要求每阶段都有效。','',
        '模型为公开ImageWAM FLUX.2 4B RoboTwin权重，策略冻结，只更新10cm/Q128纹理。这里的真实渲染指原生仿真器成像，不是真机执行。','',
        '## 本轮两张patch对照','',
        'first_flow优化首步速度MSE；final_action由图像端完整推理10步后优化最终标准化动作块MSE。'
        '两者共享24条训练轨迹、四阶段平衡采样、训练noise、EOT与500×8更新；计算成本单独记录。'
        '真实patch评估使用原生推理，没有任何人工内部速度覆盖。','',
        f'Clean：{summary["clean"]["successes"]}/{summary["clean"]["completed"]}。clean/Random共20条同规则基线透明复用；两张patch各新增10回合。','',
        '| 条件 | 完成 | 任务成功 | 配对失败/clean成功 | 失败初态 |','|---|---:|---:|---:|---|']
    for name,r in results.items():lines.append(f'| {name} | {r["completed"]}/10 | {r["task_successes"]} | {r["failures"]}/{r["eligible"]} | {r["failure_seeds"]} |')
    if final:
        a=results['first_flow'];b=results['final_action'];rand=results['random_full']
        if b['failures']>a['failures'] and b['failures']>rand['failures']:
            conclusion='本批固定初态中，完整动作目标造成的配对失败计数高于首步目标与Random；只能报告这批探索性计数，不能据此宣布普遍或显著优势。'
        elif a['failures']==0 and b['failures']==0:
            conclusion='本轮两张patch均未在clean成功初态上造成配对任务失败。是否存在更大的真实动作偏移，另看下面的同状态验证；不能把训练损失增长当成已达成干扰目标。'
        else:
            conclusion='本批计数未显示完整动作目标同时优于首步目标与Random；具体效果按逐初态配对表解释，不按训练损失大小排序。'
        lines+=['',conclusion]
    else:lines+=['','最终效果尚待训练、20个新增闭环与192项验证全部完成。']
    lines+=['','## 同状态真实渲染与代理验证','',
        f'完成{len(validation)}/192：16验证状态、2份新noise，Random/两张最终纹理×代理/真实渲染。'
        '首步L0为16×14速度差MSE，最终动作另报告标准化MSE和关节角RMSE；不能与历史求和版L0直接比较。','',
        '| 纹理 | 路径 | 条数 | 首步L0中位数 | 最终动作MSE中位数 | 关节RMSE/rad中位数 |','|---|---|---:|---:|---:|---:|']
    for r in val_rows:lines.append(f'| {r["texture"]} | {r["path"]} | {r["n"]} | {r["first_L0"]:.6g} | {r["final_action_MSE"]:.6g} | {r["joint_action_RMSE_rad"]:.6g} |')
    lines+=['','## 已有证据与解释边界','',
        '- 第32号：ImageWAM四张阶段patch，各自全程/阶段限定曝光，所有条件配对失败均0/9；不是换错模型导致的结论。',
        '- 第35号：最终阶段纹理的真实图像、VAE/FLUX表示和动作响应确有变化；接近/抓取真实L0中位数约为同状态Random的3.4倍，完整动作块关节RMSE仍约0.002–0.004 rad。'
        '不能跨表示空间按距离比例宣称某层“消除了”扰动。',
        '- 原生预处理、首步速度和完整动作曾逐值核对；真实像素保真通过，但代理/真实模型响应仍有方向偏差。保真诊断不能代替真实闭环结果。',
        '- 第34/36号人工速度覆盖只作为内部节点敏感性与恢复控制。末步覆盖靠近输出，不能证明图像patch可跳过前9步，也不能据此确定patch的最少去噪步数。'
        '原数据与SHA256证据保留，解释已修正。',
        '- 当前评估仅一个任务、10个已多次使用的初态，属于探索性配对实验；尚不能推断普遍鲁棒性、跨任务或跨架构效果。'
        '训练EOT覆盖不等于已完成变换位置/光照下的闭环鲁棒性验证。','',
        '## 收尾状态与资料入口','',
        ('当前实验自有进程已退出，停止继续训练、扩样或追加实验；后续只在用户另行指示后推进。' if final else '等待当前既定实验收尾；结束后只核验、整理并关闭本实验自有进程，不扩展实验。'),'',
        '- [当前patch详细报告](37_imagewam_fullflow_patch.md)',
        '- [解释修正与输入端可达性](36_imagewam_control_scope.md)',
        '- [内部速度控制](34_imagewam_orthogonal_control.md)',
        '- [FLUX特性诊断](35_imagewam_flux_diagnosis.md)',
        '- [分阶段真实闭环](32_robotwin_stage_specialists.md)',
        f'- 原始实验：`{root}`',f'- 逐初态CSV：`{dest / "paired_seed_results.csv"}`',
        f'- 验证汇总CSV：`{dest / "validation_summary.csv"}`',f'- 机器可读汇总：`{dest / "summary.json"}`',
        f'- 原始正交控制证据包：`{ROOT / "robotwinprobe/out/wam_orthogonal_confirmed_20261027/evidence.tar.gz"}`',
        '',f'运行错误记录：{errors}。']
    REPORT.write_text('\n'.join(lines)+'\n')
    if final:
        scope=ROOT/'result/36_imagewam_control_scope.md'
        text=scope.read_text()
        historical='当前仍运行的调度器会保留旧格式自动生成原始控制表；该表按本解释说明阅读。报告生成器已更新解释口径，任务结束后再统一重生成当前汇总，不重跑或改写实验数据。'
        finalized='运行期间的旧调度器曾缓存原始控制表格式；本轮结束后，已使用更新后的报告生成器统一解释口径。原始数值、封存证据与实验数据保持原样，没有重跑实验。'
        if historical in text:scope.write_text(text.replace(historical,finalized))
        selected=[path for path in root.rglob('*') if path.is_file() and
                  path.relative_to(root).parts[0] not in {'consolidation','milestones'} and
                  path.name not in {'consolidation_status.json','consolidation_worker.log'} and not path.name.endswith('.tmp')]
        manifest=[dict(path=str(path.relative_to(root)),size=path.stat().st_size,sha256=sha(path)) for path in sorted(selected)]
        dump(dest/'evidence_manifest.json',dict(created_at=final_time,files=manifest,report=str(REPORT)))
        readme=ROOT/'result/README.md';content=readme.read_text()
        entry='**2026-10-27 ImageWAM真实渲染专项已收尾：[汇总与解释修正](38_imagewam_render_attack_summary.md)。** 本轮完成后按用户要求停止新增实验；主结果以真实图像输入下的动作与任务配对结果为准。\n\n'
        if '38_imagewam_render_attack_summary.md' not in content:
            first,rest=content.split('\n',1);readme.write_text(first+'\n\n'+entry+rest.lstrip('\n'))
    return summary


def wait_and_finish(root,thread):
    waiting_since=None;cleanup_attempt=None
    status_file=root/'consolidation_status.json'
    while True:
        policy=json.loads((root/'finish_policy.json').read_text())
        if not policy.get('enabled'):
            dump(status_file,dict(stage='cancelled',updated=time.time()));return
        assert sha(root/'protocol.json')==policy['protocol_sha256'],'Numeric protocol changed; review before consolidating'
        status=json.loads((root/'status.json').read_text())
        if status['stage']!='complete':
            dump(status_file,dict(stage='waiting_current_run',run_stage=status['stage'],pid=os.getpid(),updated=time.time()))
            time.sleep(20);continue
        progress=[json.loads(f.read_text()) for f in (root/'patches/wam').glob('*/progress.json')]
        valid=(sum(p.get('complete') and p['iteration']==500 for p in progress)==2 and
            sum(len(complete_records(root/'episodes/wam'/name)) for name in OBJECTIVES)==20 and
            len(list((root/'patch_validation').glob('*.json')))==192)
        if not valid:
            dump(status_file,dict(stage='waiting_verified_completion',pid=os.getpid(),updated=time.time()))
            time.sleep(10);continue
        active=run_processes(root)
        if active:
            if waiting_since is None:waiting_since=time.monotonic()
            dump(status_file,dict(stage='waiting_owned_cleanup',pid=os.getpid(),processes=active,updated=time.time()))
            if time.monotonic()-waiting_since>90:
                if cleanup_attempt is None:
                    cleanup(root,active,signal.SIGTERM);cleanup_attempt=time.monotonic()
                elif time.monotonic()-cleanup_attempt>20:cleanup(root,active,signal.SIGKILL)
            time.sleep(5);continue
        from wam_steps_to_patch import report
        report(root)  # The original writer has exited; now apply corrected wording safely.
        result=build(root,final=True)
        dump(root/'audit/stop_after_current.json',dict(passed=True,updated=time.time(),active_owned_processes=[],
            protocol_sha256=sha(root/'protocol.json'),no_new_experiments=True,summary_report=str(REPORT)))
        dump(status_file,dict(stage='complete',pid=os.getpid(),updated=time.time(),report=str(REPORT)))
        previous=root/'last_notification.json'
        delivered=json.loads(previous.read_text()) if previous.exists() else {}
        if thread and not (delivered.get('kind')=='完成通知' and delivered.get('returncode')==0):
            message=f'ImageWAM当前轮次已完成并按用户要求停止，已整理至{REPORT}。请读取{root}/consolidation/summary.json后中文汇报；不要启动新实验。'
            completed=subprocess.run(['/home/user1/.local/bin/codex','queue','--thread',thread,'--message',message],capture_output=True,text=True,timeout=30)
            dump(root/'consolidation/final_notification.json',dict(returncode=completed.returncode,receipt=completed.stdout,error=completed.stderr))
        return


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT);parser.add_argument('--wait',action='store_true');parser.add_argument('--thread')
    args=parser.parse_args()
    if args.wait:wait_and_finish(args.root,args.thread)
    else:build(args.root,final=False)
