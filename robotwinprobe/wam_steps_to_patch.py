"""Preserved velocity result, minimum-step search, and matched physical patches."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import traceback

import numpy as np

from crossarch_common import ROOT,PY,dump,sha
from crossarch_stage_common import DEFAULT as SOURCE
import wam_orthogonal_control as CONTROL
from wam_flow_patch import OBJECTIVES

DEFAULT=ROOT/'robotwinprobe/out/wam_steps_to_patch_20261027'
SEALED=ROOT/'robotwinprobe/out/wam_orthogonal_confirmed_20261027'
SCAN_REPORT=ROOT/'result/36_imagewam_minimum_denoising_steps.md'
PATCH_REPORT=ROOT/'result/37_imagewam_fullflow_patch.md'
SCOPE_NOTE=ROOT/'result/36_imagewam_control_scope.md'
PORT=18572


def reference(source,target,condition,protocol_hash):
    old=json.loads(source.read_text())
    new=dict(old,condition=condition,protocol_sha256=protocol_hash,reference_reused=True,
        source_record=str(source),source_record_sha256=sha(source),source_protocol_sha256=old['protocol_sha256'])
    if target.exists():assert json.loads(target.read_text())==new
    else:dump(target,new)
    array=target.with_suffix('.npz')
    if not array.exists():os.link(source.with_suffix('.npz'),array)


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    original=json.loads((SOURCE/'protocol.json').read_text());p=copy.deepcopy(original)
    p.update(experiment='imagewam_steps_to_patch',source_run=str(SOURCE),control_port=PORT,
        confirmed_evidence=str(SEALED/'evidence.tar.gz'),confirmed_manifest_sha256=sha(SEALED/'manifest.json'))
    masks=dict(clean=[],single_01=[0],all_10=list(range(10)))
    masks.update({f'single_{i:02d}':[i-1] for i in range(2,11)})
    masks.update({f'prefix_{i:02d}':list(range(i)) for i in range(2,10)})
    p['intervention_masks']=masks
    p['step_search']=dict(seeds=[22000,22001,22002],single_order=list(range(10,1,-1)),prefix_order=list(range(2,10)),
        reliable_effect='all three clean-success episodes fail; runtime errors excluded',
        single_stopping_rule='stop single-step search when one location causes all three failures; then minimum cardinality is one for these seeds',
        prefix_stopping_rule='test consecutive prefixes in ascending order, stop at first 3/3 failure; all-ten endpoint already confirmed',
        no_monotonicity_assumption=True,global_minimum='only claim global minimum=1 if a singleton succeeds; otherwise report minimum within tested prefixes',
        reused_conditions={'clean':'clean','single_01':'orthogonal_first','all_10':'orthogonal_all'},maximum_total_episodes=60)
    p['patch_trial']=dict(objectives=list(OBJECTIVES),steps=500,batch=8,lr=.01,seed=20261027,
        sampling='two samples from each of the four task phases per update; both objectives share identical records and noise',
        initialization='gray .5, Q128,10cm; original 3cm/15degree/three-light calibrated EOT',
        first_flow='effective14 first predicted velocity MSE',
        final_action='effective14 normalized final action-block MSE after complete ten-step differentiable rollout',
        model_frozen=True,scope='two universal full-task textures, not the earlier stage specialists',
        compute='same update/sample count; report separate wall and gradient-compute times, not equal FLOP budgets',
        training_start='after the first single-step location group; objectives fixed before observing search outcomes')
    p['evaluation'].update(conditions=['clean','random_full']+list(OBJECTIVES),seeds=list(range(22000,22010)),
        scope='exploratory same-model first-flow/full-flow comparison, ten previously used fixed initial states',
        exposures=['full'],baseline_reuse='same native inference and shared phase-boundary replanning; new server clean22000 must exactly reproduce the prior full trajectory',
        expected_new_patch_episodes=20,expected_reused_baselines=20)
    diagnosis=json.loads((ROOT/'robotwinprobe/out/wam_flux_diagnosis_20261027/protocol.json').read_text())['diagnosis']
    p['diagnosis']=dict(records=diagnosis['records'],action_noise_seeds=diagnosis['action_noise_seeds'],
        textures=['random']+list(OBJECTIVES),paths=['proxy','render'],expected_measurements=192)
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p,'Preserve this run protocol'
    else:dump(path,p)
    protocol_hash=sha(path)
    for name in ('prompts.npz','provenance.json'):
        if not (root/name).exists():os.link(SOURCE/name,root/name)
    for mode,old in p['step_search']['reused_conditions'].items():
        for seed in p['step_search']['seeds']:
            reference(CONTROL.DEFAULT/'episodes'/old/f'seed{seed}.json',root/'episodes'/mode/f'seed{seed}.json',mode,protocol_hash)
    for mode in ('clean','random_full'):
        for seed in p['evaluation']['seeds']:
            reference(SOURCE/'episodes/wam'/mode/f'seed{seed}.json',root/'episodes/wam'/mode/f'seed{seed}.json',mode,protocol_hash)
    return p


def condition_rows(root,p):
    result=[]
    for mode,mask in p['intervention_masks'].items():
        values=[json.loads(f.read_text()) for f in (root/'episodes'/mode).glob('seed*.json')]
        completed=[r for r in values if 'success' in r]
        result.append(dict(condition=mode,steps=[i+1 for i in mask],count=len(mask),completed=len(completed),
            successes=sum(r['success'] for r in completed),failures=sum(not r['success'] for r in completed),
            reliable=len(completed)==3 and all(not r['success'] for r in completed),
            reused=sum(bool(r.get('reference_reused')) for r in completed)))
    return result


def report(root):
    p=json.loads((root/'protocol.json').read_text());status=json.loads((root/'status.json').read_text());rows=condition_rows(root,p)
    done=root/'scan_complete.json';scan_complete=json.loads(done.read_text()) if done.exists() else None
    single=[r for r in rows if r['condition'].startswith('single_') and r['reliable']]
    progress={}
    for objective in OBJECTIVES:
        f=root/'patches/wam'/objective/'progress.json'
        if f.exists():progress[objective]=json.loads(f.read_text())
    eval_rows=[]
    clean={r['seed']:r for f in (root/'episodes/wam/clean').glob('seed*.json') if 'success' in (r:=json.loads(f.read_text()))}
    for objective in ['random_full']+list(OBJECTIVES):
        values=[json.loads(f.read_text()) for f in (root/'episodes/wam'/objective).glob('seed*.json')]
        values=[r for r in values if 'success' in r];eligible=[r for r in values if clean[r['seed']]['success']]
        eval_rows.append(dict(condition=objective,completed=len(values),task_successes=sum(r['success'] for r in values),
            eligible=len(eligible),failures=sum(not r['success'] for r in eligible)))
    errors=[dict(path=str(f),error=r['error']) for f in (root/'episodes').glob('**/seed*.json') if 'error' in (r:=json.loads(f.read_text()))]
    if status.get('error'):errors.insert(0,dict(path=str(root/'status.json'),error=status['error']))
    validation=[json.loads(f.read_text()) for f in (root/'patch_validation').glob('*.json')]
    val_summary=[]
    for texture in ['random']+list(OBJECTIVES):
        for path in ['proxy','render']:
            selected=[r for r in validation if r['texture']==texture and r['path']==path]
            if selected:val_summary.append(dict(texture=texture,path=path,n=len(selected),
                **{k:float(np.median([r[k] for r in selected])) for k in ('first_L0','final_action_MSE','joint_action_RMSE_rad')}))
    summary=dict(status=status,confirmed_archive=str(SEALED/'evidence.tar.gz'),scan_rows=rows,
        scan_completed_episodes=sum(r['completed'] for r in rows),scan_complete=scan_complete,
        singleton_minimum=1 if single else None,confirmed_singleton_locations=[r['steps'][0] for r in single],
        training=progress,patch_updates=sum(r['iteration'] for r in progress.values()),patch_updates_expected=1000,
        patches_complete=sum(bool(r['complete']) for r in progress.values()),patches_expected=2,
        patch_evaluation=eval_rows,new_patch_episodes=sum(r['completed'] for r in eval_rows if r['condition'] in OBJECTIVES),
        expected_new_patch_episodes=20,validation_completed=len(validation),validation_expected=192,validation=val_summary,errors=errors)
    review=root/'interpretation_review.json'
    summary['interpretation_review']=json.loads(review.read_text()) if review.exists() else None
    summary['minimum_internal_intervention_steps']=summary['singleton_minimum']
    summary['patch_minimum_steps']=None
    summary['patch_reachability_proven_by_internal_control']=False
    dump(root/'summary.json',summary)
    lines=['# ImageWAM：内部去噪速度节点敏感性控制','',f'状态：{status["stage"]}。','',
        f'解释修正：[内部控制与输入端patch的区别]({SCOPE_NOTE})。'
        '自由覆盖第10步内部速度，只能作为靠近输出端的敏感性控制；不证明patch能够跳过前9步、仅影响末步或用一个去噪步完成攻击。','',
        '已固化先前3初态结果：仅首步正交0/3任务失败，全部10步正交3/3失败。证据包含代码、协议、原始数组、clean源记录与独立复算，并有逐文件及压缩包SHA256。','',
        f'证据包：`{SEALED / "evidence.tar.gz"}`。','',
        '本轮保持相同3初态和正交噪声种子。每次重规划选定的去噪步被替换为等长正交速度。'
        '先从第10步向前试单步；若一个单步在3初态均致败，即已确认当前样本在自由内部节点覆盖条件下1个节点足够，停止其余单步搜索。'
        '再按2、3、…、9测试从首步连续干预的前缀，不假设效果单调；首个3/3失败前缀即为此类窗口的最小值，10步端点复用已确认结果。','',
        f'内部节点控制单步结果：{summary["singleton_minimum"]}；位置：{summary["confirmed_singleton_locations"]}；前缀控制：{scan_complete}。'
        '图像/物理patch的最少去噪步数：未确定。','',
        '| 条件 | 注入的去噪步（1起算） | 已完成 | 任务成功 | 配对失败 | 复用记录 |',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['completed']:lines.append(f'| {r["condition"]} | {r["steps"]} | {r["completed"]}/3 | {r["successes"]} | {r["failures"]}/3 | {r["reused"]} |')
    lines+=['','“稳定有效”只指本轮3个固定clean成功初态全部失败；同时保留逐初态结果。'
        '单步若均未达到3/3，只能报告测试过的前缀最小值，不能宣称全局最小组合已找到。'
        '从首步连续覆盖的结果也只描述此类人工干预下的恢复响应；不等于图像端可达性。'
        '内部强制干预不是物理patch ASR。', '',f'原始目录：`{root}`。错误：{errors}。']
    SCAN_REPORT.write_text('\n'.join(lines)+'\n')
    lines=['# ImageWAM：首步L0与完整去噪动作目标的patch对照','',f'状态：{status["stage"]}。','',
        '研究目标：固定patch实际放入RoboTwin场景，在真实SAPIEN图像输入下干扰ImageWAM的实际执行，以动作响应、任务成败及Random对照判断效果。'
        '当前仅place_container_plate一个任务；四阶段覆盖是训练采样范围，不要求各阶段都有效，也不声称跨任务通用。'
        '两张同一任务内的候选纹理：first_flow优化首步有效14维速度MSE；final_action通过完整10步反传，优化最终标准化16×14动作块MSE。'
        '模型权重冻结。均从灰色起步，10cm/Q128、500×8、lr0.01，按四阶段每批各2样本；共享同一抽样计划与原EOT标定。'
        '相同更新次数不等于相同计算量，分别记录梯度计算和墙钟时间。'
        '两张纹理都从图像端进入，final_action不是末步速度覆盖；patch闭环使用原生推理，没有人工内部速度注入。'
        f'解释说明：[输入端训练与内部控制]({SCOPE_NOTE})。','',
        '| 目标 | 更新 | 训练损失 | 梯度计算秒 |','|---|---:|---:|---:|']
    for name in OBJECTIVES:
        row=progress.get(name,{})
        lines.append(f'| {name} | {row.get("iteration",0)}/500 | {row.get("loss","未开始")} | {row.get("compute_s",0):.1f} |')
    lines+=['','| 条件 | 完成 | 任务成功 | 配对失败/clean成功 |','|---|---:|---:|---:|']
    for row in eval_rows:lines.append(f'| {row["condition"]} | {row["completed"]}/10 | {row["task_successes"]} | {row["failures"]}/{row["eligible"]} |')
    lines+=['','clean和Random全程基线透明复用第32号同初态/原生推理/阶段边界重规划的20条记录；'
        '新服务另完整复跑clean22000并逐值核对。两张新纹理各新增10回合，均为全程真实渲染曝光。'
        '这些初态已经使用过，属于探索性同条件比较。','',
        f'最终纹理验证：{len(validation)}/192；固定16验证状态、2份新noise，比较Random/两训练纹理的代理与真实渲染。','',
        '| 纹理 | 路径 | 条数 | 首步L0中位数 | 最终动作MSE中位数 | 关节RMSE/rad中位数 |',
        '|---|---|---:|---:|---:|---:|']
    for r in val_summary:lines.append(f'| {r["texture"]} | {r["path"]} | {r["n"]} | {r["first_L0"]:.6g} | {r["final_action_MSE"]:.6g} | {r["joint_action_RMSE_rad"]:.6g} |')
    lines+=['','能否将内部干预结论转化成物理patch效果，以真实闭环和同状态Random对照为准；不预设必定成功。','',
        f'原始目录：`{root}`。错误：{errors}。']
    PATCH_REPORT.write_text('\n'.join(lines)+'\n')
    return summary


def pipeline(a):
    p=prepare(a.root);lock=(a.root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logs=a.root/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',PYTHONHASHSEED='0')
    owned=[];trainer=None;last_notice=time.monotonic()
    def status(stage,**extra):
        dump(a.root/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),
            training_pid=trainer.pid if trainer else None,**extra))
        return report(a.root)
    def notify(kind,s):
        nonlocal last_notice
        if not a.thread:return
        msg=(f'ImageWAM内部节点控制与输入端patch对照：{kind}。请核对{a.root}/status.json和summary.json后中文汇报。'
            f'必须同时读取{a.root}/interpretation_review.json和{SCOPE_NOTE}；用户已纠正末步覆盖不能证明patch可达。'
            '目标：保留内部节点敏感性控制，比较首步L0与图像端完整10步动作目标的两张patch。'
            f'当前{s["status"]["stage"]}；内部控制记录{s["scan_completed_episodes"]}条（含9条复用），单节点控制{s["singleton_minimum"]}，前缀{s["scan_complete"]}；'
            f'patch更新{s["patch_updates"]}/1000、完成{s["patches_complete"]}/2；新增patch闭环{s["new_patch_episodes"]}/20；'
            f'真实/代理验证{s["validation_completed"]}/192；错误{s["errors"]}。'
            f'报告{SCAN_REPORT}和{PATCH_REPORT}；证据包{SEALED / "evidence.tar.gz"}。'
            '报告当前目标、分阶段计划进度、状态/错误及估计剩余时间，说明并行扫描/训练和基线复用。'
            '运行故障保留日志/checkpoint并尽量自行修复；机制或方法问题及时报告，不擅自改协议。')
        try:
            result=subprocess.run(['/home/user1/.local/bin/codex','queue','--thread',a.thread,'--message',msg],capture_output=True,text=True,timeout=30)
            dump(a.root/'last_notification.json',dict(kind=kind,time=time.time(),returncode=result.returncode,receipt=result.stdout,error=result.stderr))
        except Exception as error:dump(a.root/'last_notification.json',dict(kind=kind,time=time.time(),error=repr(error)))
        last_notice=time.monotonic()
    def check_training():
        if trainer is not None and trainer.poll() not in (None,0):raise RuntimeError('Patch trainer exited; see logs/patch_training.log')
    def launch(name,argv,gpu):
        with (logs/(name+'.log')).open('a') as log:
            proc=subprocess.Popen(argv,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=gpu),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        owned.append(proc);return proc
    def job(name,argv,gpu):
        status(name);proc=launch(name,argv,gpu)
        while proc.poll() is None:
            time.sleep(5);check_training();s=status(name,child_pid=proc.pid)
            if time.monotonic()-last_notice>=1800:notify('每30分钟进度',s)
        if proc.returncode:raise RuntimeError(f'{name} exited {proc.returncode}; see logs/{name}.log')
    def sim(mode,seed,**extra):
        args=['bash',str(ROOT/'robotwinprobe/run_sim.sh'),str(Path(__file__).resolve()),mode,'--root',str(a.root),'--seed',str(seed)]
        for key,value in extra.items():args.extend(['--'+key.replace('_','-'),str(value)])
        return args
    def group(mode):
        for seed in p['step_search']['seeds']:
            path=a.root/'episodes'/mode/f'seed{seed}.json'
            if path.exists():
                row=json.loads(path.read_text());assert 'success' in row
                continue
            job('scan_'+mode+'_'+str(seed),sim('scan_rollout',seed,condition=mode),a.sim_gpu)
        return all(not json.loads((a.root/'episodes'/mode/f'seed{seed}.json').read_text())['success'] for seed in p['step_search']['seeds'])
    try:
        with socket.socket() as probe:probe.bind(('127.0.0.1',PORT))
        while True:
            query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            free={line.split(',')[0].strip():int(line.split(',')[1]) for line in query.stdout.strip().splitlines()}
            if free[a.gpu]>=22000 and free[a.sim_gpu]>=5000:break
            s=status('waiting_for_gpu',free_MiB=free,model_gpu=a.gpu,sim_gpu=a.sim_gpu)
            if time.monotonic()-last_notice>=1800:notify('每30分钟进度',s)
            time.sleep(10)
        ready=a.root/'ready.json';ready.unlink(missing_ok=True);status('start_backend')
        server=launch('backend',[PY['wam'],'-u',str(Path(__file__).resolve()),'server','--root',str(a.root)],a.gpu)
        deadline=time.monotonic()+600
        while not ready.exists():
            if server.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Backend failed or timed out; inspect logs/backend.log')
            time.sleep(2);status('start_backend')
        assert CONTROL.call(a.root,'ping')['ready']
        job('native_clean_confirmation',sim('native_check',22000),a.sim_gpu)
        for index,step in enumerate(p['step_search']['single_order']):
            success=group(f'single_{step:02d}')
            if trainer is None:
                trainer=launch('patch_training',[PY['cpu'],'-u',str(ROOT/'robotwinprobe/wam_flow_patch.py'),'train','--root',str(a.root)],'')
            if success:break
        minimum_prefix=10
        for count in p['step_search']['prefix_order']:
            if group(f'prefix_{count:02d}'):
                minimum_prefix=count;break
        dump(a.root/'scan_complete.json',dict(complete=True,minimum_prefix=minimum_prefix,
            criterion='all three fixed clean-success seeds fail',global_minimum_known=any(r['reliable'] for r in condition_rows(a.root,p) if r['condition'].startswith('single_'))))
        notify('内部速度前缀控制完成',status('patch_training'))
        while trainer.poll() is None:
            time.sleep(10);s=status('patch_training')
            if time.monotonic()-last_notice>=1800:notify('每30分钟进度',s)
        check_training()
        assert (a.root/'patch_training_complete.json').exists()
        for seed in p['evaluation']['seeds']:
            for objective in OBJECTIVES:
                path=a.root/'episodes/wam'/objective/f'seed{seed}.json'
                if path.exists():assert 'success' in json.loads(path.read_text());continue
                job('patch_eval_'+objective+'_'+str(seed),['bash',str(ROOT/'robotwinprobe/run_sim.sh'),
                    str(ROOT/'robotwinprobe/wam_flow_patch.py'),'patch_rollout','--root',str(a.root),
                    '--objective',objective,'--seed',str(seed)],a.sim_gpu)
        for seed in sorted({r['seed'] for r in p['diagnosis']['records']}):
            job('capture_final_'+str(seed),['bash',str(ROOT/'robotwinprobe/run_sim.sh'),str(ROOT/'robotwinprobe/wam_flow_patch.py'),
                'capture','--root',str(a.root),'--seed',str(seed)],a.sim_gpu)
        job('validate_final',[PY['cpu'],'-u',str(ROOT/'robotwinprobe/wam_flow_patch.py'),'validate','--root',str(a.root)],'')
        s=status('complete')
        assert s['new_patch_episodes']==20 and s['validation_completed']==192 and s['patches_complete']==2
        notify('完成通知',s)
    except Exception as error:
        s=status('stopped_on_error',error=repr(error),traceback=traceback.format_exc());notify('即时异常告警',s)
        raise
    finally:
        for proc in reversed(owned):
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['pipeline','server','scan_rollout','native_check','report'])
    parser.add_argument('--root',type=Path,default=DEFAULT);parser.add_argument('--seed',type=int)
    parser.add_argument('--condition');parser.add_argument('--gpu',default='4');parser.add_argument('--sim-gpu',default='2')
    parser.add_argument('--thread');a=parser.parse_args()
    if a.mode=='pipeline':pipeline(a)
    elif a.mode=='server':CONTROL.serve(a.root)
    elif a.mode=='scan_rollout':CONTROL.rollout(a.root,a.condition,a.seed)
    elif a.mode=='native_check':
        native=a.root/'native_check';native.mkdir(exist_ok=True)
        if not (native/'protocol.json').exists():shutil.copy2(a.root/'protocol.json',native/'protocol.json')
        CONTROL.rollout(native,'clean',a.seed)
    else:report(a.root)
