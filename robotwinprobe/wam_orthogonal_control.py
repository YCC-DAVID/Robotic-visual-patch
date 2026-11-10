"""Small, isolated ImageWAM velocity intervention control; no patch optimization."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

import numpy as np

from crossarch_common import ROOT, PY, AUTH, CAMERAS, WIRE, dump, sha
from crossarch_stage_common import DEFAULT as SOURCE, PhaseTracker

DEFAULT = ROOT / 'robotwinprobe/out/wam_orthogonal_control_20261027'
CONDITIONS = ('clean', 'orthogonal_first', 'orthogonal_all')
PORT = 18562
REPORT = ROOT / 'result/34_imagewam_orthogonal_control.md'


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    original = json.loads((SOURCE / 'protocol.json').read_text())
    p = copy.deepcopy(original)
    p.update(experiment='imagewam_orthogonal_velocity_control', source_run=str(SOURCE),
             source_protocol_sha256=sha(SOURCE / 'protocol.json'))
    p['evaluation'].update(seeds=[22000, 22001, 22002], conditions=list(CONDITIONS),
        scope='three preselected initial seeds, exploratory internal intervention, no physical patch',
        intervention='At EVERY policy query, rotate the full 16x14 predicted velocity to an equal-norm random orthogonal vector; first denoising step only versus all ten steps.',
        orthogonal_reference='unmodified local velocity prediction at the same action latent and timestep',
        noise_seed='independent reproducible Gaussian direction from 20261027 + 17*action_noise_seed + flow_step',
        native_control='same public ImageWAM weights, native preprocessing and ten-step scheduler; clean full trajectories must reproduce the source run exactly')
    p['control_port'] = PORT
    path = root / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == p, 'Preserve the existing control protocol'
    else:
        dump(path, p)
    for name in ('prompts.npz', 'provenance.json'):
        if not (root / name).exists():
            os.link(SOURCE / name, root / name)
    return p


def rotate_orthogonal(value, seed):
    """Equal-norm v_new perpendicular to v, rather than just delta perpendicular."""
    v = np.asarray(value, dtype=np.float64)
    flat = v.reshape(-1)
    squared = float(flat @ flat)
    if squared <= 1e-20:
        raise ValueError('Zero velocity has no well-defined orthogonal direction')
    direction = np.random.default_rng(seed).standard_normal(flat.size)
    direction -= float(direction @ flat) / squared * flat
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise ValueError('Degenerate random orthogonal direction')
    return (direction * np.sqrt(squared) / length).reshape(v.shape).astype(np.float32)


def intervention_masks(protocol):
    masks=protocol.get('intervention_masks',dict(clean=[],orthogonal_first=[0],orthogonal_all=list(range(10))))
    assert all(steps==sorted(set(steps)) and all(isinstance(i,int) and 0<=i<10 for i in steps)
               for steps in masks.values()), masks
    return masks


def call(root, command, **data):
    from multiprocessing.connection import Client
    port = json.loads((root / 'protocol.json').read_text())['control_port']
    with Client(('127.0.0.1', port), authkey=AUTH) as conn:
        conn.send(dict(command=command, **data))
        result = conn.recv()
    if 'error' in result:
        raise RuntimeError(result['error'] + '\n' + result.get('traceback', ''))
    return result


def serve(root):
    import torch
    from multiprocessing.connection import Listener
    from crossarch_backend import Backend
    backend = Backend('wam', root)
    backend.audit()
    protocol=json.loads((root/'protocol.json').read_text())
    masks=intervention_masks(protocol)
    teacher_cache={}

    def infer(req):
        mode = req['condition']
        assert mode in masks
        active=masks[mode]
        original = backend.model.action_expert.post_dit

        def forward(intervene):
            before, after = [], []

            def hooked(*args, **kwargs):
                value = original(*args, **kwargs)
                index = len(before)
                old = value.detach().float().cpu().numpy()
                new = value
                if intervene and index in active:
                    direction_seed = 20261027 + 17 * int(req['seed']) + index
                    rotated = rotate_orthogonal(old, direction_seed)
                    new = torch.from_numpy(rotated).to(device=value.device, dtype=value.dtype)
                before.append(old)
                after.append(new.detach().float().cpu().numpy())
                return new

            backend.model.action_expert.post_dit = hooked
            try:
                result = backend.infer(req)
            finally:
                backend.model.action_expert.post_dit = original
            assert len(before) == 10, len(before)
            return result['actions'], np.stack(before), np.stack(after)

        with torch.no_grad():
            clean_action, clean_velocity, _ = forward(False)
            if mode == 'clean':
                action, before, after = clean_action, clean_velocity, clean_velocity
            else:
                action, before, after = forward(True)
        # First prediction is bit-exact before intervention; no policy RNG was changed.
        assert np.array_equal(before[0], clean_velocity[0])
        old, new = before.reshape(10, -1).astype(float), after.reshape(10, -1).astype(float)
        norms_old, norms_new = np.linalg.norm(old, axis=1), np.linalg.norm(new, axis=1)
        cosines = (old * new).sum(1) / (norms_old * norms_new)
        ratios = norms_new / norms_old
        if active:
            assert np.max(np.abs(cosines[active])) < .005, cosines
            assert np.max(np.abs(ratios[active] - 1)) < .005, ratios
        l0 = float(np.mean((after[0] - clean_velocity[0]) ** 2))
        return dict(actions=action, first_L0=l0,
            first_relative_L0=l0 / float(np.mean(clean_velocity[0] ** 2)),
            cosine=cosines.tolist(), norm_ratio=ratios.tolist(), modified_steps=active,
            final_action_MSE=float(np.mean((action - clean_action) ** 2)),
            joint_action_RMSE_rad=float(np.sqrt(np.mean((action[:, [0,1,2,3,4,5,7,8,9,10,11,12]] -
                                                        clean_action[:, [0,1,2,3,4,5,7,8,9,10,11,12]]) ** 2))),
            velocity_before=before, velocity_after=after)

    port=protocol.get('control_port',PORT)
    with Listener(('127.0.0.1', port), authkey=AUTH) as listener:
        dump(root / 'ready.json', dict(pid=os.getpid(), port=port, ready=True))
        while True:
            with listener.accept() as conn:
                try:
                    req = conn.recv()
                    command=req['command']
                    if command=='ping':result=dict(ready=True)
                    elif command=='infer':result=infer(req)
                    elif command=='infer_image':result=backend.infer(req)
                    elif command in ('patch_gradient','patch_diagnostic'):
                        from wam_flow_patch import gradient,diagnostic
                        result=(gradient if command=='patch_gradient' else diagnostic)(backend,req,teacher_cache)
                    else:raise ValueError('Unknown command: '+command)
                    conn.send(result)
                except Exception as error:
                    conn.send(dict(error=repr(error), traceback=traceback.format_exc()))
                    traceback.print_exc()


def rollout(root, mode, seed):
    import bootstrap as B
    B.simulator_paths()
    from scene_smoke import make_task
    from crossarch_stage_rollout import digest
    p = json.loads((root / 'protocol.json').read_text())
    folder = root / 'episodes' / mode
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f'seed{seed}.json'
    protocol_hash = sha(root / 'protocol.json')
    if output.exists():
        prior = json.loads(output.read_text())
        assert 'success' in prior and prior['protocol_sha256'] == protocol_hash
        return
    source = SOURCE / 'episodes/wam/clean' / f'seed{seed}.json'
    original_clean = json.loads(source.read_text())
    with np.load(source.with_suffix('.npz')) as z:
        reference = {k: z[k].copy() for k in ('states14', 'actions', 'query_steps', 'rgb_sha256', 'object_plate')}
    task = None
    began = time.monotonic()
    try:
        task, _, _ = make_task(p['task'], seed, folder)
        prompt = 'Place the ' + ('bowl' if task.actor_name == '002_bowl' else 'cup') + ' on the plate.'
        task.set_instruction(prompt)
        initial = task.get_obs()
        initial_state = initial['joint_action']['vector'].astype(np.float32)
        initial_hash = digest(np.concatenate([initial_state.view(np.uint8).ravel()] +
            [initial['observation'][n]['rgb'].ravel() for n in CAMERAS]))
        assert initial_hash == original_clean['initial_clean_sha256']
        tracker = PhaseTracker(task.container.get_pose().p)
        states, actions, queries, phases, diagnostics, velocities = [], [], [], [], [], []
        while task.take_action_cnt < task.step_lim and not task.eval_success:
            obs = task.get_obs()
            state = obs['joint_action']['vector'].astype(np.float32)
            q = len(states)
            if mode == 'clean':
                assert np.array_equal(state, reference['states14'][q]), ('clean state', seed, q)
                assert task.take_action_cnt == reference['query_steps'][q]
                assert [digest(obs['observation'][n]['rgb']) for n in CAMERAS] == reference['rgb_sha256'][q].tolist()
                assert np.array_equal(np.asarray([task.container.get_pose().p,task.plate.get_pose().p]), reference['object_plate'][q])
            result = call(root, 'infer', condition=mode, seed=20261006 + seed * 1000 + q,
                state=state, prompt=prompt, images={w:obs['observation'][n]['rgb'] for n,w in zip(CAMERAS,WIRE)})
            action = np.asarray(result.pop('actions'))
            assert action.shape == (16,14) and np.isfinite(action).all()
            if mode == 'clean':
                assert np.array_equal(action, reference['actions'][q]), ('clean actions', seed, q)
            velocities.append([result.pop('velocity_before'), result.pop('velocity_after')])
            diagnostics.append(result)
            states.append(state);actions.append(action);queries.append(task.take_action_cnt);phases.append(tracker.phase)
            for command in action:
                task.take_action(command, action_type='qpos')
                current = np.asarray(task.robot.get_left_arm_jointState() + task.robot.get_right_arm_jointState())
                boundary = tracker.update(current, task.container.get_pose().p, task.plate.get_pose().p, task.take_action_cnt)
                if boundary or task.eval_success or task.take_action_cnt >= task.step_lim:
                    break
        if mode == 'clean':
            assert bool(task.eval_success) == original_clean['success'] and len(states) == len(reference['states14'])
        np.savez_compressed(output.with_suffix('.npz'), states14=states, actions=actions,
            query_steps=queries, query_phases=phases, velocities=velocities)
        dump(output, dict(seed=seed, condition=mode, success=bool(task.eval_success),
            steps=task.take_action_cnt, queries=len(states), protocol_sha256=protocol_hash,
            initial_clean_sha256=initial_hash, diagnostics=diagnostics,
            clean_source_reproduced=mode=='clean', elapsed_s=time.monotonic()-began))
        print('[orthogonal rollout]', mode, seed, bool(task.eval_success), flush=True)
    except Exception as error:
        dump(output, dict(seed=seed, condition=mode, error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        if task is not None:
            task.close_env()


def report(root):
    p = json.loads((root / 'protocol.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    data = {mode:{r['seed']:r for f in (root/'episodes'/mode).glob('seed*.json')
                  if 'success' in (r:=json.loads(f.read_text()))} for mode in CONDITIONS}
    clean = data['clean']
    rows=[]
    for mode, records in data.items():
        eligible=[s for s in records if s in clean and clean[s]['success']]
        metrics=[d for r in records.values() for d in r['diagnostics']]
        rows.append(dict(condition=mode, completed=len(records), successes=sum(r['success'] for r in records.values()),
            clean_eligible=len(eligible), failures=sum(not records[s]['success'] for s in eligible),
            mean_first_L0=float(np.mean([d['first_L0'] for d in metrics])) if metrics else None,
            mean_first_cosine=float(np.mean([d['cosine'][0] for d in metrics])) if metrics else None,
            mean_first_relative_L0=float(np.mean([d['first_relative_L0'] for d in metrics])) if metrics else None))
    errors=[dict(path=str(f),error=r['error']) for f in (root/'episodes').glob('*/*.json')
            if 'error' in (r:=json.loads(f.read_text()))]
    summary=dict(status=status,rows=rows,completed=sum(len(r) for r in data.values()),
        expected=len(CONDITIONS)*len(p['evaluation']['seeds']),errors=errors)
    dump(root/'summary.json',summary)
    lines=['# ImageWAM：强制正交速度干预', '',
        f'状态：{status["stage"]}；闭环{summary["completed"]}/{summary["expected"]}。', '',
        '同一公开RoboTwin权重；固定初态22000–22002。每次重规划干预16×14完整速度场，保持L2范数并旋转90°。'
        '比较仅第一个去噪步与全部10个去噪步；不是只改任务第一次查询。', '',
        '画面保持clean；clean组完整轨迹、状态、相机图像和动作须与第32号实验逐值复现。'
        '正交方向由独立固定高斯噪声投影构造，模型原本的动作起始noise保持配对。', '',
        '| 条件 | 完成 | 任务成功 | 配对失败/clean成功 | 首步L0均值 | 首步余弦均值 | 相对L0均值 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values=[f'{row[k]:.6g}' if row[k] is not None else '待测'
                for k in ('mean_first_L0','mean_first_cosine','mean_first_relative_L0')]
        lines.append(f'| {row["condition"]} | {row["completed"]}/3 | {row["successes"]} | '
            f'{row["failures"]}/{row["clean_eligible"]} | '+ ' | '.join(values)+' |')
    lines+=['', 'BF16量化后逐步检查|cos|<0.005、范数比误差<0.005；等长正交使首步L0约为clean速度能量的2倍。', '',
        '这是直接修改模型内部速度的机制控制，不代表图像/物理patch能产生同样扰动。'
        '只有3个固定初态，不能据此估计普遍攻击成功率。不同条件访问不同状态，轨迹平均L0不是同状态因果比较。', '',
        f'原始数据：`{root}`。错误：{errors}。']
    REPORT.write_text('\n'.join(lines)+'\n')
    return summary


def pipeline(a):
    p=prepare(a.root)
    lock=(a.root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logs=a.root/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',PYTHONHASHSEED='0')
    owned=[];last_notice=time.monotonic()
    def status(stage, **extra):
        dump(a.root/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**extra))
        return report(a.root)
    def launch(name,argv,gpu):
        with (logs/(name+'.log')).open('a') as log:
            proc=subprocess.Popen(argv,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=gpu),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        owned.append(proc);return proc
    def notify(kind,summary):
        nonlocal last_notice
        if not a.thread:return
        message=(f'ImageWAM正交速度机制控制：{kind}。请核对{a.root}/status.json与summary.json后汇报。'
            f'目标：等长正交速度，比较每次查询仅首步/全部去噪步干预；固定3初态共9闭环。'
            f'当前状态{summary["status"]["stage"]}，完成{summary["completed"]}/9，错误{summary["errors"]}。'
            f'报告{REPORT}。此为用户已授权的内部干预测试，不是patch ASR。'
            '请报告目标、计划进度、状态/错误、剩余时间与配对结果；运行故障保留证据并尽量自行修复，机制/方法问题报告，不擅自改协议。')
        result=subprocess.run(['/home/user1/.local/bin/codex','queue','--thread',a.thread,'--message',message],
            capture_output=True,text=True,timeout=30)
        dump(a.root/'last_notification.json',dict(time=time.time(),kind=kind,returncode=result.returncode,
            receipt=result.stdout.strip(),error=result.stderr.strip()))
        last_notice=time.monotonic()
    try:
        with socket.socket() as probe:probe.bind(('127.0.0.1',PORT))
        ready=a.root/'ready.json';ready.unlink(missing_ok=True)
        status('start_backend')
        backend=launch('backend',[PY['wam'],'-u',str(Path(__file__).resolve()),'server','--root',str(a.root)],a.gpu)
        deadline=time.monotonic()+600
        while not ready.exists():
            if backend.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Backend startup failed; inspect logs/backend.log')
            time.sleep(2);status('start_backend')
        assert call(a.root,'ping')['ready']
        for seed in p['evaluation']['seeds']:
            for mode in CONDITIONS:
                phase=f'eval_{mode}_{seed}';status(phase)
                proc=launch(phase,['bash',str(ROOT/'robotwinprobe/run_sim.sh'),str(Path(__file__).resolve()),
                    'rollout','--root',str(a.root),'--condition',mode,'--seed',str(seed)],a.sim_gpu)
                while proc.poll() is None:
                    time.sleep(5);summary=status(phase,child_pid=proc.pid)
                    if time.monotonic()-last_notice>=1800:notify('每30分钟进度',summary)
                if proc.returncode:raise RuntimeError(f'{phase} exited {proc.returncode}; inspect logs/{phase}.log')
        notify('完成通知',status('complete'))
    except Exception as error:
        summary=status('stopped_on_error',error=repr(error),traceback=traceback.format_exc())
        notify('即时异常告警',summary)
        raise
    finally:
        for proc in reversed(owned):
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['server','rollout','pipeline','report'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--condition',choices=CONDITIONS,default='clean')
    parser.add_argument('--seed',type=int,default=22000)
    parser.add_argument('--gpu',default='4');parser.add_argument('--sim-gpu',default='2')
    parser.add_argument('--thread')
    args=parser.parse_args()
    if args.mode=='server':serve(args.root)
    elif args.mode=='rollout':rollout(args.root,args.condition,args.seed)
    elif args.mode=='report':report(args.root)
    else:pipeline(args)
