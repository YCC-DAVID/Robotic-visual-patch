"""#59 follow-up: multi-task patch with equal per-task budget.

#59 multi-task used 500x8 shared by 4 tasks (1000 frame-forwards per task, 1/4 of
single-task). Here: 2000x8 (2/task per batch) = 4000 per task, same as each
single-task patch. Everything else identical (frames, seed, noise, lr, position 101).
Eval: clean + multi_task_2000 on the same 7 tasks, inits 40-49 (140 episodes).
"""
from concurrent.futures import ThreadPoolExecutor
import json, os, subprocess, sys
import multitask_pipeline as P

DEST = P.ROOT/'out/multitask_patch_2000'
CONDITIONS = ['clean', 'multi_task_2000']


def stage(name, **kw):
    P.dump(DEST/'progress.json', dict(stage=name, **kw)); print('[stage]', name, kw, flush=True)


def evaluate(task, gpu, port):
    folder = DEST/'eval'/task
    for attempt in range(3):
        out = folder/('rollout' if attempt == 0 else f'rollout_attempt{attempt}')
        if (out/'complete.json').exists(): return
        try:
            with P.Server(DEST/'services', gpu, port):
                with (DEST/'logs'/f'eval_{task}_{attempt}.log').open('w') as f:
                    subprocess.run([P.SIM, 'cross_task_transfer_rollout.py', '--manifest', str(DEST/'manifest.json'), '--task', task,
                                    '--conditions', ','.join(CONDITIONS), '--indices', ','.join(map(str, P.INDICES)), '--port', str(port),
                                    '--out', str(out.relative_to(P.ROOT/'out'))],
                                   cwd=P.ROOT, env=P.sim_env(gpu), stdout=f, stderr=subprocess.STDOUT, check=True, timeout=10800)
            P.dump(out/'complete.json', dict(complete=True, attempt=attempt)); return
        except subprocess.CalledProcessError as error:
            print('[retry]', task, attempt, error, flush=True)
    raise RuntimeError('evaluation failed 3 times: '+task)


def main():
    gpus = json.loads(sys.argv[1])
    for name in ['logs', 'train', 'eval', 'services']: (DEST/name).mkdir(parents=True, exist_ok=True)
    P.dump(DEST/'protocol.json', dict(experiment='#59 multi-task patch, equal per-task budget', tasks=P.TRAIN_TASKS,
        recipe='E26 both-camera, 2000x8 (2/task), seed 20261005, noise 20260901, gray init, no EOT; frames from #59 captures',
        eval_tasks=P.TRAIN_TASKS+P.HELDOUT_TASKS, conditions=CONDITIONS, indices=P.INDICES, gpus=gpus, host=os.uname().nodename))
    train = DEST/'train'
    if not (train/'results.json').exists():
        stage('train')
        with (DEST/'logs/train.log').open('w') as f:
            subprocess.run([P.MODEL, 'multitask_train.py', '--tasks-root', str(P.DEST/'tasks'), '--tasks', ','.join(P.TRAIN_TASKS),
                            '--iters', '2000', '--save-every', '100', '--out', str(train)], cwd=P.ROOT, stdout=f,
                           stderr=subprocess.STDOUT, check=True, timeout=21600,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpus[0]['uuid'], 'OMP_NUM_THREADS': '4'})
    tex = train/'texture_2000.png'
    P.dump(DEST/'manifest.json', dict(source_task='multi:'+'+'.join(P.TRAIN_TASKS), stage='multitask2000_eval',
        geometry=json.loads((P.ROOT/'out/dual_view/manifest_both/manifest.json').read_text())['geometry'],
        textures={'clean': None, 'multi_task_2000': str(tex)}, texture_sha256={'clean': None, 'multi_task_2000': P.sha(tex)}))
    tasks = P.TRAIN_TASKS+P.HELDOUT_TASKS
    stage('evaluate', episodes=len(tasks)*len(CONDITIONS)*10)
    with ThreadPoolExecutor(len(tasks)) as ex:
        for f in [ex.submit(evaluate, t, gpus[i % len(gpus)], 8361+i) for i, t in enumerate(tasks)]: f.result()
    stage('complete')


if __name__ == '__main__':
    main()
