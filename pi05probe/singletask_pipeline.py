"""Per-task single-task patches (control for #59 multi-task patch).

Same E26 recipe via multitask_train.py with one task (batch 8 from that task,
seed 20261005: identical sampling rule to E26 dual_view_train). Training frames
reuse #59 captures. Evaluate all four single-task patches on all 7 tasks.
"""
from concurrent.futures import ThreadPoolExecutor
import json, os, subprocess, sys
import multitask_pipeline as P

DEST = P.ROOT/'out/singletask_patch'
SHORT = {'put_the_bowl_on_the_plate': 'st_plate', 'put_the_bowl_on_top_of_the_cabinet': 'st_cabinet',
         'put_the_wine_bottle_on_the_rack': 'st_wine', 'turn_on_the_stove': 'st_stove'}
CONDITIONS = ['clean', 'random']+list(SHORT.values())


def stage(name, **kw):
    P.dump(DEST/'progress.json', dict(stage=name, **kw)); print('[stage]', name, kw, flush=True)


def train(task, gpu):
    out = DEST/'train'/SHORT[task]
    if (out/'results.json').exists(): return
    with (DEST/'logs'/f'train_{SHORT[task]}.log').open('w') as f:
        subprocess.run([P.MODEL, 'multitask_train.py', '--tasks-root', str(P.DEST/'tasks'), '--tasks', task, '--out', str(out)],
                       cwd=P.ROOT, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=14400,
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu['uuid'], 'OMP_NUM_THREADS': '4'})


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
    P.dump(DEST/'protocol.json', dict(experiment='single-task patches, control for #59', tasks=list(SHORT),
        recipe='E26 both-camera, 500x8, seed 20261005, noise 20260901, gray init, no EOT; frames from #59 captures',
        eval_tasks=P.TRAIN_TASKS+P.HELDOUT_TASKS, conditions=CONDITIONS, indices=P.INDICES, gpus=gpus,
        note='st_plate retrains E26 on the recaptured plate pool (parity check vs E26 5/9)'))
    stage('train')
    with ThreadPoolExecutor(4) as ex:
        for f in [ex.submit(train, t, gpus[i % len(gpus)]) for i, t in enumerate(SHORT)]: f.result()
    textures = {'clean': None, 'random': P.ROOT/'out/dual_view/manifest_both/random.png',
                **{s: DEST/'train'/s/'texture_0500.png' for s in SHORT.values()}}
    P.dump(DEST/'manifest.json', dict(source_task='single-task patches', stage='singletask_eval',
        geometry=json.loads((P.ROOT/'out/dual_view/manifest_both/manifest.json').read_text())['geometry'],
        textures={k: None if p is None else str(p) for k, p in textures.items()},
        texture_sha256={k: None if p is None else P.sha(p) for k, p in textures.items()}))
    tasks = P.TRAIN_TASKS+P.HELDOUT_TASKS
    stage('evaluate', episodes=len(tasks)*len(CONDITIONS)*10)
    with ThreadPoolExecutor(len(tasks)) as ex:
        for f in [ex.submit(evaluate, t, gpus[i % len(gpus)], 8341+i) for i, t in enumerate(tasks)]: f.result()
    stage('complete')


if __name__ == '__main__':
    main()
