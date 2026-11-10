"""Re-run init49 (all four conditions, incl. paired clean) for tasks whose rollout
stopped on the per-episode RGB repeat self-check. Separate output folder; the
partial original episodes.json is kept untouched."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import multitask_pipeline as P

TASKS = ['put_the_bowl_on_top_of_the_cabinet', 'turn_on_the_stove']


def run(task, gpu, port):
    folder = P.DEST/'eval'/task
    with P.Server(P.DEST/'services'/'retry49', gpu, port):
        with (P.DEST/'logs'/f'retry49_{task}.log').open('w') as f:
            subprocess.run([P.SIM, 'cross_task_transfer_rollout.py', '--manifest', str(P.DEST/'manifest.json'), '--task', task,
                            '--conditions', ','.join(P.CONDITIONS), '--indices', '49', '--port', str(port),
                            '--out', str((folder/'rollout_retry49').relative_to(P.ROOT/'out'))],
                           cwd=P.ROOT, env=P.sim_env(gpu), stdout=f, stderr=subprocess.STDOUT, check=True, timeout=3600)


if __name__ == '__main__':
    gpus = json.loads(sys.argv[1])
    with ThreadPoolExecutor(2) as ex:
        for f in [ex.submit(run, t, gpus[i], 8331+i) for i, t in enumerate(TASKS)]: f.result()
    print('[retry complete]', flush=True)
