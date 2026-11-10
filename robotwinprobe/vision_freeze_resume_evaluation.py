"""Resume verified vision-freeze study; never retrain or bypass pilot gates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

import checkpoint_study_common as S
from vision_freeze_pipeline import Pipeline

PREFLIGHT = '/home/user1/.codex/skills/gpu-preflight/scripts/check_gpus.py'


class ResumePipeline(Pipeline):
    def __init__(self, args):
        super().__init__(args)
        self.base['NCCL_SHM_DISABLE'] = '0'
        # Missing simulation dependencies on a migrated host are kept local
        # to this study rather than installed into a shared Python environment.
        deps = self.root/'python_deps'
        if deps.is_dir():
            self.base['PYTHONPATH'] = str(deps) + (
                os.pathsep + self.base['PYTHONPATH'] if self.base.get('PYTHONPATH') else '')
        self.host = socket.gethostname().split('.')[0]
        self.preflight_dir = self.root/'evaluation_resource_checks'
        self.preflight_dir.mkdir(exist_ok=True)

    def stage(self, name, **extra):
        return super().stage(name, execution_host=self.host, **extra)

    def check_resources(self, slots, all_hosts=False, _cooldown_attempt=0):
        cmd = [S.PY_CPU, PREFLIGHT, '--json']
        if not all_hosts:
            cmd += ['--hosts', self.host]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        data = json.loads(result.stdout)
        S.dump(self.preflight_dir/f'{time.time_ns()}.json', data)
        node = next(n for n in data['nodes'] if n['host'] == self.host)
        if node['status'] != 'ok':
            raise RuntimeError(f'GPU preflight unknown: {node}')
        selected = []
        for slot in slots:
            index = int(self.gpus[int(slot)])
            gpu = next(g for g in node['gpus'] if g['index'] == index)
            if not gpu['eligible']:
                # Utilization is a sampled window and can remain nonzero just
                # after our preceding child has exited. Re-query, never waive
                # the idle gate or wait through a real process allocation.
                if (not gpu['pids'] and gpu['used_mib'] <= 256
                        and gpu['utilization_pct'] > 0 and _cooldown_attempt < 3):
                    time.sleep(5)
                    return self.check_resources(slots, _cooldown_attempt=_cooldown_attempt+1)
                raise RuntimeError(f'GPU {index} is occupied; no task launched')
            selected.append(gpu['uuid'])
        return ','.join(selected)

    def fit(self, variant, smoke=False):
        # The finished training/restore evidence is mandatory, including smoke.
        name = ('preflight_v6_' if smoke else '') + variant
        run = self.root/'training'/name
        status = json.loads((run/'status.json').read_text())
        audit = json.loads((run/'restore_audit.json').read_text())
        assert status['complete'] and audit['passed']
        assert audit['step'] == (4 if smoke else self.model_steps)

    def run(self, name, argv, gpu='', jax=False):
        previous = self.logs/(name+'.json')
        if previous.exists():
            saved = json.loads(previous.read_text())
            if saved.get('complete') and saved.get('argv') == list(map(str, argv)):
                print('[reuse completed]', name, flush=True)
                return
        selected = self.check_resources(gpu.split(',')) if gpu else ''
        return super().run(name, argv, selected, jax)

    def server(self, model, gpu, port):
        return super().server(model, self.check_resources(gpu.split(',')), port)

    def evaluate_pair(self, model, pilot=False):
        if pilot:
            dest = self.root/'pilot'/model/'clean'
            saved = self.logs/f'pilot_{model}_clean.json'
            argv = ['bash', 'robotwinprobe/run_sim.sh',
                    'robotwinprobe/vision_freeze_rollout.py', '--root', str(self.root),
                    '--model', model, '--condition', 'clean', '--port', '8479', '--pilot']
            if saved.exists() and (dest/'results.json').exists() and (dest/'protocol.json').exists():
                job = json.loads(saved.read_text())
                result = json.loads((dest/'results.json').read_text())
                protocol = json.loads((dest/'protocol.json').read_text())
                episodes = result.get('episodes', [])
                seeds = list(range(23000, 23010))
                if (job.get('complete') and job.get('argv') == argv
                        and result.get('complete') and result.get('target') == 10
                        and [e.get('seed') for e in episodes] == seeds
                        and all(not e.get('error') and isinstance(e.get('success'), bool) for e in episodes)
                        and result.get('task_successes') == sum(e['success'] for e in episodes)
                        and protocol.get('model') == model and protocol.get('condition') == 'clean'
                        and protocol.get('seeds') == seeds
                        and protocol.get('checkpoint_sha256') == S.sha(self.root/'models'/model/'model.safetensors')
                        and protocol.get('source_sha256') == S.sha(S.ROOT/'robotwinprobe/vision_freeze_rollout.py')):
                    print('[reuse verified pilot; no server launch]', model, flush=True)
                    return
        # The parent main() still enforces both pilots' >=5/10 success gate.
        return super().evaluate_pair(model, pilot=pilot)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpus', default='0,1')
    args = p.parse_args()
    args.model_steps = 10000
    args.train_only = False
    with (args.root/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipe = ResumePipeline(args)
        try:
            pipe.check_resources(['0', '1'], all_hosts=True)
            pipe.main()
        except Exception as exc:
            S.dump(args.root/'pipeline_status.json', dict(
                stage='evaluation_resume_stopped', complete=False, pid=os.getpid(),
                error=repr(exc), traceback=traceback.format_exc()))
            pipe.report()
            raise
