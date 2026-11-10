"""Adopt the running SHM F arm and train a fresh U arm with validated SHM."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from vision_freeze_parallel import ParallelPipeline, command
from vision_freeze_pipeline import Pipeline
import checkpoint_study_common as S


class FastUPipeline(ParallelPipeline):
    def __init__(self, args):
        super().__init__(args)
        gate = json.loads((self.root/'shm_validation_20261027/validation_status.json').read_text())
        assert gate['complete'] and gate['stage'] == 'passed' and gate['paired_initial_passed']
        assert all(gate['restores'][v]['passed'] for v in ['F', 'U'])
        self.base['NCCL_SHM_DISABLE'] = '0'

    def run(self, name, argv, gpu='', jax=False):
        if name != 'fit_U':
            return super().run(name, argv, gpu, jax)
        # Verify the new step-zero pairing while U starts, rather than waiting
        # until all 10k updates have been spent on a potentially invalid pair.
        with (self.logs/(name+'.log')).open('a') as log:
            child = subprocess.Popen(argv, cwd=S.ROOT, env=self.env(gpu, jax),
                                     stdout=log, stderr=subprocess.STDOUT)
            S.dump(self.logs/(name+'.json'), dict(running=True, pid=child.pid, argv=argv, gpu=gpu))
            paired = False
            try:
                while child.poll() is None:
                    if not paired and (self.root/'training/U/initial_audit.json').exists():
                        self.paired_audit('')
                        paired = True
                    time.sleep(5)
                code = child.wait()
                assert code == 0, f'U exited {code}; inspect fit_U.log'
                if not paired:
                    self.paired_audit('')
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=60)
                raise
        S.dump(self.logs/(name+'.json'), dict(running=False, complete=True, returncode=0, argv=argv, gpu=gpu))

    def adopt_shm_f(self):
        run = self.root/'training/F'
        initial = (run/'initial_audit.json').read_bytes()
        pid = self.args.adopt_f_pid
        while True:
            status = json.loads((run/'status.json').read_text())
            if status.get('complete') and status.get('step') == self.model_steps:
                break
            cmd = command(pid)
            assert b'--name' in cmd and cmd[cmd.index(b'--name')+1] == b'F'
            assert str(self.root).encode() in cmd and any(a.endswith(b'vision_freeze_ft.py') for a in cmd)
            self.stage('adopted_shm_training', pid=pid, step=status.get('step'), target=self.model_steps)
            time.sleep(15)
        while any(a.endswith(b'vision_freeze_ft.py') for a in command(pid)):
            time.sleep(1)
        saved = json.loads((run/'latest_checkpoint.json').read_text())
        assert saved['step'] == self.model_steps and Path(saved['path']).is_dir()
        argv = [S.PY_TRAIN, 'robotwinprobe/vision_freeze_ft.py', '--root', str(self.root),
                '--name', 'F', '--seed', '202610230', '--steps', str(self.model_steps),
                '--resume', '--verify-only']
        self.stage('verifying_final_F', step=self.model_steps)
        self.wait_resources()
        self.run('restore_F', argv, ','.join(self.gpus), jax=True)
        audit = json.loads((run/'restore_audit.json').read_text())
        assert audit['passed'] and audit['step'] == self.model_steps
        assert (run/'initial_audit.json').read_bytes() == initial

    def train_models(self):
        def arm(variant):
            self.local.arm = variant
            self.local.gpus = (self.args.gpus if variant == 'F' else self.args.u_gpus).split(',')
            try:
                if variant == 'F':
                    self.adopt_shm_f()
                else:
                    Pipeline.fit(self, 'U')
                    audit = json.loads((self.root/'training/U/restore_audit.json').read_text())
                    assert audit['passed'] and audit['step'] == self.model_steps
                self.stage('verified_complete', target=self.model_steps)
            except Exception as exc:
                self.stage('failed', error=repr(exc), traceback=traceback.format_exc())
                raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(arm, v) for v in ['F', 'U']]
            for task in concurrent.futures.as_completed(pending):
                task.result()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpus', default='5,6')
    p.add_argument('--u-gpus', default='0,1')
    p.add_argument('--model-steps', type=int, default=10000)
    p.add_argument('--adopt-f-pid', type=int, required=True)
    p.add_argument('--train-only', action='store_true')
    args = p.parse_args()
    assert args.model_steps == 10000
    assert set(args.gpus.split(',')).isdisjoint(args.u_gpus.split(','))
    with (args.root/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        pipe = FastUPipeline(args)
        try:
            pipe.main()
        except Exception as exc:
            S.dump(args.root/'pipeline_status.json', dict(stage='failed', complete=False,
                pid=os.getpid(), error=repr(exc), traceback=traceback.format_exc(), arms=pipe.arms))
            pipe.report()
            raise
