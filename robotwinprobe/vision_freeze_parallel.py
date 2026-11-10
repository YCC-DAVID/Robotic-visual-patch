"""Adopt the live frozen arm, train U separately, join before paired evaluation."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import time
import traceback
from vision_freeze_pipeline import Pipeline
import checkpoint_study_common as S


def command(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:return []


def is_adopted_training(pid,root):
    args=command(pid)
    return (any(a.endswith(b'vision_freeze_ft.py') for a in args)
        and str(root).encode() in args and b'--name' in args
        and args[args.index(b'--name')+1]==b'F')


class ParallelPipeline(Pipeline):
    def __init__(self,args):
        self.local=threading.local();self.mutex=threading.RLock();self.arms={}
        super().__init__(args)

    @property
    def gpus(self):return getattr(self.local,'gpus',self.default_gpus)
    @gpus.setter
    def gpus(self,value):self.default_gpus=value

    def stage(self,name,**extra):
        with self.mutex:
            arm=getattr(self.local,'arm',None)
            if arm:
                self.arms[arm]=dict(stage=name,gpus=self.gpus,**extra)
                S.dump(self.root/'pipeline_status.json',dict(stage='parallel_model_training',complete=False,
                    pid=os.getpid(),updated_unix=time.time(),target_steps=self.model_steps,arms=self.arms))
                self.report()
            else:super().stage(name,**extra)

    def wait_probe(self):
        path=self.root/'speed_probe_20261027/suite_status.json'
        process=self.root/'speed_probe_20261027/process.json'
        pid=json.loads(process.read_text())['pid']
        while any(a.endswith(b'vision_freeze_speed_suite.py') for a in command(pid)):
            info=json.loads(path.read_text()) if path.exists() else {}
            self.stage('waiting_for_speed_probe',probe_case=info.get('case'),probe_pid=pid)
            time.sleep(30)
        info=json.loads(path.read_text())
        assert info.get('complete'),'Speed suite exited unexpectedly; inspect before using GPUs'
        # Timing results alone never select a new production training recipe.

    def adopt_f(self):
        pid=self.args.adopt_f_pid;run=self.root/'training/F'
        initial=(run/'initial_audit.json').read_bytes()
        before=json.loads((run/'config.json').read_text())
        assert before['seed']==202610230 and before['steps']>=self.model_steps
        assert self.model_steps%5000==0,'Adopted process only saves every 5000 steps'
        while True:
            latest=run/'latest_checkpoint.json'
            saved=json.loads(latest.read_text()) if latest.exists() else {}
            if saved.get('step',0)>=self.model_steps:break
            assert is_adopted_training(pid,self.root),'Adopted F exited before target checkpoint'
            status=json.loads((run/'status.json').read_text())
            if (self.root/'training/U/initial_audit.json').exists() and not (self.root/'paired_audit.json').exists():
                self.paired_audit('')
            self.stage('adopted_training',pid=pid,step=status.get('step'),
                target=self.model_steps,original_process_target=before['steps'])
            time.sleep(15)
        assert saved['step']==self.model_steps,'Unexpected checkpoint: refuse to silently select a different step'
        ck=Path(saved['path']);reference=run/f'checkpoint_reference_{self.model_steps}.json'
        assert ck.is_dir() and reference.exists()
        assert is_adopted_training(pid,self.root),'Verify process identity before stopping adopted F'
        # latest_checkpoint is emitted only after manager.wait_until_finished().
        S.dump(run/'budget_stop_audit.json',dict(pid=pid,step=self.model_steps,checkpoint=str(ck),
            reference_sha256=S.sha(reference),initial_audit_sha256=S.sha(run/'initial_audit.json'),
            reason='User reduced both arms to 10000; target checkpoint fully saved'))
        os.kill(pid,signal.SIGTERM)
        deadline=time.monotonic()+60
        while is_adopted_training(pid,self.root) and time.monotonic()<deadline:time.sleep(1)
        assert not is_adopted_training(pid,self.root),'F did not exit; refusing duplicate training'
        # Run zero additional updates from the saved target to finalize audits.
        # The trainer records its resume-time initial audit, so restore the true
        # step-zero pairing evidence after finalization, without losing either.
        try:super().fit('F')
        finally:
            current=run/'initial_audit.json'
            if current.exists() and current.read_bytes()!=initial:
                (run/'resume_initial_audit.json').write_bytes(current.read_bytes())
                current.write_bytes(initial)

    def train_models(self):
        def arm(variant):
            self.local.arm=variant
            self.local.gpus=(self.args.gpus if variant=='F' else self.args.u_gpus).split(',')
            try:
                if variant=='F':self.adopt_f()
                else:
                    self.wait_probe()
                    super(ParallelPipeline,self).fit('U')
                self.stage('verified_complete',target=self.model_steps)
            except Exception as exc:
                self.stage('failed',error=repr(exc),traceback=traceback.format_exc())
                raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending=[pool.submit(arm,v) for v in ['F','U']]
            for task in concurrent.futures.as_completed(pending):task.result()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpus',default='5,6');p.add_argument('--u-gpus',default='0,1')
    p.add_argument('--model-steps',type=int,default=10000);p.add_argument('--adopt-f-pid',type=int,required=True)
    p.add_argument('--train-only',action='store_true');args=p.parse_args()
    assert set(args.gpus.split(',')).isdisjoint(args.u_gpus.split(','))
    with (args.root/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pipe=ParallelPipeline(args)
        try:pipe.main()
        except Exception as exc:
            S.dump(args.root/'pipeline_status.json',dict(stage='failed',complete=False,pid=os.getpid(),
                error=repr(exc),traceback=traceback.format_exc(),arms=pipe.arms))
            pipe.report();raise
