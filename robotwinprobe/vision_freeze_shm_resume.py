"""Adopt both live arms and switch SHM only after the paired 5000-step boundary."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import traceback
from vision_freeze_parallel import ParallelPipeline,command
from vision_freeze_pipeline import Pipeline
import checkpoint_study_common as S


def owns(pid,root,variant):
    args=command(pid)
    return (any(a.endswith(b'vision_freeze_ft.py') for a in args) and str(root).encode() in args
        and b'--name' in args and args[args.index(b'--name')+1]==variant.encode())


def await_saved(root,variant,pid,step,progress):
    run=root/'training'/variant
    while True:
        path=run/'latest_checkpoint.json'
        saved=json.loads(path.read_text()) if path.exists() else {}
        if saved.get('step',0)>=step:break
        assert owns(pid,root,variant),f'{variant} exited before saved switch boundary'
        info=json.loads((run/'status.json').read_text())
        progress('adopted_training_before_shm',pid=pid,step=info.get('step'),switch_at=step,target=10000)
        time.sleep(15)
    assert saved['step']==step,'Refuse an unexpected checkpoint boundary'
    reference=run/f'checkpoint_reference_{step}.json'
    assert Path(saved['path']).is_dir() and reference.exists()
    assert owns(pid,root,variant),'Refuse to terminate an unrelated process'
    S.dump(run/'shm_boundary_stop.json',dict(pid=pid,step=step,checkpoint=saved['path'],
        reference_sha256=S.sha(reference),reason='Completed save confirmed before SHM transition'))
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+60
    while owns(pid,root,variant) and time.monotonic()<deadline:time.sleep(1)
    assert not owns(pid,root,variant),'Old training still alive; no duplicate launched'


class SwitchPipeline(ParallelPipeline):
    def __init__(self,args):
        super().__init__(args)
        gate=self.root/'shm_validation_20261027/validation_status.json'
        audit=json.loads(gate.read_text())
        assert audit['complete'] and audit['stage']=='passed' and audit['paired_initial_passed']
        assert all(audit['restores'][v]['passed'] for v in ['F','U'])
        self.gate_hash=S.sha(gate)
        # Same algorithm, microbatch, noise and LR schedule in both arms.
        self.base['NCCL_SHM_DISABLE']='0'

    def switch_arm(self,variant,pid):
        run=self.root/'training'/variant;initial=(run/'initial_audit.json').read_bytes()
        plan=dict(variant=variant,source_pid=pid,switch_step=self.args.switch_step,target=self.model_steps,
            validation_sha256=self.gate_hash,old_shm_disable='1',new_shm_disable='0',micro_batch=2,
            production_switched=False,stage='waiting_for_saved_boundary')
        S.dump(run/'shm_transition.json',plan)
        await_saved(self.root,variant,pid,self.args.switch_step,self.stage)
        argv=[S.PY_TRAIN,'robotwinprobe/vision_freeze_ft.py','--root',str(self.root),'--name',variant,
            '--seed','202610230','--steps',str(self.args.switch_step),'--resume']
        if variant=='U':argv+=['--unfreeze-vision']
        try:
            # A zero-update invocation finalizes the boundary's vision audit;
            # its independent restore checks the original saved outputs under SHM.
            self.stage('verifying_shm_resume',step=self.args.switch_step)
            S.dump(run/'shm_transition.json',plan|dict(stage='verifying_resume'))
            self.run('shm_finalize_'+variant,argv,','.join(self.gpus),jax=True)
            self.run('shm_restore_'+variant,argv+['--verify-only'],','.join(self.gpus),jax=True)
            audit=json.loads((run/'restore_audit.json').read_text())
            assert audit['passed'] and audit['step']==self.args.switch_step
            for name in ['restore_audit','restore_comparison','vision_update_audit']:
                path=run/(name+'.json')
                (run/f'{name}_shm_{self.args.switch_step}.json').write_bytes(path.read_bytes())
            # Keep the step-zero pairing reference, even though the trainer
            # writes its current resume state to initial_audit.json.
            (run/'initial_audit_shm_resume.json').write_bytes((run/'initial_audit.json').read_bytes())
            (run/'initial_audit.json').write_bytes(initial)
            S.dump(run/'status.json',dict(stage='resume_pending',complete=False,step=self.args.switch_step,target=self.model_steps))
            S.dump(run/'shm_transition.json',plan|dict(stage='resume_verified',production_switched=True,
                restore_audit_sha256=S.sha(run/'restore_audit.json')))
            Pipeline.fit(self,variant)
            S.dump(run/'shm_transition.json',plan|dict(stage='complete',production_switched=True,complete=True))
        finally:
            current=run/'initial_audit.json'
            if current.exists() and current.read_bytes()!=initial:
                (run/'initial_audit_after_shm_resume.json').write_bytes(current.read_bytes())
                current.write_bytes(initial)

    def train_models(self):
        def arm(variant):
            self.local.arm=variant
            self.local.gpus=(self.args.gpus if variant=='F' else self.args.u_gpus).split(',')
            pid=self.args.adopt_f_pid if variant=='F' else self.args.adopt_u_pid
            try:
                self.switch_arm(variant,pid)
                self.stage('verified_complete',target=self.model_steps)
            except Exception as exc:
                self.stage('failed',error=repr(exc),traceback=traceback.format_exc());raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending=[pool.submit(arm,v) for v in ['F','U']]
            for task in concurrent.futures.as_completed(pending):task.result()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpus',default='5,6');p.add_argument('--u-gpus',default='0,1')
    p.add_argument('--model-steps',type=int,default=10000);p.add_argument('--switch-step',type=int,default=5000)
    p.add_argument('--adopt-f-pid',type=int,required=True);p.add_argument('--adopt-u-pid',type=int,required=True)
    p.add_argument('--train-only',action='store_true');args=p.parse_args()
    assert args.switch_step==5000 and args.model_steps==10000
    assert set(args.gpus.split(',')).isdisjoint(args.u_gpus.split(','))
    with (args.root/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pipe=SwitchPipeline(args)
        try:pipe.main()
        except Exception as exc:
            S.dump(args.root/'pipeline_status.json',dict(stage='failed',complete=False,pid=os.getpid(),
                error=repr(exc),traceback=traceback.format_exc(),arms=pipe.arms));pipe.report();raise
