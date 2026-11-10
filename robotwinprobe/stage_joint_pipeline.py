"""Owned durable execution of the approved three-patch, at-most2400-record study."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from stage_joint_common import DEFAULT,ROOT,PY,PORT,load,prepare,request,dump,sha
from stage_joint_report import report,audit

PLOT_PY='/home/user1/miniconda3/envs/openpi-libero/bin/python'


def matching_processes(script,root):
    result=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=path.read_bytes().decode().split('\0')
            if any(Path(a).name==script for a in args) and str(root) in args:
                result.append(int(path.parent.name))
        except (OSError,UnicodeError):pass
    return result


def alive(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        return stat.split(') ')[1][0]!='Z'
    except OSError:return False


class Pipeline:
    def __init__(self,a):
        self.a=a;self.root=a.root;self.children=[];self.adopted=[]
        self.logs=self.root/'logs';self.logs.mkdir(exist_ok=True)
        self.env=dict(os.environ,HF_HUB_OFFLINE='1',JAX_PLATFORMS='cpu',
            XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            PYTHONHASHSEED='0',MPLCONFIGDIR=str(self.root/'mplcache'),
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        self.env.pop('PYTHONPATH',None)
        signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))

    def status(self,stage,**extra):
        dump(self.root/'status.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**extra))
        report(self.root)

    def launch(self,label,argv,gpu=''):
        log=(self.logs/(label+'.log')).open('a')
        proc=subprocess.Popen(argv,cwd=ROOT,env=dict(self.env,CUDA_VISIBLE_DEVICES=str(gpu)),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close();self.children.append(proc)
        return proc

    def sim(self,script,*args):
        return ['bash',str(ROOT/'robotwinprobe/run_sim.sh'),str(ROOT/'robotwinprobe'/script),
            '--root',str(self.root),*map(str,args)]

    def model(self,script,*args):
        return [PY,'-u',str(ROOT/'robotwinprobe'/script),'--root',str(self.root),*map(str,args)]

    def plot(self,script,*args):
        return [PLOT_PY,'-u',str(ROOT/'robotwinprobe'/script),'--root',str(self.root),*map(str,args)]

    def call(self,label,argv,gpu=''):
        self.status(label);proc=self.launch(label,argv,gpu)
        while proc.poll() is None:
            time.sleep(10);self.status(label,child_pid=proc.pid)
        if proc.returncode:raise RuntimeError(f'{label} exited {proc.returncode}; see logs/{label}.log')

    def backend(self):
        existing=matching_processes('stage_joint_backend.py',self.root)
        if existing:
            assert len(existing)==1,existing
            self.adopted.append((existing[0],'stage_joint_backend.py'))
            self.status('adopting_backend',backend_pid=existing[0])
            deadline=time.monotonic()+600
            while not (self.root/'backend_ready.json').exists():
                if not alive(existing[0]):raise RuntimeError('Adopted backend exited')
                if time.monotonic()>deadline:raise TimeoutError('Adopted backend startup')
                time.sleep(5)
        else:
            (self.root/'backend_ready.json').unlink(missing_ok=True)
            self.status('starting_backend')
            proc=self.launch('backend',self.model('stage_joint_backend.py'),self.a.model_gpu)
            deadline=time.monotonic()+600
            while not (self.root/'backend_ready.json').exists():
                if proc.poll() is not None:raise RuntimeError('Backend exited before readiness')
                if time.monotonic()>deadline:raise TimeoutError('Backend startup')
                time.sleep(5)
        assert request(dict(command='ping'),self.root)['model']=='pi'
        assert json.loads((self.root/'audit/pi.json').read_text())['passed']

    def capture(self):
        status=self.root/'capture_status.json'
        if status.exists() and json.loads(status.read_text()).get('stage')=='complete':return
        existing=matching_processes('stage_joint_capture.py',self.root)
        if existing:
            assert len(existing)==1,existing
            self.adopted.append((existing[0],'stage_joint_capture.py'))
            while alive(existing[0]):
                self.status('capture',adopted_pid=existing[0],
                    train_records=len(list((self.root/'bank/train').glob('*.npz'))),
                    validation_records=len(list((self.root/'bank/validation').glob('*.npz'))))
                # Run the non-gating diagnostic as soon as validation is ready;
                # the independent simulator keeps collecting training frames.
                if len(list((self.root/'bank/validation').glob('*.npz')))==128 and not (self.root/'audit/proxy_review.json').exists():
                    self.call('proxy_diagnostic',self.model('stage_joint_diagnostics.py','--mode','proxy'))
                time.sleep(10)
            if status.exists() and json.loads(status.read_text()).get('stage')=='complete':return
            error_log=(self.logs/'capture.log').read_text()
            if 'Traceback' in error_log:
                raise RuntimeError('Capture stopped with an error; inspect logs/capture.log')
        self.call('capture',self.sim('stage_joint_capture.py'),self.a.sim_gpu)

    def freeze_sources(self):
        stamp=time.strftime('%Y%m%d_%H%M%S');dest=self.root/'source_snapshots'/stamp
        dest.mkdir(parents=True,exist_ok=True)
        names=list((ROOT/'robotwinprobe').glob('stage_joint*.py'))
        names += [ROOT/'robotwinprobe'/n for n in ['crossarch_backend.py','crossarch_capture.py','crossarch_common.py',
            'crossarch_proxy.py','checkpoint_study_common.py','render_model.py','render_proxy.py','render_geometry.py',
            'bootstrap.py','scene_smoke.py','run_sim.sh','policy_server.py','eot_conditions.py']]
        hashes={}
        for path in names:
            shutil.copy2(path,dest/path.name);hashes[path.name]=sha(path)
        dump(dest/'execution.json',dict(model_gpu=self.a.model_gpu,sim_gpu=self.a.sim_gpu,
            sim_candidates=self.a.sim_gpus,protocol_sha256=sha(self.root/'protocol.json'),hashes=hashes))

    def available_gpus(self):
        if self.a.sim_gpus!='auto':return self.a.sim_gpus.split(',')
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=20)
        result=[]
        for line in text.strip().splitlines():
            index,free,util=[v.strip() for v in line.split(',')]
            if index!=self.a.model_gpu and int(free)>=32768 and int(util)<=10:result.append(index)
        if self.a.sim_gpu in result:result=[self.a.sim_gpu]+[g for g in result if g!=self.a.sim_gpu]
        return result

    def formal(self):
        p=load(self.root);pending=[]
        for seed in p['evaluation']['seeds']:
            paths=[self.root/'episodes'/(j['suite']+'_'+j['stage']+'_'+j['condition']+f'_s{seed}.json')
                for j in json.loads((self.root/'jobs.json').read_text()) if j['seed']==seed]
            if not all(path.exists() and json.loads(path.read_text()).get('status') in ('complete','no_opportunity') for path in paths):
                pending.append(seed)
        active={};last=0
        while pending or active:
            for gpu,(proc,seed) in list(active.items()):
                if proc.poll() is not None:
                    if proc.returncode:raise RuntimeError(f'Formal seed {seed} on GPU {gpu} failed; inspect its log')
                    del active[gpu]
            if pending:
                for gpu in self.available_gpus():
                    if gpu in active or len(active)>=self.a.max_workers:continue
                    seed=pending.pop(0)
                    proc=self.launch(f'formal_seed{seed}_gpu{gpu}',self.sim('stage_joint_rollout.py','--seed',seed),gpu)
                    active[gpu]=(proc,seed)
                    if not pending:break
            if time.monotonic()-last>30:
                self.status('formal_closed_loops',pending_initial_states=len(pending),
                    active={g:dict(pid=p.pid,seed=s) for g,(p,s) in active.items()})
                last=time.monotonic()
            if pending or active:time.sleep(5)

    def close(self):
        owned=[]
        for proc in self.children:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            owned.append(dict(pid=proc.pid,stopped=True))
        for pid,script in self.adopted:
            if alive(pid) and pid in matching_processes(script,self.root):
                os.kill(pid,signal.SIGTERM);deadline=time.monotonic()+20
                while alive(pid) and time.monotonic()<deadline:time.sleep(.5)
                if alive(pid):os.kill(pid,signal.SIGKILL)
            owned.append(dict(pid=pid,script=script,adopted=True,stopped=not alive(pid)))
        dump(self.root/'owned_processes.json',dict(processes=owned,finished_unix=time.time()))


def run(a):
    prepare(a.root)
    lock=(a.root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runner=Pipeline(a)
    try:
        runner.freeze_sources()
        if not (a.root/'provenance.json').exists():
            checkpoint=Path(load(a.root)['policies']['pi'])/'model.safetensors'
            identity=sha(checkpoint)
            original=json.loads((ROOT/'robotwinprobe/out/crossarch_joint_20261025/provenance.json').read_text())
            assert original[str(checkpoint)]==identity,'Public policy weights changed'
            dump(a.root/'provenance.json',dict(checkpoint=str(checkpoint),sha256=identity,
                bytes=checkpoint.stat().st_size,policy_parameters_frozen=True))
        runner.backend();runner.capture()
        if not (a.root/'audit/proxy_review.json').exists():
            runner.call('proxy_diagnostic',runner.model('stage_joint_diagnostics.py','--mode','proxy'))
        if not (a.root/'audit/pilot.json').exists():
            for seed in load(a.root)['evaluation']['pilot_seeds']:
                runner.call(f'pilot_{seed}',runner.sim('stage_joint_rollout.py','--seed',seed,'--pilot'),a.sim_gpu)
            rows=[json.loads(f.read_text()) for f in (a.root/'pilot').glob('*.json')]
            assert rows and all(r['status'] in ('complete','no_opportunity') for r in rows)
            assert any(r.get('active_queries')==1 and r['job']['suite']=='pulse' for r in rows),'No actual pulse exercised'
            assert any(r.get('prefix_verified_actions',0)>0 and r.get('active_queries',0)>0 for r in rows),'No delayed prefix exercised'
            dump(a.root/'audit/pilot.json',dict(passed=True,records=len(rows),
                native_clean_successes=sum(r.get('success',False) for r in rows if r['job']['suite']=='native_full' and r['job']['condition']=='clean')))
        if not (a.root/'texture_manifest.json').exists():
            runner.freeze_sources()
            runner.call('patch_training',runner.model('stage_joint_train.py'))
        if not (a.root/'diagnostics/summary.json').exists():
            runner.call('dense_true_render_diagnostics',runner.sim('stage_joint_diagnostics.py','--mode','dense'),a.sim_gpu)
        runner.call('diagnostic_figures',runner.plot('stage_joint_diagnostics.py','--mode','plot'))
        runner.formal()
        runner.status('final_audit');audit(a.root)
        runner.call('figures_and_videos',runner.plot('stage_joint_report.py','--artifacts'))
        runner.status('complete',complete=True)
    except BaseException as e:
        runner.status('stopped_on_error',complete=False,error=repr(e),traceback=traceback.format_exc())
        raise
    finally:runner.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--model-gpu',default='1')
    ap.add_argument('--sim-gpu',default='0');ap.add_argument('--sim-gpus',default='auto')
    ap.add_argument('--max-workers',type=int,default=4);ap.add_argument('--launch',action='store_true')
    a=ap.parse_args();a.root=a.root.resolve()
    if a.launch:
        prepare(a.root);(a.root/'logs').mkdir(exist_ok=True)
        argv=[v for v in sys.argv if v!='--launch']
        with (a.root/'logs/pipeline.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',*argv],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        dump(a.root/'launch.json',dict(pid=proc.pid,argv=argv,started_unix=time.time()))
        print(json.dumps(dict(pid=proc.pid,root=str(a.root))),flush=True)
    else:run(a)
