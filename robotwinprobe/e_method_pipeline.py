"""Durable controlled E-method round; development selection then independent test."""
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

from e_method_common import (DEFAULT,ROOT,SOURCE,PY,OBJECTIVES,SCREEN,BASELINES,
                            cohort,dump,load,prepare,report,request,results,sha)
from stage_joint_pipeline import alive,matching_processes


def choose(data,settings):
    rows={r['condition']:r for r in data['rows']}
    candidates=[rows[name] for name in OBJECTIVES]
    assert all(r['completed_pairs']==data['expected_seeds'] and r['n'] for r in candidates)
    winner=max(candidates,key=lambda r:(r['excess_ASR'],r['ASR'],-OBJECTIVES.index(r['condition'])))
    meets=winner['ASR']>=settings['target']['ASR'] and winner['excess_ASR']>=settings['target']['excess_ASR']
    return dict(objective=winner['condition'],development_result=winner,
                development_target_met=meets,criterion='development excess ASR, ASR, predeclared objective order')


def acceptance(data,conditions,settings):
    target=settings['target'];rows={r['condition']:r for r in data['rows']}
    details={}
    for name in conditions:
        r=rows[name]
        details[name]=dict(ASR=r['ASR'],excess_ASR=r['excess_ASR'],excess_CI95=r['excess_CI95'],n=r['n'],
            passed=bool(r['completed_pairs']==data['expected_seeds'] and r['n']>=target['min_test_clean_successes'] and
                r['ASR']>=target['ASR'] and r['excess_ASR']>=target['excess_ASR'] and
                (not target['positive_paired_CI'] or r['excess_CI95'][0]>0)))
    return dict(passed=all(r['passed'] for r in details.values()),replicas=details,target=target)


class Pipeline:
    def __init__(self,args):
        self.args=args;self.root=args.root;self.children=[];self.adopted=[]
        self.settings=load(self.root)['method_study']
        self.env=dict(os.environ,HF_HUB_OFFLINE='1',JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',
            OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONHASHSEED='0',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',MPLCONFIGDIR=str(self.root/'mplcache'))
        self.env.pop('PYTHONPATH',None)
        signal.signal(signal.SIGTERM,lambda *a:(_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))

    def status(self,stage,**extra):
        dump(self.root/'status.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**extra))
        report(self.root)

    def launch(self,label,argv,gpu=''):
        path=self.root/'logs'/f'{label}.log';path.parent.mkdir(exist_ok=True)
        with path.open('a') as log:
            child=subprocess.Popen(argv,cwd=ROOT,env=dict(self.env,CUDA_VISIBLE_DEVICES=str(gpu)),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        self.children.append(child);return child

    def command(self,script,*args,sim=False):
        argv=(["bash",str(ROOT/'robotwinprobe/run_sim.sh')] if sim else [PY,'-u'])
        return [*argv,str(ROOT/'robotwinprobe'/script),'--root',str(self.root),*map(str,args)]

    def call(self,label,argv,gpu=''):
        proc=self.launch(label,argv,gpu)
        while proc.poll() is None:
            self.status(label,child_pid=proc.pid);time.sleep(10)
        if proc.returncode:raise RuntimeError(f'{label} exited {proc.returncode}; see logs/{label}.log')

    def snapshot_sources(self):
        dest=self.root/'source_snapshots'/time.strftime('%Y%m%d_%H%M%S');dest.mkdir(parents=True,exist_ok=True)
        files=list((ROOT/'robotwinprobe').glob('e_method*.py'))
        files += [ROOT/'robotwinprobe'/n for n in ['crossarch_backend.py','crossarch_proxy.py','crossarch_common.py',
            'render_model.py','render_proxy.py','render_geometry.py','stage_joint_rollout.py','stage_joint_common.py',
            'stage_joint_report.py','stage_joint_pipeline.py','stage_joint_monitor.py','bootstrap.py','run_sim.sh','test_e_method.py']]
        for path in files:shutil.copy2(path,dest/path.name)
        dump(dest/'hashes.json',{p.name:sha(p) for p in files})

    def backend(self):
        matches=matching_processes('e_method_backend.py',self.root)
        if matches:
            assert len(matches)==1;pid=matches[0];self.adopted.append((pid,'e_method_backend.py'))
        else:
            (self.root/'backend_ready.json').unlink(missing_ok=True)
            pid=self.launch('backend',self.command('e_method_backend.py'),self.args.model_gpu).pid
        deadline=time.monotonic()+600
        while not (self.root/'backend_ready.json').exists():
            if not alive(pid):raise RuntimeError('Backend exited during startup')
            if time.monotonic()>deadline:raise TimeoutError('Backend startup timed out')
            self.status('starting_backend',backend_pid=pid);time.sleep(5)
        assert request(dict(command='ping'),self.root)['ready']
        assert json.loads((self.root/'audit/pi.json').read_text())['passed']
        if not (self.root/'audit/method_gradients.json').exists():
            source=next(p for p in sorted((SOURCE/'bank/train').glob('*.npz')) if '_f0000_' in p.name)
            self.status('method_gradient_audit')
            request(dict(command='method_audit',record=str(source)),self.root)
        assert json.loads((self.root/'audit/method_gradients.json').read_text())['passed']

    def gpus(self):
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=20)
        available=[]
        for row in text.strip().splitlines():
            index,memory,util=(x.strip() for x in row.split(','))
            if index!=self.args.model_gpu and int(memory)>=12000 and int(util)<=20:available.append(index)
        preferred=self.args.sim_gpu
        return ([preferred] if preferred in available else [])+[g for g in available if g!=preferred]

    def collection(self):
        if (self.root/'collection_manifest.json').exists():return
        matches=matching_processes('e_method_sim.py',self.root)
        # Collection is the only simulator worker allowed before development.
        if matches:
            assert len(matches)==1;pid=matches[0];self.adopted.append((pid,'e_method_sim.py'))
            while alive(pid):
                self.status('policy_state_collection',child_pid=pid);time.sleep(10)
            if not (self.root/'collection_manifest.json').exists():raise RuntimeError('Adopted collection stopped; see logs/collection.log')
            return
        while not self.gpus():
            self.status('waiting_collection_gpu');time.sleep(10)
        self.call('policy_state_collection',self.command('e_method_sim.py','--mode','collect',sim=True),self.gpus()[0])

    def train(self,name,kind,seed):
        complete=self.root/'training'/name/'complete.json'
        if complete.exists():
            saved=json.loads(complete.read_text())
            assert saved['kind']==kind and saved['training_seed']==seed,'Completed training identity mismatch'
            assert sha(self.root/'patches'/name/'texture_0500.png')==saved['texture_sha256']
            return
        self.call('train_'+name,self.command('e_method_train.py','--name',name,'--kind',kind,'--training-seed',seed))

    def evaluate(self,split,conditions):
        folder=cohort(self.root,split);assignment_path=folder/'gpu_assignments.json'
        assignments=json.loads(assignment_path.read_text()) if assignment_path.exists() else {}
        pending=[]
        for seed in self.settings[split+'_seeds']:
            paths=[folder/'episodes'/f'native_full_approach_{c}_s{seed}.json' for c in conditions]
            if not all(path.exists() and json.loads(path.read_text()).get('status')=='complete' for path in paths):pending.append(seed)
        active={}
        while pending or active:
            for gpu,(proc,seed) in list(active.items()):
                if proc.poll() is not None:
                    if proc.returncode:raise RuntimeError(f'{split} seed {seed} exited {proc.returncode}; see its log')
                    del active[gpu]
            available=self.gpus()
            for gpu in available:
                if gpu in active or len(active)>=self.args.max_workers:continue
                eligible=[s for s in pending if assignments.get(str(s),gpu)==gpu]
                if not eligible:continue
                seed=eligible[0];pending.remove(seed)
                assignments[str(seed)]=gpu;dump(assignment_path,assignments)
                proc=self.launch(f'{split}_seed{seed}',self.command('e_method_sim.py','--mode','evaluate','--split',split,
                    '--seed',seed,'--conditions',','.join(conditions),sim=True),gpu)
                active[gpu]=(proc,seed)
            self.status(split+'_evaluation',pending_initial_states=len(pending),
                conditions=list(conditions),active={g:dict(pid=p.pid,seed=s) for g,(p,s) in active.items()})
            if pending or active:time.sleep(10)

    def audit(self,split,conditions):
        from stage_joint_rollout import digest,noise_seed
        folder=cohort(self.root,split);p=load(self.root);count=0
        for seed in self.settings[split+'_seeds']:
            clean=json.loads((folder/'episodes'/f'native_full_approach_clean_s{seed}.json').read_text())
            for condition in conditions:
                path=folder/'episodes'/f'native_full_approach_{condition}_s{seed}.json'
                row=json.loads(path.read_text());assert row['status']=='complete' and not row.get('error')
                assert row['protocol_sha256']==sha(folder/'protocol.json')
                assert row['initial_clean_sha256']==clean['initial_clean_sha256'] and row['prompt']==clean['prompt']
                assert sha(path.with_suffix('.npz'))==row['artifact_sha256']
                with np_load(path.with_suffix('.npz')) as z:
                    expected=[noise_seed(p['evaluation']['noise_base'],seed,'approach','native_full',q) for q in range(len(z['noise_seeds']))]
                    import numpy as np
                    assert np.array_equal(expected,z['noise_seeds'])
                    assert len(z['executed_actions'])==row['steps']
                    if condition!='clean':
                        assert z['patch_active'].all() and row['affected_actions']==row['steps']
                count+=1
        result=dict(passed=True,split=split,records=count,checks='protocol/artifact hashes; same initial state/RGB/prompt; paired noise; persistent patch and exact action counts')
        dump(self.root/'audit'/f'{split}.json',result)

    def close(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        for pid,script in self.adopted:
            if alive(pid) and pid in matching_processes(script,self.root):os.kill(pid,signal.SIGTERM)


def np_load(path):
    import numpy as np
    return np.load(path)


def run(args):
    prepare(args.root)
    with (args.root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runner=Pipeline(args)
        try:
            runner.snapshot_sources()
            checkpoint=Path(load(args.root)['policies']['pi'])/'model.safetensors'
            identity=sha(checkpoint)
            assert identity==json.loads((SOURCE/'provenance.json').read_text())['sha256']
            dump(args.root/'provenance.json',dict(checkpoint=str(checkpoint),sha256=identity,policy_frozen=True))
            runner.backend()
            if args.diagnose:
                runner.call('transfer_diagnostic',runner.command('e_method_probe.py'))
                runner.status('diagnostic_complete_needs_method_iteration',complete=False,
                    next_action='Analyze development-only real-render L0/action transfer diagnosis and execute next controlled method round.')
                return
            runner.collection();runner.snapshot_sources()
            for kind in OBJECTIVES:runner.train(kind,kind,runner.settings['training_random_seeds'][0])
            runner.evaluate('development',SCREEN);runner.audit('development',SCREEN)
            development=results(args.root,'development',SCREEN)
            selection=choose(development,runner.settings);dump(args.root/'selection.json',selection)
            if not selection['development_target_met']:
                runner.status('round_complete_below_target',complete=False,
                    next_action='Analyze controlled losses and real-render behavior; continue method improvement. Do not claim target achieved.')
                return
            winner=selection['objective']
            for index,seed in enumerate(runner.settings['training_random_seeds'][1:],1):
                runner.train('replica'+str(index),winner,seed)
            runner.evaluate('development',(*SCREEN,'replica1','replica2'))
            runner.audit('development',(*SCREEN,'replica1','replica2'))
            dump(args.root/'frozen_selection.json',dict(selection=selection,
                replica_textures={name:sha(args.root/'patches'/name/'texture_0500.png') for name in (winner,'replica1','replica2')},
                frozen_before_test_unix=time.time(),test_used_for_selection=False))
            final_conditions=(*BASELINES,winner,'replica1','replica2')
            runner.evaluate('test',final_conditions);runner.status('final_audit')
            runner.audit('test',final_conditions)
            decision=acceptance(results(args.root,'test',final_conditions),(winner,'replica1','replica2'),runner.settings)
            dump(args.root/'acceptance.json',decision)
            runner.status('complete' if decision['passed'] else 'round_complete_below_target',
                complete=decision['passed'],independent_test_complete=True,
                next_action='Report held-out result; if below target analyze next method round without tuning on this test cohort.')
        except BaseException as error:
            runner.status('stopped_on_error',error=repr(error),traceback=traceback.format_exc(),complete=False)
            raise
        finally:runner.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--model-gpu',default='4');parser.add_argument('--sim-gpu',default='2')
    parser.add_argument('--max-workers',type=int,default=2);parser.add_argument('--launch',action='store_true')
    parser.add_argument('--diagnose',action='store_true')
    args=parser.parse_args();args.root=args.root.resolve()
    if args.launch:
        prepare(args.root);(args.root/'logs').mkdir(exist_ok=True)
        argv=[v for v in sys.argv if v!='--launch']
        with (args.root/'logs/pipeline.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',*argv],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        dump(args.root/'launch.json',dict(pid=child.pid,started_unix=time.time(),argv=argv))
        print(json.dumps(dict(pid=child.pid,root=str(args.root))))
    else:run(args)
