"""Restartable controlled study; only owns processes it starts and GPUs supplied.

Collection -> exact-data cache -> matched FT -> export/parity -> clean pilot ->
patch replicas -> complete real-render cross matrix -> fixed-state diagnostics.
"""
import argparse
import concurrent.futures as futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import checkpoint_study_common as S


class Pipeline:
    def __init__(self,args):
        self.root=args.root.resolve();self.args=args;self.p=S.prepare_protocol(self.root)
        self.logs=self.root/'logs';self.logs.mkdir(parents=True,exist_ok=True)
        self.gpus=args.gpus.split(',');assert len(self.gpus)==3
        self.base=os.environ.copy();self.base.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
            XLA_PYTHON_CLIENT_PREALLOCATE='false',HF_HUB_OFFLINE='1',WANDB_MODE='disabled',
            PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',MPLCONFIGDIR='/tmp/wam_checkpoint_mpl')

    def env(self,gpu='',jax=False):
        return self.base|dict(CUDA_VISIBLE_DEVICES=gpu,JAX_PLATFORMS='cuda,cpu' if jax else 'cpu')

    def stage(self,name,**extra):
        S.dump(self.root/'pipeline_status.json',dict(stage=name,complete=False,pid=os.getpid(),updated_unix=time.time(),**extra))
        print('[stage]',name,extra,flush=True)

    def run(self,name,argv,gpu='',jax=False):
        status=self.logs/(name+'.json')
        with (self.logs/(name+'.log')).open('a') as log:
            child=subprocess.Popen(argv,cwd=S.ROOT,env=self.env(gpu,jax),stdout=log,stderr=subprocess.STDOUT)
            S.dump(status,dict(running=True,pid=child.pid,argv=list(map(str,argv)),gpu=gpu))
            code=child.wait()
        S.dump(status,dict(running=False,complete=code==0,returncode=code,argv=list(map(str,argv)),gpu=gpu))
        if code:raise RuntimeError(f'{name} exited {code}; see {self.logs/(name+".log")}')

    def command(self,script,*args,python=S.PY_MODEL):
        return [python,'robotwinprobe/checkpoint_study_'+script+'.py','--root',str(self.root),*map(str,args)]

    def collect(self,split,count,seed,gpu,adopt):
        # Existing collectors are observed, never killed or duplicated.
        while adopt:
            path=Path(f'/proc/{adopt}/cmdline')
            try:command=path.read_bytes()
            except FileNotFoundError:break
            if b'checkpoint_study_collect.py' not in command:break
            time.sleep(20)
        path=self.root/'expert'/f'{split}_manifest.json'
        if path.exists() and json.loads(path.read_text()).get('complete'):return
        self.run('collect_'+split,['bash','robotwinprobe/run_sim.sh','robotwinprobe/checkpoint_study_collect.py',
            '--root',str(self.root),'--split',split,'--count',str(count),'--seed',str(seed)],gpu)

    def train(self,i,gpu):
        name=f'M{i}';status=self.root/'training'/name/'status.json'
        if status.exists() and json.loads(status.read_text()).get('complete'):return
        cmd=self.command('ft','--name',name,'--seed',self.p['model_seeds'][i],
            '--steps',self.p['model_steps'],python=S.PY_TRAIN)
        if (self.root/'training'/name/'config.json').exists():cmd.append('--resume')
        self.run('train_'+name,cmd,gpu,jax=True)

    def wait_for_training_gpus(self):
        while True:
            output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={a.strip():int(b.strip()) for line in output.splitlines() for a,b in [line.split(',')]}
            busy={g:memory[g] for g in self.gpus if memory[g]>2048}
            if not busy:return
            self.stage('waiting_for_training_gpus',occupied_MiB=busy,action='waiting; unrelated processes are not terminated')
            self.report()
            time.sleep(30)

    def server(self,model,gpu,port):
        ready=self.root/'servers'/f'{model}_{port}.json';ready.parent.mkdir(exist_ok=True)
        if ready.exists():ready.unlink()
        log=(self.logs/f'server_{model}_{port}.log').open('a')
        cmd=[S.PY_MODEL,'robotwinprobe/checkpoint_study_server.py','--checkpoint',str(self.root/'models'/model),
            '--port',str(port),'--ready',str(ready)]
        child=subprocess.Popen(cmd,cwd=S.ROOT,env=self.env(gpu),stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+600
        while not ready.exists():
            if child.poll() is not None or time.monotonic()>deadline:
                if child.poll() is None:child.terminate();child.wait()
                log.close();raise RuntimeError(f'Server failed: {model}; see server log')
            time.sleep(2)
        return child,log

    def evaluate(self,model,gpu,slot,pilot=False):
        port=8370+slot;child,log=self.server(model,gpu,port)
        try:
            textures=[] if pilot else json.loads((self.root/'texture_manifest.json').read_text())
            for condition in ['clean']+[r['id'] for r in textures]:
                stage='pilot' if pilot else 'closedloop'
                cmd=['bash','robotwinprobe/run_sim.sh','robotwinprobe/checkpoint_study_rollout.py',
                    '--root',str(self.root),'--model',model,'--condition',condition,'--port',str(port)]
                if pilot:cmd.append('--pilot')
                elif model=='M0_s20000' and condition=='clean':cmd.append('--capture')
                self.run(f'{stage}_{model}_{condition}',cmd,gpu)
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=60)
            log.close()

    def patches(self,models,gpu):
        for model in models:
            for seed in self.p['patch']['seeds']:
                self.run(f'patch_{model}_{seed}',self.command('patch','--model',model,'--seed',seed),gpu)

    def report(self):
        self.run('report',self.command('report',python=S.PY_CPU))

    def main(self):
        assert json.loads((self.root/'smoke_parity/parity.json').read_text())['passed'], 'Preflight parity required'
        assert json.loads((self.root/'expert_replay/audit.json').read_text())['passed'], 'Expert replay required'
        assert json.loads((self.root/'smoke_interface.json').read_text())['passed'], 'Closed-loop interface preflight required'
        smoke_patch=self.root/'patches/smoke/P202610240/status.json'
        assert json.loads(smoke_patch.read_text())['complete'], 'Proxy gradient preflight required'
        assert json.loads((self.root/'preflight_render/diagnostics/smoke/status.json').read_text())['complete'], 'Actual-render diagnostics preflight required'
        self.stage('fingerprinting_inputs')
        inventory=self.root/'base_inventory.json'
        base_hashes={str(path.relative_to(S.BASE)):S.sha(path) for path in sorted(S.BASE.rglob('*')) if path.is_file()}
        if inventory.exists():assert json.loads(inventory.read_text())==base_hashes,'Pretrained base changed'
        else:S.dump(inventory,base_hashes)
        sources={str(path.relative_to(S.ROOT)):S.sha(path) for path in sorted((S.ROOT/'robotwinprobe').glob('checkpoint_study_*.py'))}
        upstream={str(path.relative_to(S.SERVER)):S.sha(path) for path in sorted((S.SERVER/'src/openpi').rglob('*.py'))}
        S.dump(self.root/'execution_manifest.json',dict(started_unix=time.time(),sources=sources,
            upstream_source_hashes=upstream,
            python_hash_seed=0,
            protocol_sha256=S.sha(self.root/'protocol.json'),base_inventory_sha256=S.sha(inventory),gpus=self.gpus,
            independent_models=3,distinct_checkpoints=5,patches=15,pilot_episodes=30,primary_closed_loops=8500))
        self.stage('collecting_expert_data')
        with futures.ThreadPoolExecutor(2) as pool:
            jobs=[pool.submit(self.collect,'train',100,20000,self.gpus[1],self.args.adopt_train),
                  pool.submit(self.collect,'validation',20,21000,self.gpus[2],self.args.adopt_validation)]
            while not all(job.done() for job in jobs):
                for job in jobs:
                    if job.done():job.result()
                futures.wait(jobs,timeout=30)
                self.report()
            for job in jobs:job.result()
        self.stage('preparing_fixed_data')
        self.run('prepare_data',self.command('data',python=S.PY_TRAIN))
        self.report();self.wait_for_training_gpus()
        self.stage('matched_finetuning',model_seeds=self.p['model_seeds'],gpus=self.gpus)
        with futures.ThreadPoolExecutor(3) as pool:
            jobs=[pool.submit(self.train,i,gpu) for i,gpu in enumerate(self.gpus)]
            for job in jobs:job.result()
        configs=[json.loads((self.root/'training'/f'M{i}'/'config.json').read_text()) for i in range(3)]
        assert configs[0]['recipe']==configs[1]['recipe']==configs[2]['recipe']
        assert all(c['steps']==20000 and c['seed']==s for c,s in zip(configs,self.p['model_seeds']))
        S.dump(self.root/'matched_recipe_audit.json',dict(passed=True,only_model_seed_differs=True,
            configs_sha256=[S.sha(self.root/'training'/f'M{i}'/'config.json') for i in range(3)]))
        models=S.checkpoint_specs(self.root)
        self.stage('export_and_parity');self.report()
        for m in models:
            ckpt=self.root/'checkpoints/pi05_robotwin_controlled'/m['train_id']/str(m['step'])
            self.run('export_'+m['id'],[S.PY_MODEL,'robotwinprobe/checkpoint_study_export.py',
                '--checkpoint',str(ckpt),'--out',m['path']])
            parity=self.root/'parity'/m['id']
            for backend,path,py,jax in [('jax',ckpt,S.PY_TRAIN,True),('torch',m['path'],S.PY_MODEL,False)]:
                self.run('parity_'+m['id']+'_'+backend,self.command('parity','--backend',backend,
                    '--checkpoint',path,'--out',parity,python=py),self.gpus[0],jax=jax)
        self.stage('clean_pilot')
        with futures.ThreadPoolExecutor(3) as pool:
            jobs=[pool.submit(self.evaluate,f'M{i}_s20000',gpu,i,True) for i,gpu in enumerate(self.gpus)]
            for job in jobs:job.result()
        pilot=[json.loads((self.root/'pilot'/f'M{i}_s20000'/'clean/results.json').read_text()) for i in range(3)]
        gate=all(p['complete'] and p['task_successes']>=5 for p in pilot)
        S.dump(self.root/'clean_pilot_gate.json',dict(passed=gate,rule='each final checkpoint at least 5/10 clean successes; fixed pilot seeds 23000..23009',successes=[p['task_successes'] for p in pilot]))
        assert gate,'Clean pilot inadequate; inspect policy/data before spending patch/evaluation budget'
        self.stage('matched_patch_training')
        buckets=[[m['id'] for m in models][i::3] for i in range(3)]
        with futures.ThreadPoolExecutor(3) as pool:
            jobs=[pool.submit(self.patches,b,gpu) for b,gpu in zip(buckets,self.gpus)]
            for job in jobs:job.result()
        from PIL import Image
        random_path=self.root/'Random.png'
        Image.fromarray(np.random.default_rng(202610270).integers(0,256,(128,128,3),dtype=np.uint8)).save(random_path)
        textures=[dict(id='Random',path=str(random_path),sha256=S.sha(random_path))]
        for m in models:
            for i,seed in enumerate(self.p['patch']['seeds']):
                path=self.root/'patches'/m['id']/f'P{seed}'/'texture_0500.png'
                textures.append(dict(id=m['id']+f'_P{i}',model=m['id'],patch_seed=seed,path=str(path),sha256=S.sha(path)))
        S.dump(self.root/'texture_manifest.json',textures)
        self.report();self.stage('real_render_cross_matrix',expected_episodes=8500)
        def evaluation_lane(i):
            for model in buckets[i]:self.evaluate(model,self.gpus[i],i)
        with futures.ThreadPoolExecutor(3) as pool:
            jobs=[pool.submit(evaluation_lane,i) for i in range(3)]
            for job in jobs:job.result()
        self.stage('fixed_state_diagnostics');self.report()
        def diagnostic_lane(i):
            for model in buckets[i]:self.run('diagnostic_'+model,self.command('diagnostics','--model',model),self.gpus[i])
        with futures.ThreadPoolExecutor(3) as pool:
            jobs=[pool.submit(diagnostic_lane,i) for i in range(3)]
            for job in jobs:job.result()
        self.report()
        S.dump(self.root/'pipeline_status.json',dict(stage='complete',complete=True,completed_unix=time.time()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=S.DEFAULT)
    ap.add_argument('--gpus',default='3,5,6');ap.add_argument('--adopt-train',type=int,default=0)
    ap.add_argument('--adopt-validation',type=int,default=0);args=ap.parse_args()
    args.root=args.root.resolve();args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pipeline=Pipeline(args)
        try:pipeline.main()
        except Exception as exc:
            S.dump(args.root/'pipeline_status.json',dict(stage='failed',complete=False,
                error=repr(exc),traceback=traceback.format_exc(),updated_unix=time.time()))
            try:pipeline.report()
            except Exception as report_error:print('[report failed]',repr(report_error),flush=True)
            raise
