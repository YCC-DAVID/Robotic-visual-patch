"""Paired vision freeze ablation. No shared-study process or result is modified."""
import argparse, fcntl, json, os, subprocess, time, traceback
from pathlib import Path
import numpy as np
import checkpoint_study_common as S
from checkpoint_study_pipeline import Pipeline as Existing

DEFAULT=S.ROOT/'robotwinprobe/out/vision_freeze_ablation_20261025'
SMOKE_PREFIX='preflight_v6_'

class Pipeline(Existing):
    def __init__(self,args):
        self.args=args;self.root=args.root.resolve();self.p=S.prepare_protocol(self.root)
        self.logs=self.root/'logs';self.logs.mkdir(exist_ok=True)
        self.base=dict(os.environ,PYTHONHASHSEED='0',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
            XLA_PYTHON_CLIENT_PREALLOCATE='false',HF_HUB_OFFLINE='1',WANDB_MODE='disabled',
            PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/vision_freeze_mpl',
            NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_SHM_DISABLE='1')
        self.gpus=args.gpus.split(',')
        self.model_steps=getattr(args,'model_steps',self.p['model_steps'])
        assert len(self.gpus)==2 and len(set(self.gpus))==2
    def report(self):
        subprocess.run([S.PY_CPU,'robotwinprobe/vision_freeze_report.py','--root',str(self.root)],cwd=S.ROOT,check=True)
    def stage(self,name,**extra):
        super().stage(name,**extra);self.report()
    def wait_resources(self):
        while True:
            output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={a.strip():int(b) for line in output.splitlines() for a,b in [line.split(',')]}
            if all(free[g]>=34000 for g in self.gpus):return
            self.stage('waiting_for_gpu_memory',free_MiB={g:free[g] for g in self.gpus},required_each_MiB=34000)
            time.sleep(30)
    def fit(self,variant,smoke=False):
        name=(SMOKE_PREFIX if smoke else '')+variant
        path=self.root/'training'/name/'status.json'
        completed=path.exists() and json.loads(path.read_text()).get('complete')
        audit=self.root/'training'/name/'restore_audit.json'
        if completed and audit.exists() and json.loads(audit.read_text()).get('passed'):return
        self.wait_resources()
        argv=[S.PY_TRAIN,'robotwinprobe/vision_freeze_ft.py','--root',str(self.root),
            '--name',name,'--seed','202610230','--steps','4' if smoke else str(self.model_steps)]
        if variant=='U':argv+=['--unfreeze-vision']
        if smoke:argv+=['--smoke']
        if (self.root/'training'/name/'config.json').exists():
            if path.exists() and json.loads(path.read_text()).get('stage')=='failed':
                raise RuntimeError(f'{name} previously failed; inspect before resuming')
            argv+=['--resume']
        self.stage('preflight_'+variant if smoke else 'training_'+variant)
        if not completed:self.run('fit_'+name,argv,','.join(self.gpus),jax=True)
        verify=argv+(['--resume'] if '--resume' not in argv else [])+['--verify-only']
        self.run('restore_'+name,verify,','.join(self.gpus),jax=True)
    def paired_audit(self,prefix):
        a=json.loads((self.root/'training'/(prefix+'F')/'initial_audit.json').read_text())
        b=json.loads((self.root/'training'/(prefix+'U')/'initial_audit.json').read_text())
        assert a['vision_hashes']==b['vision_hashes'],'Initial vision weights differ between arms'
        assert all(abs(a['monitor'][k]-b['monitor'][k])<1e-6 for k in a['monitor']),'Initial response differs'
        configs=[json.loads((self.root/'training'/(prefix+v)/'config.json').read_text()) for v in ['F','U']]
        left=dict(configs[0]['recipe']);right=dict(configs[1]['recipe'])
        left.pop('freeze_vision');right.pop('freeze_vision');assert left==right
        S.dump(self.root/(prefix+'paired_audit.json'),dict(passed=True,only_freeze_vision_differs=True))
    def evaluate_pair(self,model,pilot=False):
        child,log=self.server(model,'0',8479)
        try:
            textures=[] if pilot else json.loads((self.root/'texture_manifest.json').read_text())
            for condition in ['clean']+[t['id'] for t in textures]:
                argv=['bash','robotwinprobe/run_sim.sh','robotwinprobe/vision_freeze_rollout.py',
                    '--root',str(self.root),'--model',model,'--condition',condition,'--port','8479']
                if pilot:argv+=['--pilot']
                elif model==f'F_s{self.model_steps}' and condition=='clean':argv+=['--capture']
                self.run(('pilot_' if pilot else 'closedloop_')+model+'_'+condition,argv,'1')
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=60)
            log.close()
    def train_models(self):
        for variant in ['F','U']:self.fit(variant)
    def main(self):
        self.stage('awaiting_expert_data')
        collector=json.loads((self.root/'collector_process.json').read_text())['pid']
        while Path(f'/proc/{collector}/cmdline').exists():
            if b'checkpoint_study_collect.py' not in Path(f'/proc/{collector}/cmdline').read_bytes():break
            time.sleep(30);self.report()
        assert json.loads((self.root/'expert/train_manifest.json').read_text())['complete'],'Collection incomplete; inspect collector log'
        for variant in ['U','F']:self.fit(variant,smoke=True)
        self.paired_audit(SMOKE_PREFIX)
        self.stage('preparing_data')
        pending=self.root/'logs/prepare_data_ahead_process.json'
        if pending.exists():
            pid=json.loads(pending.read_text())['pid']
            while Path(f'/proc/{pid}/cmdline').exists():
                if b'checkpoint_study_data.py' not in Path(f'/proc/{pid}/cmdline').read_bytes():break
                time.sleep(30)
        self.run('prepare_data',self.command('data',python=S.PY_TRAIN))
        self.train_models()
        self.paired_audit('')
        if self.args.train_only:
            self.stage('model_training_complete',model_training_complete=True,
                next_stage='export/parity, clean pilot, patches and cross evaluation remain')
            return
        for variant in ['F','U']:
            model=variant+f'_s{self.model_steps}';ck=self.root/'checkpoints/pi05_robotwin_vision_ablation'/variant/str(self.model_steps)
            self.stage('export_'+variant)
            self.run('export_'+variant,[S.PY_MODEL,'robotwinprobe/checkpoint_study_export.py',
                '--checkpoint',str(ck),'--out',str(self.root/'models'/model)])
            for backend,path,py,jax in [('jax',ck,S.PY_TRAIN,True),('torch',self.root/'models'/model,S.PY_MODEL,False)]:
                self.run('parity_'+variant+'_'+backend,self.command('parity','--backend',backend,
                    '--checkpoint',str(path),'--out',str(self.root/'parity'/variant),python=py),'0',jax=jax)
            self.stage('clean_pilot_'+variant);self.evaluate_pair(model,pilot=True)
        pilot=[json.loads((self.root/'pilot'/(v+f'_s{self.model_steps}')/'clean/results.json').read_text()) for v in ['F','U']]
        assert all(p['complete'] and p['task_successes']>=5 for p in pilot),'Clean pilot below5/10: stop before patch training'
        self.stage('patch_training')
        for variant in ['F','U']:
            self.patches([variant+f'_s{self.model_steps}'],'0')
        from PIL import Image
        random_path=self.root/'Random.png';Image.fromarray(np.random.default_rng(202610270).integers(0,256,(128,128,3),dtype=np.uint8)).save(random_path)
        textures=[dict(id='Random',path=str(random_path),sha256=S.sha(random_path))]
        for v in ['F','U']:
            for i,seed in enumerate(self.p['patch']['seeds']):
                path=self.root/'patches'/(v+f'_s{self.model_steps}')/f'P{seed}'/'texture_0500.png'
                textures.append(dict(id=v+f'_P{i}',model=v+f'_s{self.model_steps}',patch_seed=seed,path=str(path),sha256=S.sha(path)))
        S.dump(self.root/'texture_manifest.json',textures)
        self.stage('real_render_cross_evaluation')
        for v in ['F','U']:self.evaluate_pair(v+f'_s{self.model_steps}')
        self.stage('fixed_state_diagnostics')
        for v in ['F','U']:self.run('diagnostics_'+v,[S.PY_MODEL,'robotwinprobe/vision_freeze_diagnostics.py','--root',str(self.root),'--model',v+f'_s{self.model_steps}'],'0')
        S.dump(self.root/'pipeline_status.json',dict(stage='complete',complete=True));self.report()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT)
    ap.add_argument('--gpus',default='0,1');ap.add_argument('--train-only',action='store_true');args=ap.parse_args()
    with (args.root/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pipe=Pipeline(args)
        try:pipe.main()
        except Exception as exc:
            S.dump(args.root/'pipeline_status.json',dict(stage='failed',complete=False,error=repr(exc),traceback=traceback.format_exc()))
            pipe.report();raise
