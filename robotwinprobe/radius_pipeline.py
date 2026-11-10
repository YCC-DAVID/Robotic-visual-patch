"""Owned, restartable continuous-radius experiment; exact calibration then frozen testing."""
import argparse, hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
import bootstrap as B
from radius_plan import prepare, dump, REFERENCE, RADII

PYTHON='/home/user1/miniconda3/envs/openpi-server/bin/python'


class Runner:
    def __init__(self, root, gpu, name):
        self.root=root;self.gpu=gpu;self.name=name;self.children=[];self.logs=root/'logs';self.logs.mkdir(exist_ok=True)
        self.env=os.environ.copy();self.env.pop('PYTHONPATH',None)
        self.env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONNOUSERSITE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        signal.signal(signal.SIGTERM,lambda sig,frame: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    def status(self,stage,**extra):
        d={'stage':stage,'pid':os.getpid(),'gpu':self.gpu,'updated_unix':time.time(),**extra}
        dump(self.root/(self.name+'_status.json'),d);print('[stage]',self.name,d,flush=True)
    def model(self,name,*args):return [PYTHON,'-u',str(B.PROBE/name),*map(str,args)]
    def sim(self,name,*args):return ['bash',str(B.PROBE/'run_sim.sh'),str(B.PROBE/name),*map(str,args)]
    def launch(self,name,cmd,env=None):
        log=open(self.logs/(name+'.log'),'a')
        p=subprocess.Popen(cmd,cwd=B.ROOT,env=env or self.env,stdout=log,stderr=subprocess.STDOUT)
        self.children.append((p,log));return p
    def run(self,name,cmd):
        self.status(name);p=self.launch(name,cmd);code=p.wait()
        for pair in self.children[:]:
            if pair[0] is p:pair[1].close();self.children.remove(pair)
        if code:raise RuntimeError(f'{name} exited {code}; see logs/{name}.log')
    def server(self,port):
        meta=self.root/(self.name+'_server.json')
        if meta.exists():meta.unlink()
        p=self.launch(self.name+'_server',self.model('policy_server.py','--port',port,'--metadata',meta))
        dump(self.root/(self.name+'_owned_server.json'),{'pid':p.pid,'port':port,'stopped':False})
        deadline=time.monotonic()+360
        while not meta.exists():
            if p.poll() is not None:raise RuntimeError('owned server failed to start')
            if time.monotonic()>deadline:raise TimeoutError('owned server readiness')
            time.sleep(1)
        return p
    def close(self):
        for p,log in self.children:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            log.close()
        owned=self.root/(self.name+'_owned_server.json')
        if owned.exists():
            d=json.loads(owned.read_text());d['stopped']=True;dump(owned,d)


def train_worker(run,radius):
    root=run.root
    for split in ['validation','train']:
        if split=='train':
            result=root/'validation'/f'r{radius:02d}'/'results.json'
            if not result.exists():run.run(f'validate_r{radius:02d}',run.model('radius_validate.py','--root',root,'--radius',radius))
            assert json.loads(result.read_text())['passed']
        run.run(f'capture_r{radius:02d}_{split}',run.sim('radius_capture.py','--root',root,'--radius',radius,'--split',split))
    # Capture hashes bind the exact continuously sampled calibration to this training run.
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root/'capture'/f'r{radius:02d}').glob('*/*.npz'))}
    dump(root/'capture'/f'r{radius:02d}'/'sha256.json',hashes)
    progress=root/f'train_r{radius:02d}/progress.json'
    if not (progress.exists() and json.loads(progress.read_text())['complete']):
        run.run(f'train_r{radius:02d}',run.model('radius_train.py','--root',root,'--radius',radius))
    run.status('complete',radius_cm=radius)


def evaluate_worker(run,methods,port):
    root=run.root;count=json.loads((root/'cohort.json').read_text())['count']
    run.server(port)
    for method in methods:
        run.run('eval_'+method,run.sim('radius_eval.py','--root',root,'--method',method,'--port',port,'--count',count))
    run.status('complete',methods=methods)


def smoke(run,port):
    root=run.root;run.server(port)
    for method,limit in [('clean',1),('baseline',2)]:
        run.run('smoke_eval_'+method,run.sim('radius_eval.py','--root',root,'--method',method,'--port',port,
                                         '--count',50,'--max-episodes',limit))
    clean=json.loads((root/'eval/clean/seed10400_p00.json').read_text())
    for i in [0,1]:
        d=json.loads((root/'eval/baseline'/f'seed10400_p{i:02d}.json').read_text())
        for k in ['initial_rgb_sha256','initial_state_sha256','first_noise_seed','prompt']:assert d[k]==clean[k]
    dump(root/'smoke_complete.json',{'passed':True,'checked':'clean, baseline origin, baseline random position; initial RGB/state/noise paired'})
    run.status('complete')


def full(run,gpus,port):
    root=run.root
    assert json.loads((root/'smoke_complete.json').read_text())['passed']
    assert json.loads((root/'trainer_smoke/regression.json').read_text())['max_loss_difference']==0
    snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True);hashes={}
    paths=list(B.PROBE.glob('radius_*.py'))+[B.PROBE/n for n in ['render_model.py','render_proxy.py','render_geometry.py','scene_smoke.py','policy_server.py']]
    for p in paths:
        data=p.read_bytes();hashes[p.name]=hashlib.sha256(data).hexdigest()
        target=snapshot/p.name
        if target.exists():assert target.read_bytes()==data,'source changed: preserve prior run'
        else:target.write_bytes(data)
    dump(snapshot/'sha256.json',hashes)
    run.status('continuous_calibration_and_training',radii_cm=RADII,gpus=gpus)
    workers=[]
    for radius,gpu in zip(RADII,gpus):
        env=run.env.copy();env['CUDA_VISIBLE_DEVICES']=gpu
        workers.append(run.launch(f'worker_r{radius:02d}',run.model('radius_pipeline.py','--root',root,'--stage','train',
            '--gpu',gpu,'--radius',radius),env))
    while workers:
        for p in workers[:]:
            if p.poll() is not None:
                if p.returncode:raise RuntimeError(f'training worker {p.pid} exited {p.returncode}')
                workers.remove(p)
        if workers:time.sleep(5)
    run.status('all_training_complete')
    if not (root/'cohort.json').exists():
        server=run.server(port)
        try:
            run.run('clean_50',run.sim('radius_eval.py','--root',root,'--method','clean','--port',port,'--count',50))
            good=sum(json.loads((root/'eval/clean'/f'seed{s}_p00.json').read_text())['success'] for s in range(10400,10450))
            count=100 if good<30 else 50
            if count==100:run.run('clean_100',run.sim('radius_eval.py','--root',root,'--method','clean','--port',port,'--count',100))
            dump(root/'cohort.json',{'count':count,'seeds':list(range(10400,10400+count)),
                'first50_clean_successes':good,'rule':'extend entire cohort iff first50 clean successes <30'})
        finally:
            if server.poll() is None:
                server.terminate()
                try:server.wait(timeout=15)
                except subprocess.TimeoutExpired:server.kill();server.wait()
    run.status('frozen_patch_real_render_evaluation')
    workers=[]
    for index,(gpu,methods) in enumerate(zip(gpus,[['baseline','random'],['r01','r03'],['r05']])):
        env=run.env.copy();env['CUDA_VISIBLE_DEVICES']=gpu
        workers.append(run.launch(f'eval_worker_{index}',run.model('radius_pipeline.py','--root',root,'--stage','eval',
            '--gpu',gpu,'--methods',','.join(methods),'--port',port+index+1),env))
    last_report=0
    while workers:
        for p in workers[:]:
            if p.poll() is not None:
                if p.returncode:raise RuntimeError(f'eval worker {p.pid} exited {p.returncode}')
                workers.remove(p)
        if time.monotonic()-last_report>300:
            run.run('partial_report',run.model('radius_report.py','--root',root));last_report=time.monotonic()
            run.status('frozen_patch_real_render_evaluation',workers=[p.pid for p in workers])
        if workers:time.sleep(5)
    run.run('final_report',run.model('radius_report.py','--root',root))
    assert json.loads((root/'summary.json').read_text())['complete']
    run.status('complete',report=str(root/'results.md'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--stage',choices=['full','train','eval','smoke'],default='full')
    ap.add_argument('--gpu',default='6');ap.add_argument('--gpus',default='2,4,6')
    ap.add_argument('--radius',type=int,choices=RADII);ap.add_argument('--methods',default='')
    ap.add_argument('--port',type=int,default=8240);args=ap.parse_args()
    root=Path(args.root).resolve();prepare(root)
    if not (root/'textures.json').exists():
        from PIL import Image
        from render_geometry import patterns
        Image.fromarray(patterns()['random']).save(root/'random.png')
        dump(root/'textures.json',{'baseline':str(REFERENCE/'train_proxy_full/texture_0500.png'),
            'random':str(root/'random.png'),**{f'r{r:02d}':str(root/f'train_r{r:02d}/texture_0500.png') for r in RADII}})
    name='pipeline' if args.stage=='full' else f'train_r{args.radius:02d}' if args.stage=='train' else 'eval_'+args.gpu if args.stage=='eval' else 'smoke'
    run=Runner(root,args.gpu,name)
    try:
        if args.stage=='full':
            gpus=args.gpus.split(',');assert len(gpus)==3 and len(set(gpus))==3
            full(run,gpus,args.port)
        elif args.stage=='train':train_worker(run,args.radius)
        elif args.stage=='eval':evaluate_worker(run,args.methods.split(','),args.port)
        else:smoke(run,args.port)
    except BaseException:
        run.status('failed',error=traceback.format_exc());raise
    finally:run.close()


if __name__=='__main__':main()
