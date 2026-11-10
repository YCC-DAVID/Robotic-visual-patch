"""Owned background pipeline: measured-bank gate, EOT training, paired real rollouts."""
import argparse, hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
import bootstrap as B
from eot_conditions import conditions, evaluation_conditions

PYTHON = '/home/user1/miniconda3/envs/openpi-server/bin/python'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--gpu',default='6');ap.add_argument('--port',type=int,default=8186)
    ap.add_argument('--capture-pid',type=int)
    ap.add_argument('--capture-gpus',help='Launch owned disjoint-seed capture workers, e.g. 2,4,6')
    args=ap.parse_args();root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    logs=root/'logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=args.gpu
    env['PYTHONNOUSERSITE']='1';env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    env.pop('PYTHONPATH',None)
    children=[];started=time.time()
    def status(stage,**extra):
        record={'stage':stage,'pid':os.getpid(),'started_unix':started,'updated_unix':time.time(),**extra}
        tmp=root/'status.tmp.json';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(root/'status.json')
        print('[stage]',record,flush=True)
    def run(name,cmd):
        status(name)
        with open(logs/(name+'.log'),'a') as log:
            child=subprocess.Popen(cmd,cwd=B.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append(child)
            code=child.wait();children.remove(child)
            if code:raise RuntimeError(f'{name} exited {code}; inspect {log.name}')
    def model(script,*argv):return [PYTHON,'-u',str(B.PROBE/script),*map(str,argv)]
    def sim(script,*argv):return ['bash',str(B.PROBE/'run_sim.sh'),str(B.PROBE/script),*map(str,argv)]
    def interrupted(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
    signal.signal(signal.SIGTERM,interrupted)
    snap=root/'source_snapshot';snap.mkdir(exist_ok=True)
    hashes={}
    for p in sorted(B.PROBE.glob('eot_*.py'))+ [B.PROBE/n for n in ['render_proxy.py','render_model.py','render_geometry.py','scene_smoke.py']]:
        data=p.read_bytes();(snap/p.name).write_bytes(data);hashes[p.name]=hashlib.sha256(data).hexdigest()
    (snap/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    try:
        status('waiting_for_measured_bank')
        if args.capture_gpus:
            gpus=args.capture_gpus.split(',')
            seeds=list(range(10100,10108))+[10110,10111]
            workers=[]
            for index,gpu in enumerate(gpus):
                worker_env=env.copy();worker_env['CUDA_VISIBLE_DEVICES']=gpu
                log=open(logs/f'capture_gpu{gpu}.log','a')
                cmd=sim('eot_capture.py','--root',root/'bank','--reference',B.PROBE/'out/render_transfer',
                        '--seeds',','.join(map(str,seeds[index::len(gpus)])))
                child=subprocess.Popen(cmd,cwd=B.ROOT,env=worker_env,stdout=log,stderr=subprocess.STDOUT)
                children.append(child);workers.append((child,log))
            status('capturing_measured_bank',workers=[{'pid':p.pid,'log':log.name} for p,log in workers])
            while workers:
                for child,log in workers[:]:
                    if child.poll() is not None:
                        code=child.returncode;log.close();children.remove(child);workers.remove((child,log))
                        if code:raise RuntimeError(f'capture worker exited {code}; {log.name}')
                if workers:time.sleep(5)
        expected=[root/'bank'/k/f'seed{s}.npz' for k in conditions()
                  for s in list(range(10100,10108))+[10110,10111]]
        while not all(p.exists() for p in expected):
            if args.capture_pid:
                try:os.kill(args.capture_pid,0)
                except ProcessLookupError:raise RuntimeError('capture stopped before bank completed; inspect capture.log')
            if time.time()-started>6*3600:raise TimeoutError('measured bank readiness')
            time.sleep(5)
        status('bank_complete',files=len(expected))
        if not (root/'validation'/'validation.json').exists():
            run('validate_bank',model('eot_validate.py','--bank',root/'bank','--out',root/'validation','--model'))
        assert json.loads((root/'validation'/'validation.json').read_text())['passed']
        # The nominal bank is byte-identical to the previous 500x8 baseline.
        # Reuse that fixed run, not extra optimization or a selected checkpoint.
        baseline=B.PROBE/'out/render_transfer/train_proxy_full'
        config=json.loads((baseline/'config.json').read_text())
        assert config['iters']==500 and config['batch']==8 and config['global_response']
        for name,digest in config['pool_sha256'].items():
            assert hashlib.sha256((root/'bank/origin__nominal'/name).read_bytes()).hexdigest()==digest
        texture_paths={'baseline':str(baseline/'texture_0500.png')}
        for mode in ['joint','position','light']:
            dest=root/('train_'+mode)
            progress=dest/'progress.json'
            if not (progress.exists() and json.loads(progress.read_text())['complete']):
                run('train_'+mode,model('eot_train.py','--bank',root/'bank','--out',dest,'--mode',mode))
            texture_paths[mode]=str(dest/'texture_0500.png')
        from PIL import Image
        from render_geometry import patterns
        Image.fromarray(patterns()['random']).save(root/'random.png')
        texture_paths['random']=str(root/'random.png')
        (root/'textures.json').write_text(json.dumps(texture_paths,indent=2)+'\n')
        # Checkpoints are fixed before examining any new test outcomes.
        status('training_complete_starting_real_render_evaluation')
        meta=root/'server.json'
        if meta.exists():meta.unlink()
        with open(logs/'server.log','a') as log:
            server=subprocess.Popen(model('policy_server.py','--port',args.port,'--metadata',meta),
                cwd=B.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append(server)
            (root/'owned_server.json').write_text(json.dumps({'pid':server.pid,'port':args.port,'stopped':False})+'\n')
            try:
                deadline=time.monotonic()+360
                while not meta.exists():
                    if server.poll() is not None:raise RuntimeError('owned policy server failed')
                    if time.monotonic()>deadline:raise TimeoutError('policy readiness')
                    time.sleep(1)
                for condition in ['nominal','light_dim','light_bright']:
                    run('eval_'+condition+'_clean',sim('eot_rollout.py','--out',root/'eval'/condition/'clean',
                        '--condition',condition,'--port',args.port))
                for condition in evaluation_conditions():
                    for mode in ['random','baseline','joint','position','light']:
                        run('eval_'+condition+'_'+mode,sim('eot_rollout.py','--out',root/'eval'/condition/mode,
                            '--condition',condition,'--texture',texture_paths[mode],'--port',args.port))
                        run('report',model('eot_report.py','--root',root))
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:server.wait(timeout=15)
                    except subprocess.TimeoutExpired:server.kill();server.wait()
                children.remove(server)
                (root/'owned_server.json').write_text(json.dumps({'pid':server.pid,'port':args.port,'stopped':True})+'\n')
        run('final_report',model('eot_report.py','--root',root))
        assert json.loads((root/'summary.json').read_text())['complete']
        status('complete',report=str(root/'results.md'))
    except BaseException:
        status('failed',error=traceback.format_exc());raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()


if __name__=='__main__':main()
