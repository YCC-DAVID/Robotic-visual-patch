"""One GPU worker: three closed-loop conditions, paired captures, model probe.

The subprocess model server belongs to this worker and is always stopped.
No external/shared inference services or environments are modified.
"""
import argparse,json,os,signal,subprocess,sys,time
import bootstrap as B
B.simulator_paths()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['full','band32'],required=True)
    ap.add_argument('--port',type=int,required=True);ap.add_argument('--root',required=True)
    args=ap.parse_args();root=B.Path(args.root).resolve()
    import render_rollout,render_capture
    variant=args.variant;work=root/f'worker_{variant}';work.mkdir(exist_ok=True)
    names=[f'digital_{variant}',f'proxy_{variant}']
    textures={n:str(root/f'train_{n}/texture_0500.png') for n in names}
    for n in names:
        progress=json.loads((root/f'train_{n}/progress.json').read_text())
        assert progress['complete'] and progress['iteration']==500
    manifest=work/'textures.json';manifest.write_text(json.dumps(textures,indent=2)+'\n')
    model_python='/home/user1/miniconda3/envs/openpi-server/bin/python'
    env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONNOUSERSITE']='1'
    env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    meta=work/'server.json'
    if meta.exists():meta.unlink()
    log=open(work/'server.log','a')
    server=subprocess.Popen([model_python,'-u',str(B.PROBE/'policy_server.py'),'--port',str(args.port),
                             '--metadata',str(meta)],cwd=B.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    (work/'owned_server.json').write_text(json.dumps({'pid':server.pid,'port':args.port,'stopped':False})+'\n')
    def stop_signal(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
    signal.signal(signal.SIGTERM,stop_signal)
    try:
        deadline=time.monotonic()+360
        while not meta.exists():
            if server.poll() is not None:raise RuntimeError(f'policy server exited {server.returncode}')
            if time.monotonic()>deadline:raise TimeoutError('policy server readiness')
            time.sleep(1)
        for name,condition in [(names[0],'digital'),(names[0],'render'),(names[1],'render')]:
            out=root/f'{name}_{condition}'
            sys.argv=['render_rollout.py','--texture',textures[name],'--condition',condition,
                      '--out',str(out),'--port',str(args.port),'--resume']
            print('[stage]',name,condition,flush=True);render_rollout.main()
    finally:
        if server.poll() is None:
            server.terminate()
            try:server.wait(timeout=15)
            except subprocess.TimeoutExpired:server.kill();server.wait()
        log.close()
        (work/'owned_server.json').write_text(json.dumps({'pid':server.pid,'port':args.port,'stopped':True,'returncode':server.returncode})+'\n')
    capture=root/f'capture_{variant}'
    missing=[s for s in [10110,10111] if not (capture/f'seed{s}.npz').exists()]
    if missing:
        sys.argv=['render_capture.py','--clean',str(root/'clean'),'--out',str(capture),
                  '--seeds',','.join(map(str,missing)),'--textures',str(manifest)]
        print('[stage] paired validation render',flush=True);render_capture.main()
    import numpy as np
    from render_geometry import CAMERAS
    for seed in [10110,10111]:
        with np.load(capture/f'seed{seed}.npz') as actual,np.load(root/'calibration'/f'seed{seed}.npz') as reference:
            assert np.array_equal(actual['state14'],reference['state14'])
            for name in CAMERAS:
                for suffix in ['clean','mask','intrinsic','extrinsic']:
                    key=name+'_'+suffix
                    assert np.array_equal(actual[key],reference[key]),(seed,key,'paired capture mismatch')
    (work/'capture_audit.json').write_text(json.dumps({'state_RGB_masks_cameras_equal':True,'seeds':[10110,10111]})+'\n')
    probe=root/f'probe_{variant}'
    if not (probe/'metrics.json').exists():
        with open(work/'probe.log','w') as output:
            subprocess.run([model_python,'-u',str(B.PROBE/'render_probe.py'),
                '--pool',str(root/'calibration'),'--out',str(probe),'--capture',str(capture),
                '--textures',str(manifest),'--global-response'],cwd=B.ROOT,env=env,
                stdout=output,stderr=subprocess.STDOUT,check=True)
    (work/'complete.json').write_text(json.dumps({'complete':True,'conditions':names,'server_stopped':True})+'\n')
    print('[worker complete]',variant,flush=True)


if __name__=='__main__':main()
