"""Bounded E26 orchestration; gate before training; own local services cleaned up."""
from concurrent.futures import ThreadPoolExecutor
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'out/dual_view'
MODEL='/home/user1/miniconda3/envs/openpi-server/bin/python'
SIM='/home/user1/miniconda3/envs/openpi-libero/bin/python'


def command(args,log,env=None,timeout=5400):
    with (DEST/log).open('w') as f:subprocess.run(args,cwd=ROOT,env={**os.environ,**(env or {})},stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)


def state(stage,**kwargs):
    (DEST/'progress.json').write_text(json.dumps({'stage':stage,**kwargs},indent=2)+'\n');print('[stage]',stage,kwargs,flush=True)


def main():
    started=time.monotonic()
    while not (DEST/'fit/checks.json').exists():
        assert time.monotonic()-started<2400,'calibration fit timeout'
        time.sleep(2)
    state('pre_training_fidelity_probe')
    command([MODEL,'dual_view_probe.py','--out','dual_view/preflight'],'preflight.log',{'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'4'})
    summary=json.loads((DEST/'preflight/summary.json').read_text())
    rows={k:v for k,v in summary.items() if k.endswith('/proxy')}
    passed=all(v['KV_relative_render']<.10 and v['paired_energy_ratio']<.10 for v in rows.values())
    (DEST/'fidelity_gate.json').write_text(json.dumps({'passed':passed,'rows':rows,'thresholds':{'KV':.10,'relative_velocity_energy':.10}},indent=2)+'\n')
    assert passed,'Imaging fidelity gate failed: no training launched. Inspect fidelity_gate.json.'
    state('training_three_modes')
    def train(mode,gpu):
        command([MODEL,'dual_view_train.py','--calibration',str(DEST/'calibration/base.npz'),'--fit',str(DEST/'fit/base.json'),
                 '--wrist-calibration',str(DEST/'calibration/wrist.npz'),'--wrist-fit',str(DEST/'fit/wrist.json'),
                 '--camera-mode',mode,'--out',str(DEST/('train_'+mode))],'train_'+mode+'.log',{'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4'})
        return mode
    with ThreadPoolExecutor(max_workers=3) as ex:
        fs=[ex.submit(train,m,g) for m,g in zip(['base','both','routed'],[4,5,6])]
        for f in fs:print('[trained]',f.result(),flush=True)
    for mode in ['base','both','routed']:
        command([MODEL,'render_scale_manifest.py','--run',str(DEST/('train_'+mode)),'--out',str(DEST/('manifest_'+mode))], 'manifest_'+mode+'.log')
    state('true_render_evaluation')
    servers=[];logs=[]
    try:
        for port,gpu in [(8156,4),(8157,5),(8158,6),(8159,7)]:
            log=(DEST/('server_'+str(port)+'.log')).open('w');logs.append(log)
            proc=subprocess.Popen([MODEL,'render_proxy_policy_server.py','--port',str(port)],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4'},stdout=log,stderr=subprocess.STDOUT);servers.append(proc)
        (DEST/'owned_services.json').write_text(json.dumps([{'pid':p.pid,'port':port} for p,port in zip(servers,[8156,8157,8158,8159])],indent=2)+'\n')
        wait=time.monotonic()
        while True:
            assert all(p.poll() is None for p in servers),'policy service stopped before ready'
            if all('[serve] ready' in (DEST/('server_'+str(port)+'.log')).read_text() for port in [8156,8157,8158,8159]):break
            assert time.monotonic()-wait<600,'service startup timeout'
            time.sleep(2)
        def rollout(name,mode,condition,port,gpu):
            command([SIM,'render_scale_rollout.py','--manifest',str(DEST/('manifest_'+mode)/'manifest.json'),
                     '--conditions',condition,'--clean-reference',str(ROOT/'out/render_proxy_scale/clean_baseline/episodes.json'),
                     '--indices','40,41,42,43,44,45,46,47,48,49','--port',str(port),'--out','dual_view/rollout_'+name],
                    'rollout_'+name+'.log',{'MUJOCO_EGL_DEVICE_ID':str(gpu),'OMP_NUM_THREADS':'2'})
            return name
        def validation():
            command([SIM,'dual_view_validation_capture.py'],'validation_capture.log',{'MUJOCO_EGL_DEVICE_ID':'3','OMP_NUM_THREADS':'2'})
            command([MODEL,'dual_view_probe.py','--out','dual_view/validation'],'validation_probe.log',{'CUDA_VISIBLE_DEVICES':'3','OMP_NUM_THREADS':'4'})
            return 'validation'
        with ThreadPoolExecutor(max_workers=5) as ex:
            fs=[ex.submit(rollout,m,m,'optimized',port,gpu) for m,port,gpu in zip(['base','both','routed'],[8156,8157,8158],[4,5,6])]
            fs+=[ex.submit(rollout,'random','base','random',8159,7),ex.submit(validation)]
            for f in fs:print('[evaluated]',f.result(),flush=True)
        state('complete',elapsed_s=time.monotonic()-started,new_rollouts=40,unique_test_states=10)
    finally:
        for p in servers:
            if p.poll() is None:p.terminate()
        for p in servers:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
        for f in logs:f.close()
        (DEST/'services_stopped.json').write_text(json.dumps({'stopped_pids':[p.pid for p in servers]},indent=2)+'\n')

if __name__=='__main__':main()
