"""Wait for fixed500 training, evaluate one new physicalQ and clean-state diagnostics."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,os,time,subprocess,hashlib
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'out/dual_view_both_low'
MODEL='/home/user1/miniconda3/envs/openpi-server/bin/python'
SIM='/home/user1/miniconda3/envs/openpi-libero/bin/python'


def command(args,log,env=None):
    with (DEST/log).open('w') as f:subprocess.run(args,cwd=ROOT,env={**os.environ,**(env or {})},stdout=f,stderr=subprocess.STDOUT,check=True,timeout=3600)


def state(name,**kwargs):
    (DEST/'progress.json').write_text(json.dumps({'stage':name,**kwargs},indent=2)+'\n');print('[stage]',name,flush=True)


def main():
    state('training');started=time.monotonic()
    while True:
        try:ready=json.loads((DEST/'train/config.json').read_text())['complete']
        except (FileNotFoundError,json.JSONDecodeError):ready=False
        if ready:break
        assert time.monotonic()-started<2400,'training timeout, inspect train.log'
        time.sleep(2)
    command([MODEL,'dual_view_both_low_manifest.py','--run',str(DEST/'train'),'--out',str(DEST/'manifest')],'manifest.log')
    state('true_render_evaluation')
    log=(DEST/'policy_server.log').open('w')
    server=subprocess.Popen([MODEL,'render_proxy_policy_server.py','--port','8160'],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'3','OMP_NUM_THREADS':'4'},stdout=log,stderr=subprocess.STDOUT)
    (DEST/'owned_service.json').write_text(json.dumps({'pid':server.pid,'port':8160},indent=2)+'\n')
    try:
        wait=time.monotonic()
        while '[serve] ready' not in (DEST/'policy_server.log').read_text():
            assert server.poll() is None and time.monotonic()-wait<600,'server startup failed'
            time.sleep(2)
        def rollout():
            command([SIM,'render_scale_rollout.py','--manifest',str(DEST/'manifest/manifest.json'),
                     '--conditions','random,optimized','--clean-reference',str(ROOT/'out/render_proxy_scale/clean_baseline/episodes.json'),
                     '--indices','40,41,42,43,44,45,46,47,48,49','--port','8160','--out','dual_view_both_low/rollout'],
                    'rollout.log',{'MUJOCO_EGL_DEVICE_ID':'3','OMP_NUM_THREADS':'2'})
        def validation():
            command([SIM,'dual_view_both_low_capture.py'],'validation_capture.log',{'MUJOCO_EGL_DEVICE_ID':'4','OMP_NUM_THREADS':'2'})
            command([MODEL,'dual_view_probe.py','--out','dual_view_both_low/validation'],'validation_probe.log',{'CUDA_VISIBLE_DEVICES':'4','OMP_NUM_THREADS':'4'})
        with ThreadPoolExecutor(max_workers=2) as ex:
            fs=[ex.submit(rollout),ex.submit(validation)]
            for f in fs:f.result()
        state('complete',new_rollouts=20,unique_test_states=10,elapsed_s=time.monotonic()-started)
    finally:
        if server.poll() is None:server.terminate()
        try:server.wait(timeout=30)
        except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
        log.close();(DEST/'service_stopped.json').write_text(json.dumps({'stopped_pid':server.pid})+'\n')

if __name__=='__main__':main()
