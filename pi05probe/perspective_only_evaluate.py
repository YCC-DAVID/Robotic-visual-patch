"""Complete the frozen geometry-only run: 20 paired rollouts and 32-state probe."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'out/perspective_only'
MODEL='/home/user1/miniconda3/envs/openpi-server/bin/python'
SIM='/home/user1/miniconda3/envs/openpi-libero/bin/python'


def command(args,log,env=None):
    with (OUT/log).open('w') as f:
        subprocess.run(args,cwd=ROOT,env={**os.environ,**(env or {})},stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    run=OUT/'train_13_101_full'
    started=time.monotonic()
    while True:
        try:
            ready=json.loads((run/'config.json').read_text()).get('complete') is True
        except (FileNotFoundError,json.JSONDecodeError):ready=False
        if ready:break
        assert time.monotonic()-started<2400,'training did not complete within 40 minutes'
        time.sleep(2)
    for stage in ['development','confirmation']:
        command([MODEL,'render_scale_manifest.py','--run',str(run),'--out',str(OUT/('manifest_'+stage)),
                 '--stage',stage],'manifest_'+stage+'.log',{'OMP_NUM_THREADS':'4'})
    def rollout(stage,ids,port,gpu):
        command([SIM,'render_scale_rollout.py','--manifest',str(OUT/('manifest_'+stage)/'manifest.json'),
                 '--conditions','random,optimized','--clean-reference',str(ROOT/'out/render_proxy_scale/clean_baseline/episodes.json'),
                 '--indices',ids,'--port',str(port),'--out','perspective_only/rollout_'+stage],
                'rollout_'+stage+'.log',{'MUJOCO_EGL_DEVICE_ID':str(gpu),'OMP_NUM_THREADS':'2'})
        return stage
    def probe():
        command([SIM,'render_scale_validation_capture.py','--manifest',str(OUT/'manifest_development/manifest.json'),
                 '--out','perspective_only/validation'],'validation_capture.log',{'MUJOCO_EGL_DEVICE_ID':'6','OMP_NUM_THREADS':'2'})
        command([MODEL,'perspective_only_probe.py','--out','perspective_only/validation'],
                'validation_probe.log',{'CUDA_VISIBLE_DEVICES':'6','OMP_NUM_THREADS':'4'})
        return 'validation'
    with ThreadPoolExecutor(max_workers=3) as executor:
        tasks=[executor.submit(rollout,'development','40,41,42,43,44',8154,3),
               executor.submit(rollout,'confirmation','45,46,47,48,49',8155,5),executor.submit(probe)]
        for future in tasks:print('[finished]',future.result(),flush=True)
    (OUT/'evaluation_complete.json').write_text(json.dumps({'complete':True,'new_rollouts':20,'unique_initial_states':10,
        'validation_states':32,'validation_episodes':2,'notes':'Paired exploratory retest on previously used40-49. No newly blind confirmation claim.'},indent=2)+'\n')


if __name__=='__main__':main()
