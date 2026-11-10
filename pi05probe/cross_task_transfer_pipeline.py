"""Frozen source-task texture transfer to three predeclared LIBERO-goal tasks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'out/cross_task_transfer'
MODEL='/home/user1/miniconda3/envs/openpi-server/bin/python'
SIM='/home/user1/miniconda3/envs/openpi-libero/bin/python'
TASKS=[('bowl_cabinet','put_the_bowl_on_top_of_the_cabinet',2,8164),
       ('bottle_rack','put_the_wine_bottle_on_the_rack',3,8165),
       ('stove','turn_on_the_stove',4,8166)]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    assert not (DEST/'protocol.json').exists(),'refuse to overwrite an experiment'
    textures={'clean':None,
              'random_full':ROOT/'out/dual_view/manifest_both/random.png',
              'full_both':ROOT/'out/dual_view/train_both/texture_0500.png',
              'routed':ROOT/'out/dual_view/train_routed/texture_0500.png',
              'random_block':ROOT/'out/dual_view_both_low/manifest/random.png',
              'block_base':ROOT/'out/dual_view_lowfreq/train_block_base/texture_0500.png'}
    source=json.loads((ROOT/'out/dual_view/manifest_both/manifest.json').read_text())
    manifest={'source_task':'put_the_bowl_on_the_plate','stage':'frozen_cross_task_transfer',
              'geometry':source['geometry'],'textures':{k:None if p is None else str(p) for k,p in textures.items()},
              'texture_sha256':{k:None if p is None else sha(p) for k,p in textures.items()},
              'frequency':'full/routed Q128 and block32 with separate matched random controls'}
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source_task':manifest['source_task'],'tasks':[{'name':n,'stem':t} for n,t,_,_ in TASKS],
              'conditions':list(textures),'indices':list(range(40,50)),
              'target_initial_states':'Each target task own official pruned_init file; identical within six paired conditions',
              'new_episodes':180,'target_training':False,'texture_checkpoint':'fixed source500, no adaptation or selection on target results',
              'geometry':source['geometry'],'environment_seed_rule':'10000+init',
              'action_noise_rule':'20261006+init*1000+replan','max_steps':300,'replan':5,
              'primary_metric':'Task failure among that target task clean-success initial states',
              'controls':'Task-local clean; matched full random for full_both/routed, matched block random for block_base',
              'secondary_metrics':'Task success, completion step, camera-visible patch pixels via object-ID segmentation along each patched rollout',
              'caveats':['Same room/geometry; not cross-scene transfer','Fixed source-selected placement can favor source task',
                         'One source training seed and10initial states per target','Task-specific trained texture differences are not tested in this experiment'],
              'manifest_sha256':sha(DEST/'manifest.json'),
              'source_sha256':{n:sha(ROOT/n) for n in ['cross_task_transfer_rollout.py','cross_task_transfer_pipeline.py',
                                                    'render_proxy_policy_server.py','scene_patch.py','drift_eval_digital.py','drift_resize_order.py']}}
    (DEST/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (DEST/'progress.json').write_text(json.dumps({'stage':'running','planned_new_episodes':180})+'\n')
    def run(name,task,gpu,port):
        folder=DEST/name;folder.mkdir()
        log=(folder/'policy_server.log').open('w')
        server=subprocess.Popen([MODEL,'render_proxy_policy_server.py','--port',str(port)],cwd=ROOT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4'},stdout=log,stderr=subprocess.STDOUT)
        (folder/'owned_service.json').write_text(json.dumps({'pid':server.pid,'port':port,'gpu':gpu})+'\n')
        try:
            start=time.monotonic()
            while '[serve] ready' not in (folder/'policy_server.log').read_text():
                assert server.poll() is None and time.monotonic()-start<600,'server failed to start'
                time.sleep(2)
            with (folder/'rollout.log').open('w') as f:
                subprocess.run([SIM,'cross_task_transfer_rollout.py','--manifest',str(DEST/'manifest.json'),
                    '--task',task,'--conditions',','.join(textures),'--indices','40,41,42,43,44,45,46,47,48,49',
                    '--port',str(port),'--out','cross_task_transfer/'+name+'/rollout'],cwd=ROOT,
                    env={**os.environ,'MUJOCO_EGL_DEVICE_ID':str(gpu),'OMP_NUM_THREADS':'2'},
                    stdout=f,stderr=subprocess.STDOUT,check=True,timeout=4800)
            (folder/'complete.json').write_text(json.dumps({'complete':True,'episodes':60,'elapsed_s':time.monotonic()-start})+'\n')
            print('[done]',name,flush=True)
        finally:
            if server.poll() is None:server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
            log.close();(folder/'service_stopped.json').write_text(json.dumps({'stopped_pid':server.pid})+'\n')
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures=[ex.submit(run,*t) for t in TASKS]
        for f in futures:f.result()
    (DEST/'progress.json').write_text(json.dumps({'stage':'complete','new_episodes':180})+'\n')
    print('[complete] 180 episodes',flush=True)


if __name__=='__main__':main()
