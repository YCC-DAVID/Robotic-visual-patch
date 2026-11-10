"""Predeclared 2^3 training and actual-render evaluation, bounded GPU workers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'out/proxy_factorial'
MODEL='/home/user1/miniconda3/envs/openpi-server/bin/python'
SIM='/home/user1/miniconda3/envs/openpi-libero/bin/python'
FIT=ROOT/'out/render_proxy_scale/fit_13_101/fit.json'
JOBS=[(0,8184,['111','000']),(0,8185,['100','011']),
      (1,8186,['010','101']),(1,8187,['001','110'])]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')


def main():
    assert json.loads((DEST/'preflight.json').read_text())['complete']
    assert not (DEST/'protocol.json').exists(), 'preserve experiment runs'
    variants=[v for _,_,jobs in JOBS for v in jobs]
    sources=['proxy_factorial_proxy.py','proxy_factorial_train.py','proxy_factorial_probe.py',
             'proxy_factorial_capture.py','proxy_factorial_check.py','proxy_factorial_pipeline.py',
             'render_world_proxy.py','render_scale_manifest.py','render_scale_rollout.py',
             'render_scale_validation_capture.py','render_proxy_policy_server.py','drift_patch_train.py']
    snapshot=DEST/'source_snapshot';snapshot.mkdir()
    for name in sources:shutil.copy2(ROOT/name,snapshot/name)
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'question':'Are individual imaging components sufficient; does omitting any component remove transfer?',
      'variants':sorted(variants),'factor_order':'MAC',
      'M':'trilinear mipmap versus level-zero bilinear; same calibrated geometry/phase',
      'A':'final256->224 bilinear antialias on/off; NOT renderer MSAA',
      'C':'black+gain calibrated scene photometry versus unlit RGB with independently measured alpha',
      'common':'worldQ128,calibrated perspective,native256,visibility/coverage,uint8STE,base-only training,clean wrist',
      'calibration':str(DEST/'calibration.npz'),'calibration_sha256':sha(DEST/'calibration.npz'),
      'fit':str(FIT),'fit_sha256':sha(FIT),'fit_reestimated_by_variant':False,
      'iterations':500,'batch':8,'lr':.01,'seed_batch':20261005,'noise_seed':20260901,
      'train_episodes':list(range(10)),'validation_episodes':[10,11],
      'test_initial_states':list(range(40,50)),'new_closed_loop_episodes':100,
      'geometry':'13cm at101','checkpoint_selection':'fixed500; no early selection',
      'primary':'paired task failures on freshly rerun clean-success states; shared random Q control',
      'limits':['single train seed/task/size; reused test states, exploratory',
                '000 is native perspective skeleton, not identical to old224-only E25',
                'C includes calibrated boundary/background scene response, not an isolated scalar lighting parameter',
                'does not test removing perspective, uint8STE, or renderer MSAA'],
      'source_sha256':{n:sha(ROOT/n) for n in sources},'workers':JOBS}
    write(DEST/'protocol.json',protocol)
    baseline_ready=threading.Event(); failed=threading.Event()
    def run(argv,log,env):
        with (DEST/'logs'/log).open('w') as output:
            subprocess.run(argv,cwd=ROOT,env={**os.environ,**env},stdout=output,
                stderr=subprocess.STDOUT,check=True)
    def worker(index,job):
        gpu,port,names=job
        worker_dir=DEST/f'worker{index}';worker_dir.mkdir()
        env={'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4'}
        renderenv={'MUJOCO_EGL_DEVICE_ID':str(gpu),'OMP_NUM_THREADS':'2',
                   'NUMBA_CACHE_DIR':f'/tmp/wam-factorial-{index}'}
        state={'gpu':gpu,'port':port,'variants':names}
        def stage(text):
            state['stage']=text;state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            write(worker_dir/'progress.json',state);print('[worker]',index,text,flush=True)
        server=None
        try:
            for variant in names:
                stage('training_'+variant)
                run([MODEL,'proxy_factorial_train.py','--variant',variant,'--calibration',str(DEST/'calibration.npz'),
                     '--fit',str(FIT),'--out',str(DEST/('train_'+variant))],f'train_{variant}.log',env)
                run([MODEL,'render_scale_manifest.py','--run',str(DEST/('train_'+variant)),
                     '--out',str(DEST/('manifest_'+variant))],f'manifest_{variant}.log',env)
            stage('starting_policy')
            with (worker_dir/'policy_server.log').open('w') as log:
                server=subprocess.Popen([MODEL,'render_proxy_policy_server.py','--port',str(port)],
                    cwd=ROOT,env={**os.environ,**env},stdout=log,stderr=subprocess.STDOUT)
            write(worker_dir/'owned_service.json',{'pid':server.pid,'gpu':gpu,'port':port})
            start=time.monotonic()
            while '[serve] ready' not in (worker_dir/'policy_server.log').read_text():
                assert server.poll() is None and time.monotonic()-start<900,'policy startup failed'
                time.sleep(2)
            if index==0:
                stage('clean_and_random_baseline')
                run([SIM,'render_scale_rollout.py','--manifest',str(DEST/'manifest_111/manifest.json'),
                     '--conditions','clean,random','--indices',','.join(map(str,range(40,50))),
                     '--port',str(port),'--out','proxy_factorial/baseline'], 'baseline.log',renderenv)
                baseline_ready.set()
            else:
                stage('waiting_baseline')
                while not baseline_ready.wait(3):
                    assert not failed.is_set(),'another worker failed before baseline completion'
            for variant in names:
                stage('rollout_'+variant)
                run([SIM,'render_scale_rollout.py','--manifest',str(DEST/f'manifest_{variant}/manifest.json'),
                     '--conditions','optimized','--clean-reference',str(DEST/'baseline/episodes.json'),
                     '--indices',','.join(map(str,range(40,50))),'--port',str(port),
                     '--out',f'proxy_factorial/rollout_{variant}'],f'rollout_{variant}.log',renderenv)
                stage('capture_'+variant)
                run([SIM,'render_scale_validation_capture.py','--manifest',str(DEST/f'manifest_{variant}/manifest.json'),
                     '--out',f'proxy_factorial/validation_{variant}'],f'capture_{variant}.log',renderenv)
            server.terminate();server.wait(timeout=30);server=None
            for variant in names:
                stage('model_probe_'+variant)
                run([MODEL,'proxy_factorial_probe.py','--variant',variant,
                     '--out',f'proxy_factorial/validation_{variant}'],f'probe_{variant}.log',env)
            stage('complete')
        except BaseException as error:
            failed.set();stage('failed: '+str(error));raise
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                try:server.wait(timeout=30)
                except subprocess.TimeoutExpired:server.kill();server.wait()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(worker,i,job) for i,job in enumerate(JOBS)]
        for f in futures:f.result()
    write(DEST/'complete.json',{'complete':True,'variants':variants,'new_episodes':100})
    print('[all complete]',flush=True)


if __name__=='__main__':main()
