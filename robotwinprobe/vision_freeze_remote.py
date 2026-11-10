"""Move only this paired study to nnmc63; mirror status and return for rollout."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import traceback
import checkpoint_study_common as S

ROOT=S.ROOT/'robotwinprobe/out/vision_freeze_ablation_20261025'
HOST='nnmc63'
GPUS='4,5'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST]


def remote(cmd):
    return subprocess.check_output(SSH+[cmd],text=True)


def sync(source,target):
    subprocess.run(['rsync','-aL','--partial','--exclude=__pycache__','--exclude=.git',
                    str(source),str(target)],check=True)


def report():
    subprocess.run([S.PY_CPU,str(S.ROOT/'robotwinprobe/vision_freeze_report.py'),
                    '--root',str(ROOT)],check=True,cwd=S.ROOT)


def stage(name,**extra):
    S.dump(ROOT/'pipeline_status.json',dict(stage=name,complete=False,
        execution_host=HOST,supervisor_pid=os.getpid(),updated_unix=time.time(),**extra))
    report()


def setup():
    stage('syncing_to_nnmc63',gpus=GPUS.split(','))
    staging=ROOT/'remote_code_stage';staging.mkdir(exist_ok=True)
    for p in (S.ROOT/'robotwinprobe').glob('*.py'):shutil.copy2(p,staging/p.name)
    sync(str(staging)+'/',f'{HOST}:{S.ROOT}/robotwinprobe/')
    for name in ['g1_simple_pick_ft.py','g1_simple_pick_ft_data.py']:
        sync(S.ROOT/'realprobe'/name,f'{HOST}:{S.ROOT}/realprobe/{name}')
    remote('mkdir -p '+shlex.quote(str(S.SERVER.parent)))
    sync(str(S.SERVER)+'/',f'{HOST}:{S.SERVER}/')
    sync(str(S.ROOT/'realprobe/runtime_py')+'/',f'{HOST}:{S.ROOT}/realprobe/runtime_py/')
    remote('mkdir -p '+shlex.quote(str(ROOT/'expert'))+' '+shlex.quote(str(ROOT/'logs')))
    for name in ['data_cache','data_smoke','assets','smoke_assets']:
        sync(str(ROOT/name)+'/',f'{HOST}:{ROOT}/{name}/')
    for name in ['protocol.json','collector_process.json','initialization_v2_check.json',
                 'actual_model_schema_audit.json','functional_step_check.json',
                 'representation_interface_check.json']:
        if (ROOT/name).exists():sync(ROOT/name,f'{HOST}:{ROOT}/{name}')
    for name in ['train_manifest.json','validation_manifest.json']:
        sync(ROOT/'expert'/name,f'{HOST}:{ROOT}/expert/{name}')
    sync(str(ROOT/'data_audit')+'/',f'{HOST}:{ROOT}/data_audit/')
    stage('verifying_remote_inputs',gpus=GPUS.split(','))
    # Hash all base checkpoint files: reuse the remote cache only if identical.
    hash_code="""import hashlib,json
from pathlib import Path
root=Path('/home/user1/.cache/openpi/openpi-assets/checkpoints/pi05_base/params')
result={}
for p in sorted(root.rglob('*')):
 if p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  result[str(p.relative_to(root))]=h.hexdigest()
print(json.dumps(result))
"""
    local_hashes=json.loads(subprocess.check_output([S.PY_CPU,'-c',hash_code],text=True))
    remote_hashes=json.loads(remote(shlex.join([S.PY_CPU,'-c',hash_code])))
    assert local_hashes==remote_hashes,'Remote official base differs; do not train'
    # rsync's checksum dry run checks local/remote data bytes, without rewriting.
    for name in ['data_cache','data_smoke','assets','smoke_assets']:
        diff=subprocess.check_output(['rsync','-aLcn','--out-format=%n','--exclude=__pycache__',
            str(ROOT/name)+'/',f'{HOST}:{ROOT}/{name}/'],text=True)
        assert not diff.strip(),f'Remote data mismatch: {name}: {diff}'
    S.dump(ROOT/'remote_input_audit.json',dict(passed=True,host=HOST,
        official_base_files=len(local_hashes),base_byte_identical=True,data_byte_identical=True))


def launch():
    launch_code=f"""import subprocess,json,os
from pathlib import Path
r=Path({str(ROOT)!r})
with (r/'logs/remote_pipeline.log').open('a') as log:
 p=subprocess.Popen([{S.PY_CPU!r},'-u','robotwinprobe/vision_freeze_pipeline.py','--root',str(r),'--gpus',{GPUS!r},'--train-only'],cwd={str(S.ROOT)!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,host='nnmc63',gpus={GPUS.split(',')!r})
(r/'remote_process.json').write_text(json.dumps(record))
print(json.dumps(record))
"""
    info=json.loads(remote(shlex.join([S.PY_CPU,'-c',launch_code])))
    S.dump(ROOT/'remote_process.json',info)
    return info


def mirror(info):
    for name in ['training','logs']:
        sync(f'{HOST}:{ROOT}/{name}/',str(ROOT/name)+'/')
    for name in ['preflight_v2_paired_audit.json','preflight_v3_paired_audit.json','preflight_v4_paired_audit.json','preflight_v5_paired_audit.json','paired_audit.json']:
        if remote('test -f '+shlex.quote(str(ROOT/name))+' && echo yes || true').strip():
            sync(f'{HOST}:{ROOT}/{name}',ROOT/name)
    status=json.loads(remote('cat '+shlex.quote(str(ROOT/'pipeline_status.json'))))
    status.update(execution_host=HOST,remote_pid=info['pid'],supervisor_pid=os.getpid(),mirrored_at=time.time())
    S.dump(ROOT/'pipeline_status.json',status);report()
    return status


def main():
    with (ROOT/'remote_supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'remote_process.json').exists(),'Remote job already recorded; inspect instead of duplicating'
        audit=ROOT/'remote_input_audit.json'
        if not audit.exists() or not json.loads(audit.read_text()).get('passed'):setup()
        info=launch()
        while True:
            time.sleep(30)
            status=mirror(info)
            if status['stage']=='failed':return
            if status['stage']=='model_training_complete':break
            health=remote(shlex.join([S.PY_CPU,'-c',
                f"from pathlib import Path; p=Path('/proc/{info['pid']}/cmdline'); print(p.exists() and b'vision_freeze_pipeline.py' in p.read_bytes())"]))
            if health.strip()!='True':raise RuntimeError('Remote pipeline exited without a terminal status')
        stage('fetching_trained_models',gpus=GPUS.split(','))
        for name in ['F','U']:
            source=ROOT/'checkpoints/pi05_robotwin_vision_ablation'/name/'20000'
            source.mkdir(parents=True,exist_ok=True)
            sync(f'{HOST}:{source}/',str(source)+'/')
        # The local pipeline skips verified model training and executes export,
        # parity, clean pilots, patch optimization, and real-render cross tests.
        with (ROOT/'logs/pipeline_after_remote.log').open('a') as log:
            p=subprocess.Popen([S.PY_CPU,'-u','robotwinprobe/vision_freeze_pipeline.py',
                '--root',str(ROOT),'--gpus','0,1'],cwd=S.ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        S.dump(ROOT/'local_continuation_process.json',dict(pid=p.pid))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        S.dump(ROOT/'remote_supervisor_error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        stage('remote_supervisor_failed',error=repr(exc))
        raise
