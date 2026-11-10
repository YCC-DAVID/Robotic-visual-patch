"""CPU-only resource queue for the independent pi0 DRIFT experiment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from drift_pi0_common import ROOT, dump, now, read, sha
from drift_pi0_pipeline import protocol


def main(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root / 'launch.json').exists(), 'a controller has already been launched'
        checkpoint = Path(args.checkpoint)
        manifest = read(checkpoint / 'download_manifest.json')
        assert manifest['source'] == 'gs://openpi-assets/checkpoints/pi0_libero'
        assert manifest['source_md5_verified'] and manifest['generation_pinned']
        for row in manifest['files']:
            assert (checkpoint / row['path']).stat().st_size == row['bytes']
        dump(root / 'queue_progress.json', dict(stage='freezing_protocol', pid=os.getpid(), updated=now()))
        if not (root / 'protocol.json').exists():
            dump(root / 'protocol.json', protocol(SimpleNamespace(checkpoint=str(checkpoint))))
        frozen = read(root / 'protocol.json')
        for name, digest in frozen['source_sha256'].items():
            assert sha(ROOT / 'pi05probe' / name) == digest, 'source changed after freeze'
        snapshot = root / 'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        for name in ['common', 'model', 'sim', 'pipeline', 'validate', 'finalize', 'queue', 'fetch']:
            p = ROOT / 'pi05probe' / ('drift_pi0_' + name + '.py')
            shutil.copy2(p, snapshot / p.name)
        resources = root / 'resources'
        resources.mkdir(exist_ok=True)
        while True:
            check = subprocess.run([sys.executable, str(ROOT / 'skills/gpu-preflight/scripts/check_gpus.py'),
                '--json', '--min-free-mib', '24000'], capture_output=True, text=True, timeout=100)
            if check.returncode not in (0, 2):
                dump(root / 'queue_progress.json', dict(stage='waiting_for_preflight', pid=os.getpid(),
                     updated=now(), error=check.stderr[-1500:]))
                time.sleep(30)
                continue
            report = json.loads(check.stdout)
            dump(resources / 'queue_latest_fleet.json', report)
            candidates = []
            for node in report['nodes']:
                if node.get('status') != 'ok':
                    continue
                for gpu in node['gpus']:
                    if (gpu.get('eligible') and gpu.get('idle') and not gpu.get('faulty')
                            and ('L40S' in gpu['name'] or 'H100' in gpu['name'])
                            and not (node['host'] == 'nnmc63' and gpu['index'] == 0)):
                        candidates.append(dict(gpu, host=node['host']))
            candidates.sort(key=lambda g: ('L40S' not in g['name'], g['host'], g['index']))
            if not candidates:
                dump(root / 'queue_progress.json', dict(stage='waiting_for_idle_gpu', pid=os.getpid(),
                    updated=now(), min_free_mib=24000, unknown_nodes=[n['host'] for n in report['nodes'] if n.get('status') != 'ok'],
                    note='No idle eligible card. Existing pi0.5 and other tasks are untouched; no stop operations.'))
                time.sleep(30)
                continue
            launched = False
            for gpu in candidates:
                audit = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', gpu['host'],
                    'env', 'CUDA_VISIBLE_DEVICES=' + gpu['uuid'],
                    '/shared/user65/miniconda3/envs/openpi-server/bin/python', str(ROOT / 'pi05probe/egl_device_audit.py'),
                    '--uuid', gpu['uuid']], capture_output=True, text=True, timeout=40)
                if audit.returncode:
                    continue
                mapping = json.loads(audit.stdout)
                assigned = {k: gpu[k] for k in ['host', 'index', 'uuid', 'name']}
                assigned['egl_index'] = mapping['egl_index']
                payload = dict(project=str(ROOT), root=str(root), checkpoint=str(checkpoint), gpu=assigned)
                script = 'payload = ' + repr(payload) + '\n' + REMOTE_LAUNCH
                launch = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', gpu['host'], 'python3', '-'],
                    input=script, capture_output=True, text=True, timeout=40)
                if launch.returncode:
                    dump(root / 'queue_progress.json', dict(stage='launch_retry', pid=os.getpid(), updated=now(), error=launch.stderr[-2000:]))
                    continue
                value = json.loads(launch.stdout)
                dump(root / 'queue_progress.json', dict(stage='controller_launched', pid=os.getpid(), updated=now(), launch=value))
                print(json.dumps(value), flush=True)
                launched = True
                break
            if launched:
                return
            time.sleep(30)


REMOTE_LAUNCH = r'''
from pathlib import Path
import fcntl,json,os,subprocess,datetime
project=Path(payload['project']);root=Path(payload['root']);gpu=payload['gpu']
assert os.uname().nodename.split('.')[0]==gpu['host']
with (root/'pipeline.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'launch.json').exists()
cmd=['/shared/user65/miniconda3/envs/openpi-server/bin/python','-u',str(project/'pi05probe/drift_pi0_pipeline.py'),'--root',str(root),'--checkpoint',payload['checkpoint'],'--gpu',json.dumps(gpu)]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',WAMATTACK_PROJECT_ROOT=str(project),WAMATTACK_ENVROOT='/shared/user65/miniconda3/envs',PYTHONNOUSERSITE='1')
with (root/'controller.log').open('w') as stream:
    p=subprocess.Popen(cmd,cwd=project/'pi05probe',env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
value=dict(pid=p.pid,host=gpu['host'],gpu=gpu,command=cmd,started=datetime.datetime.now(datetime.timezone.utc).isoformat())
(root/'launch.json').write_text(json.dumps(value,indent=2)+'\n')
final=['/shared/user65/miniconda3/envs/openpi-server/bin/python','-u',str(project/'pi05probe/drift_pi0_finalize.py'),'--root',str(root),'--controller-pid',str(p.pid)]
with (root/'finalization.log').open('w') as stream:
    f=subprocess.Popen(final,cwd=project/'pi05probe',env=dict(env,JAX_PLATFORMS='cpu'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
value['finalizer_pid']=f.pid
(root/'finalization_launch.json').write_text(json.dumps(dict(pid=f.pid,command=final),indent=2)+'\n')
print(json.dumps(value))
'''


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--checkpoint', default='/shared/user65/.cache/openpi/openpi-assets/checkpoints/pi0_libero')
    args = p.parse_args()
    try:
        main(args)
    except BaseException as error:
        dump(Path(args.root) / 'queue_progress.json', dict(stage='failed', pid=os.getpid(), updated=now(), error=repr(error)))
        raise
