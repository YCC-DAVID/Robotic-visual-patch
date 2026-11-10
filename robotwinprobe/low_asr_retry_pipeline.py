"""Independent full-trajectory retry with strict preflight and natural exits.

CPU: --prepare-only. GPU phases are allocated separately; this process has no
GPU context. A finite policy service exits when its one capture/rollout client
disconnects. No process is sent a termination signal.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import bootstrap as B
from low_asr_retry_common import DEFAULT, TASK, TRAIN_SEEDS, VALIDATION_SEEDS, DATA_SEEDS, prepare, collected_records
from main_method_common import dump, sha
sys.path.insert(0, str(B.ROOT/'pi05probe'))
import visible_pos_pipeline as V
from main_method_job_runtime import install as install_runtime, robotwin_server

ENVROOT = Path(os.environ.get('WAMATTACK_ENVROOT', '/shared/user65/miniconda3/envs'))
MODEL = str(ENVROOT/'openpi-server/bin/python')
SIM = str(ENVROOT/'RoboTwin/bin/python')


def model_env(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], PYTHONNOUSERSITE='1',
                HF_HUB_OFFLINE='1', OPENPI_DATA_HOME='/shared/user65/.cache/openpi',
                JAX_PLATFORMS='cpu', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                WAMATTACK_FIRST_CLIENT_TIMEOUT='600')


def main(args):
    root = args.out.resolve(); V.DEST = root
    protocol = prepare(root)
    for name in ['logs', 'services', 'resources']:
        (root/name).mkdir(exist_ok=True)
    if args.prepare_only:
        print(json.dumps(dict(prepared=True, root=str(root), phases=['capture', 'train', 'rollout'],
              training_seeds=TRAIN_SEEDS, validation_seeds=VALIDATION_SEEDS, checkpoint_training=False, method=protocol['method'],
              gpu_requirement='one idle L40S46GB or larger, min32GBfree, explicit native EGL+SAPIEN UUID audit',
              estimated_gpu_minutes='capture10-25, train18-25, paired30rollout5-10; provisional, no job started'), indent=2))
        return
    gpu = json.loads(args.gpu)
    assert all(k in gpu for k in ['host', 'index', 'uuid', 'egl_index'])
    assert gpu['host'] == socket.gethostname().split('.')[0], 'launch on the allocated GPU host'
    phases = args.phase.split(',')
    assert phases and len(set(phases)) == len(phases) and set(phases) <= {'capture', 'train', 'rollout'}
    install_runtime(V)
    dump(root/'execution.json', dict(gpu=gpu, phases=phases, runtime_source_sha256=sha(B.ROOT/'pi05probe/main_method_job_runtime.py'),
                                  pipeline_source_sha256=sha(__file__), protocol_sha256=sha(root/'protocol.json')))
    # Scene smoke is performed before any fresh GPU collection/training on a new device.
    check = root/'scene_check'/'passed.json'
    if not check.is_file() or json.loads(check.read_text())['binding']['authorized_uuid'] != gpu['uuid']:
        V.stage('scene_check', task=TASK)
        with V.allocation('robotwin_retry_scene_check_'+TASK, gpu, 32000):
            env = dict(model_env(gpu), ROBOTWIN_PYTHON=SIM)
            V.run(['bash', str(B.PROBE/'run_sim.sh'), str(B.PROBE/'main_method_scene_check.py'),
                   '--out', str(root/'scene_check'), '--gpu-uuid', gpu['uuid']], 'scene_check.log', env, 1800)
        assert json.loads(check.read_text())['passed']
    if 'capture' in phases:
        missing = [s for s in DATA_SEEDS if not (root/'collection'/f'seed{s}.json').is_file()]
        if missing:
            V.stage('capture', task=TASK, seeds=DATA_SEEDS)
            with V.allocation('robotwin_retry_capture_'+TASK, gpu, 32000), robotwin_server(root, gpu, args.port, MODEL, model_env, V):
                V.run(['bash', str(B.PROBE/'run_sim.sh'), str(B.PROBE/'low_asr_retry_capture.py'),
                       '--root', str(root), '--port', str(args.port), '--gpu-uuid', gpu['uuid']],
                      'capture.log', dict(model_env(gpu), ROBOTWIN_PYTHON=SIM), 14400)
        _, rows, episodes = collected_records(root)
        dump(root/'collection_manifest.json', dict(complete=True, records=rows, seeds=DATA_SEEDS,
             train_seeds=TRAIN_SEEDS, validation_seeds=VALIDATION_SEEDS,
             no_outcome_filter=True, clean_successes=sum(e['clean_success'] for e in episodes),
             protocol_sha256=sha(root/'protocol.json')))
    if 'train' in phases:
        from low_asr_retry_train import specification, compatible, write_manifest
        config, _ = specification(root)
        if not compatible(root/'train'/TASK, config):
            V.stage('train', task=TASK)
            with V.allocation('robotwin_retry_train_'+TASK, gpu, 32000):
                V.run([MODEL, '-u', str(B.PROBE/'low_asr_retry_train.py'), '--root', str(root)],
                      'train.log', model_env(gpu), 14400)
        write_manifest(root)
    if 'rollout' in phases:
        from low_asr_retry_train import write_manifest
        write_manifest(root)
        completed = root/'eval'/TASK/'complete.json'
        if completed.is_file():
            saved = json.loads(completed.read_text())
            assert saved['complete'] and saved['protocol_sha256'] == sha(root/'protocol.json')
            assert saved['manifest_sha256'] == sha(root/'manifest.json')
            assert json.loads((root/'summary.json').read_text())['complete']
        else:
            V.stage('rollout', task=TASK, episodes=30)
            with V.allocation('robotwin_retry_rollout_'+TASK, gpu, 32000), robotwin_server(root, gpu, args.port, MODEL, model_env, V):
                V.run(['bash', str(B.PROBE/'run_sim.sh'), str(B.PROBE/'main_method_rollout.py'),
                       '--root', str(root), '--port', str(args.port), '--gpu-uuid', gpu['uuid']],
                      'rollout.log', dict(model_env(gpu), ROBOTWIN_PYTHON=SIM), 14400)
            assert json.loads((root/'summary.json').read_text())['complete']
    V.stage('requested_phases_complete', phases=phases, task=TASK)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=DEFAULT)
    ap.add_argument('--gpu', default='{}', help='one allocated {host,index,uuid,egl_index}, exact UUID CVD per child')
    ap.add_argument('--phase', default='capture,train,rollout')
    ap.add_argument('--port', type=int, default=18552)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main(args)
        except BaseException as error:
            V.DEST = args.out.resolve()
            V.stage('blocked' if isinstance(error, V.ResourceBlocked) else 'failed', error=repr(error))
            raise
