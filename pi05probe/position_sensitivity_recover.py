"""Preserve aborted origin record; retry same seed and resume skipped positions."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
import numpy as np
import position_sensitivity_pipeline as P


def main():
    dest = P.DEST
    service = dest/'recovery'; service.mkdir()
    P.write(service/'protocol.json', {
        'reason': 'origin init49 distance aborted before inference: diagnostic repeat RGB equality assertion',
        'retry': 'fresh process, same simulator settings, init49, random+distance; retain original19completed records',
        'script_change': 'only save before/after RGB if same assertion fires; no computation or threshold change',
        'rollout_sha256': P.sha(P.ROOT/'position_sensitivity_retry_rollout.py'),
        'recover_script_sha256': P.sha(__file__),
        'pending_positions': ['y_minus1cm', 'x_plus3cm']})
    for name in ['position_sensitivity_retry_rollout.py', 'position_sensitivity_recover.py']:
        shutil.copy2(P.ROOT/name, service/name)
    with (service/'policy_server.log').open('w') as log:
        server = subprocess.Popen([P.MODEL, 'render_proxy_policy_server.py', '--port', '8172'], cwd=P.ROOT,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '2', 'OMP_NUM_THREADS': '4'}, stdout=log, stderr=subprocess.STDOUT)
        P.write(service/'owned_service.json', {'pid': server.pid, 'port': 8172, 'gpu': 2})
        try:
            start = time.monotonic()
            while '[serve] ready' not in (service/'policy_server.log').read_text():
                assert server.poll() is None and time.monotonic()-start < 900
                time.sleep(2)
            def run(name, output, indices, script):
                with (service/(output.replace('/', '_')+'.log')).open('w') as f:
                    subprocess.run([P.SIM, script, '--manifest', str(dest/name/'manifest.json'), '--task', P.TASK,
                        '--clean-reference', str(P.ROOT/'out/render_proxy_scale/clean_baseline/episodes.json'),
                        '--conditions', 'random,distance', '--indices', indices, '--port', '8172',
                        '--out', 'position_sensitivity/'+output], cwd=P.ROOT,
                        env={**os.environ, 'MUJOCO_EGL_DEVICE_ID': '2', 'OMP_NUM_THREADS': '2'},
                        stdout=f, stderr=subprocess.STDOUT, check=True, timeout=7200)
            run('origin', 'recovery/origin49', '49', 'position_sensitivity_retry_rollout.py')
            original = dest/'origin/rollout/episodes.json'
            old = json.loads(original.read_text())
            retry = json.loads((service/'origin49/episodes.json').read_text())
            assert len(old['per_episode']['distance']) == 9
            for condition, reference_path in [('random', original), ('distance', P.ROOT/'out/direction_loss/rollout/episodes.json')]:
                previous = json.loads(reference_path.read_text())
                oldrow = next(r for r in previous['per_episode'][condition] if r['init_index'] == 49)
                row = retry['per_episode'][condition][0]
                assert row['success'] == oldrow['success'] and row['success_step'] == oldrow['success_step']
                with np.load(service/'origin49'/row['replay_record']) as a, np.load(reference_path.parent/oldrow['replay_record']) as b:
                    for key in ['actions', 'sim_states', 'noise_seeds', 'base224', 'wrist224', 'initial_state']:
                        assert np.array_equal(a[key], b[key]), (condition, key)
            shutil.copy2(original, service/'origin_partial_19_episodes.json')
            row = retry['per_episode']['distance'][0]
            shutil.copy2(service/'origin49'/row['replay_record'], original.parent/row['replay_record'])
            old['per_episode']['distance'].append(row)
            rows = old['per_episode']['distance']
            good = {r['init_index'] for r in old['per_episode']['clean'] if r['success']}
            failures = sum(not r['success'] for r in rows if r['init_index'] in good)
            old['summary']['distance'] = {'n': 10, 'successes': sum(r['success'] for r in rows),
                'robot_success_rate': sum(r['success'] for r in rows)/10,
                'paired_ASR_clean_success_n': len(good), 'paired_ASR_failures': failures, 'paired_ASR': failures/len(good)}
            old['metadata']['recovery'] = 'init49 distance retried with identical protocol after RGB diagnostic assertion; see recovery/protocol.json'
            P.write(original, old)
            P.write(dest/'origin/complete.json', {'complete': True, 'new_episodes': 20,
                'recovered_one_episode': True, 'extra_random_repeat': 1, 'retry_matches_historical_record_exactly': True})
            print('[recovered] origin49 exact, origin20complete', flush=True)
            for name in ['y_minus1cm', 'x_plus3cm']:
                started = time.monotonic()
                run(name, name+'/rollout', ','.join(map(str, range(40, 50))), 'position_sensitivity_retry_rollout.py')
                P.write(dest/name/'complete.json', {'complete': True, 'new_episodes': 20, 'elapsed_s': time.monotonic()-started})
                print('[done]', name, flush=True)
            P.write(service/'complete.json', {'complete': True, 'additional_random_repeat': 1})
        finally:
            if server.poll() is None: server.terminate()
            try: server.wait(timeout=30)
            except subprocess.TimeoutExpired: server.kill(); server.wait(timeout=10)
            P.write(service/'service_stopped.json', {'pid': server.pid, 'stopped': True})


if __name__ == '__main__':
    main()
