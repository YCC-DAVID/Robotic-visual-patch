"""LIBERO closed-loop image-frequency intervention with exact paired renders.

The advancing environment is always the original scene. A visual-only patch
renderer is synchronized to its complete simulator state at every query. Image
frequency interventions are performed by ``infer_pair`` in the model server;
these are input-domain interventions, not physical-texture realizations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(os.environ.get('WAMATTACK_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / 'pi05probe'))
from checkpoint_study_common import dump, sha
from wam_orthogonal_control import call

CONDITIONS = (
    'native_clean', 'native_attack', 'matched_clean', 'matched_attack',
    'low_attack_high_clean', 'high_attack_low_clean',
)


def digest(array):
    """Hash the exact contiguous values; shapes/dtypes are stored separately."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def rgb_pair(obs):
    return {
        'base': np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]),
        'wrist': np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1]),
    }


def policy_state(obs, quat2axisangle):
    return np.concatenate([
        obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']),
        obs['robot0_gripper_qpos'],
    ]).astype(np.float32)


def small_json(value):
    """Preserve server audit metadata without putting full images in JSON."""
    if isinstance(value, np.ndarray):
        if value.size <= 128:
            return value.tolist()
        return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=digest(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): small_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [small_json(v) for v in value]
    return value


def source_model_root(config, kind):
    source = Path(config['source_root'])
    if not source.is_absolute():
        source = ROOT / source
    if (source / 'models' / kind / 'protocol.json').is_file():
        return source / 'models' / kind
    assert (source / 'protocol.json').is_file(), ('Missing source protocol', source)
    return source


def load_source(root, condition, index):
    path = root / 'episodes' / condition / f'init{index}.json'
    row = json.loads(path.read_text())
    assert 'success' in row and not row.get('error'), ('Invalid source episode', path)
    assert row['protocol_sha256'] == sha(root / 'protocol.json'), path
    artifact = path.with_suffix('.npz')
    with np.load(artifact) as z:
        arrays = {name: z[name].copy() for name in z.files}
    return row, arrays, dict(json=str(path), artifact=str(artifact), artifact_sha256=sha(artifact))


def assert_fixture_prefix(clean_env, patched_env):
    cm, pm = clean_env.env.sim.model, patched_env.env.sim.model
    n = len(cm.body_pos)
    assert len(pm.body_pos) >= n
    assert np.array_equal(cm.body_pos, pm.body_pos[:n]), 'Paired fixture positions differ'
    assert np.array_equal(cm.body_quat, pm.body_quat[:n]), 'Paired fixture orientations differ'


def rollout(root, condition, index):
    root = root.resolve()
    assert condition in CONDITIONS
    p = json.loads((root / 'protocol.json').read_text())
    frequency = p['frequency_restore']
    kind = p.get('model_kind', 'fast')
    kind = 'pi' if kind == 'pi' else 'fast'
    source_condition = frequency.get('source_condition', kind)
    source = source_model_root(frequency, kind)
    cutoff = float(frequency.get('primary_cutoff', frequency.get('cutoff', frequency.get('cutoff_cycles_per_pixel', .125))))
    assert 0 < cutoff <= np.sqrt(.5)
    evaluation = p['evaluation']
    horizon = int(evaluation['horizon'])
    expected = (10, 5, 300, 10) if kind == 'pi' else (32, 10, 400, 30)
    actual = (horizon, int(evaluation['replan']), int(evaluation['steps']), int(evaluation['wait']))
    assert actual == expected, ('Model-specific native evaluation changed', kind, actual, expected)
    assert int(evaluation.get('flow_steps', 10)) == 10
    output = root / 'episodes' / condition / f'init{index}.json'
    if output.exists():
        previous = json.loads(output.read_text())
        assert 'success' in previous and not previous.get('error'), ('Preserved unresolved episode error', output)
        assert previous['protocol_sha256'] == sha(root / 'protocol.json')
        assert previous['artifact_sha256'] == sha(output.with_suffix('.npz'))
        return previous
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_env = patch_env = None
    began = time.monotonic()
    step = -1
    try:
        source_clean, clean_reference, clean_provenance = load_source(source, 'clean', index)
        source_attack, attack_reference, attack_provenance = load_source(source, source_condition, index)
        assert np.array_equal(clean_reference['initial_state'], attack_reference['initial_state'])
        assert np.array_equal(clean_reference['states'][0], attack_reference['states'][0])
        assert source_clean['initial_state_sha256'] == source_attack['initial_state_sha256']

        # Respect the runner's isolated LIBERO config when the legacy helper
        # initializes its imports (it reads WAMATTACK_LIBERO_CONFIG_PATH).
        if 'LIBERO_CONFIG_PATH' in os.environ:
            os.environ.setdefault('WAMATTACK_LIBERO_CONFIG_PATH', os.environ['LIBERO_CONFIG_PATH'])
        import drift_eval_digital as D
        import scene_patch as sp
        import torch
        import yaml
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[p.get('suite', 'libero_goal')]()
        task = next(suite.get_task(i) for i in range(suite.n_tasks)
                    if Path(suite.get_task(i).bddl_file).stem == p['task'])
        bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        init_path = Path(get_libero_path('init_states')) / p.get('suite', 'libero_goal') / (p['task'] + '.pruned_init')
        inits = np.array(torch.load(str(init_path)))
        assert 0 <= index < len(inits)

        cfg = yaml.safe_load((ROOT / 'pi05probe/config/scene.yaml').read_text())
        cfg['patch']['size_wh'] = [float(p['patch']['size_m'])] * 2
        with np.load(p['calibration']) as z:
            world = z['cell_world'].copy()
        texture = Path(evaluation['texture_paths']['native_attack'])
        if not texture.is_absolute():
            texture = root / texture
        assert texture.is_file()
        clean_env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        patch_env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        patch_env.env.set_xml_processor(sp.make_xml_processor(cfg, world, str(texture)))

        # Independent seed-before-reset is essential for fixture model poses,
        # which are not included in the flattened dynamic simulator state.
        seed = 10000 + index
        clean_env.seed(seed)
        clean_env.reset()
        clean_env.set_init_state(inits[index])
        patch_env.seed(seed)
        patch_env.reset()
        patch_env.set_init_state(inits[index])
        assert_fixture_prefix(clean_env, patch_env)
        for _ in range(int(evaluation['wait'])):
            clean_env.step(D.DUMMY)
        initial = clean_env.get_sim_state().copy()
        initial_hash = digest(initial)
        assert np.array_equal(initial, clean_reference['initial_state']), 'Native initial dynamic state changed'
        assert initial_hash == source_clean['initial_state_sha256']
        cm, pm = clean_env.env.sim.model, patch_env.env.sim.model
        assert np.array_equal(cm.body_pos, clean_reference['body_pos'])
        assert np.array_equal(cm.body_quat, clean_reference['body_quat'])
        assert np.array_equal(pm.body_pos, attack_reference['body_pos'])
        assert np.array_equal(pm.body_quat, attack_reference['body_quat'])
        if condition != 'native_clean':
            own_clean_path = root / 'episodes/native_clean' / f'init{index}.json'
            own_clean = json.loads(own_clean_path.read_text())
            assert 'success' in own_clean and not own_clean.get('error'), 'Complete own native_clean first'
            assert own_clean['initial_state_sha256'] == initial_hash
            assert own_clean['protocol_sha256'] == sha(root / 'protocol.json')

        queue = []
        queries = 0
        success = False
        states, sim_states, actions, query_steps, noises = [], [], [], [], []
        rgb_hashes, pair_state_hashes, diagnostics = [], [], []
        first_images = {}
        native_first_audit = None
        prompt = p['prompt_tasks'][0]
        for step in range(int(evaluation['steps'])):
            if not queue:
                same_state = clean_env.get_sim_state().copy()
                clean_obs = clean_env.regenerate_obs_from_state(same_state)
                assert np.array_equal(clean_env.get_sim_state(), same_state), 'Clean render changed dynamics'
                attack_obs = patch_env.regenerate_obs_from_state(same_state)
                assert np.array_equal(patch_env.get_sim_state(), same_state), 'Patch render changed dynamics'
                assert_fixture_prefix(clean_env, patch_env)
                state = policy_state(clean_obs, D.quat2axisangle)
                attack_state = policy_state(attack_obs, D.quat2axisangle)
                assert np.array_equal(state, attack_state), 'Paired observation state differs'
                images_clean, images_attack = rgb_pair(clean_obs), rgb_pair(attack_obs)
                hash_row = [[digest(frame[camera]) for camera in ('base', 'wrist')]
                            for frame in (images_clean, images_attack)]
                noise = 20261006 + index * 1000 + queries
                if queries == 0:
                    for label, frame in [('clean', images_clean), ('attack', images_attack)]:
                        for camera, rgb in frame.items():
                            first_images[f'first_{label}_{camera}'] = rgb.copy()
                    # Both source render identities are checked even for hybrid
                    # modes; native cases additionally check the first action.
                    assert np.array_equal(state, clean_reference['states'][0])
                    for camera in ('base', 'wrist'):
                        assert np.array_equal(images_clean[camera], clean_reference[camera][0]), ('Source clean RGB mismatch', camera)
                        assert np.array_equal(images_attack[camera], attack_reference[camera][0]), ('Source attack RGB mismatch', camera)

                result = call(root, 'infer_pair', state=state, prompt=prompt,
                              images_clean=images_clean, images_attack=images_attack,
                              seed=noise, mode=condition, cutoff=cutoff)
                chunk = np.asarray(result['actions']).copy()
                assert chunk.shape == (horizon, 7) and np.isfinite(chunk).all()
                if kind == 'fast':
                    chunk[:, -1] = np.sign(-(2 * chunk[:, -1] - 1))
                if queries == 0 and condition in ('native_clean', 'native_attack'):
                    reference = clean_reference if condition == 'native_clean' else attack_reference
                    action_error = float(np.max(np.abs(chunk - reference['actions'][0])))
                    assert np.allclose(chunk, reference['actions'][0], rtol=0, atol=1e-6), ('Source native action mismatch', action_error)
                    native_first_audit = dict(passed=True, action_max_abs=action_error,
                                              action_atol=1e-6, exact_state_and_rgb=True)
                states.append(state.copy())
                sim_states.append(same_state)
                actions.append(chunk)
                query_steps.append(step)
                noises.append(noise)
                rgb_hashes.append(hash_row)
                pair_state_hashes.append(digest(same_state))
                diagnostics.append(small_json({k: v for k, v in result.items() if k != 'actions'}))
                queue = list(chunk[:int(evaluation['replan'])])
                queries += 1
            _, _, done, _ = clean_env.step(queue.pop(0))
            if done:
                success = True
                break

        if condition in ('native_clean', 'native_attack'):
            expected_source = source_clean if condition == 'native_clean' else source_attack
            assert success == bool(expected_source['success']), 'Native closed-loop source outcome changed'
        artifact = output.with_suffix('.npz')
        temp = artifact.with_suffix('.tmp.npz')
        np.savez_compressed(temp, states=states, sim_states=sim_states, actions=actions,
                            query_steps=query_steps, noise_seeds=noises, initial_state=initial,
                            body_pos=cm.body_pos.copy(), body_quat=cm.body_quat.copy(),
                            attack_body_pos=pm.body_pos.copy(), attack_body_quat=pm.body_quat.copy(),
                            rgb_sha256=rgb_hashes, pair_state_sha256=pair_state_hashes,
                            **first_images)
        temp.replace(artifact)
        row = dict(init_index=index, condition=condition, success=success,
                   status='complete', steps=step + 1, queries=queries, cutoff=cutoff,
                   protocol_sha256=sha(root / 'protocol.json'),
                   artifact_sha256=sha(artifact), initial_state_sha256=initial_hash,
                   first_render_sha256=dict(clean=dict(zip(('base', 'wrist'), rgb_hashes[0][0])),
                                           attack=dict(zip(('base', 'wrist'), rgb_hashes[0][1]))),
                   paired_full_state_verified_each_query=True,
                   native_first_query_audit=native_first_audit,
                   source_episodes=dict(clean=clean_provenance, attack=attack_provenance),
                   source_texture_sha256=sha(texture), query_diagnostics=diagnostics,
                   scope='Image-domain intervention using paired real MuJoCo renders; not a physically realized hybrid texture',
                   elapsed_s=time.monotonic() - began)
        dump(output, row)
        print('[LIBERO frequency]', kind, index, condition, success, step + 1, flush=True)
        return row
    except Exception as error:
        dump(output, dict(init_index=index, condition=condition, status='runtime_error',
                          protocol_sha256=sha(root / 'protocol.json'), error=repr(error),
                          traceback=traceback.format_exc(), last_step=step,
                          elapsed_s=time.monotonic() - began))
        raise
    finally:
        if patch_env is not None:
            patch_env.close()
        if clean_env is not None:
            clean_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    rollout(args.root, args.condition, args.index)
