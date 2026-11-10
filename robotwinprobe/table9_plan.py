"""Nine independent 10cm textures; shared states, joint EOT draws and noise bank."""
import hashlib
import json
from pathlib import Path

import numpy as np
import bootstrap as B
from radius_plan import disk, dump, REFERENCE
from eot_conditions import LIGHTS

DEFAULT = B.OUT / 'table9_joint_eot_10cm_20261012'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    root = Path(root).resolve()
    selection = json.loads((root / 'geometry/selection.json').read_text())
    audit = json.loads((root / 'geometry/verification.json').read_text())
    assert selection['feasible'] and audit['passed'], 'Do not train until geometry / exposure checks pass'
    if (root / 'protocol.json').exists():
        protocol = json.loads((root / 'protocol.json').read_text())
        assert protocol['selection_sha256'] == sha(root / 'geometry/selection.json')
        assert protocol['geometry_verification_sha256'] == sha(root / 'geometry/verification.json')
        assert protocol['plan_sha256'] == sha(root / 'plan.npz')
        return protocol
    assert not (root / 'plan.npz').exists(), 'Incomplete plan; inspect rather than overwrite'
    frame_seeds, frame_queries = [], []
    paths = sorted((REFERENCE / 'calibration').glob('seed*.npz'))
    for path in paths:
        with np.load(path) as z:
            frame_seeds.extend([int(z['seed'])]*len(z['query']))
            frame_queries.extend(map(int, z['query']))
    frame_seeds = np.array(frame_seeds)
    common = {(f['seed'],f['query']) for f in selection['training_common_frames']}
    train = np.array([i for i,(s,q) in enumerate(zip(frame_seeds,frame_queries)) if (int(s),int(q)) in common])
    assert len(train)==len(common) and set(frame_seeds[train])==set(range(10100,10108))
    val = np.flatnonzero(frame_seeds >= 10110)
    rng = np.random.RandomState(20261007)
    batches = np.array([rng.choice(train,8,replace=False) for _ in range(500)])
    rng = np.random.RandomState(202610120)
    offset = .03*disk(rng,(500,8))
    yaw = rng.uniform(-15,15,(500,8))
    lighting = rng.randint(3,size=(500,8))
    noise_ids = rng.randint(16,size=(500,8))
    vrng = np.random.RandomState(202610121)
    validation_offsets = .03*disk(vrng,(len(val),))
    validation_yaw = vrng.uniform(-15,15,len(val))
    validation_lighting = vrng.randint(3,size=len(val))
    np.savez_compressed(root / 'plan.npz', frame_seed=frame_seeds, frame_query=frame_queries,
        batches=batches, offset=offset, yaw=yaw, lighting=lighting, noise_ids=noise_ids,
        validation_frames=val, validation_offsets=validation_offsets, validation_yaw=validation_yaw,
        validation_lighting=validation_lighting, noise_seeds=np.arange(202610200,202610216),
        probe_noise_seeds=np.arange(202610220,202610223))
    protocol = dict(task='place_container_plate', model='pi0.5_robotwin2', task_config='demo_clean',
        patch_size_m=.10, texture_size=128, sites=selection['centers'],
        independently_optimized_textures=9, single_patch_per_scene=True,
        area_matching='Match head camera exposure and projected area; retain and report real wrist visibility.',
        three_cameras_observe_one_world_texture=True,
        iters=500, batch=8, lr=.01, initialization='gray 0.5, independently for each site',
        loss='maximize squared t=1 velocity difference across 32x32 internal components',
        train_seeds=list(range(10100,10108)), training_common_frames=selection['training_common_frames'],validation_seeds=[10110,10111],
        eot=dict(position='area-uniform 3cm disk around each fixed center',
                 rotation='world yaw uniform [-15,+15] degrees',lighting=LIGHTS,flip=False),
        shared_draws='Same 4000 frames, position offsets, yaws, lighting labels and noise IDs across all 9 sites.',
        action_noise='16 fixed training noises; each gradient frame draws one, shared with its clean reference. Three disjoint probe noises.',
        evaluation='Fixed final checkpoint, per-site true-render fidelity and L0; texture CV comparison and 9x9 true-render cross-placement probe with Random. L0 is not closed-loop ASR.',
        selection_sha256=sha(root/'geometry/selection.json'),geometry_verification_sha256=sha(root/'geometry/verification.json'),
        plan_sha256=sha(root/'plan.npz'),reference_hashes={p.name:sha(p) for p in paths},
        comparisons='One training run per site, controlling initialization and random draws; not a multi-training-seed claim.')
    dump(root / 'protocol.json', protocol)
    return protocol


def records(root, site_id, split):
    root = Path(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    site = next(s for s in protocol['sites'] if s['id'] == site_id)
    result = []
    with np.load(root / 'plan.npz') as p:
        if split == 'train':
            frames=p['batches'].ravel(); offsets=p['offset'].reshape(-1,2)
            yaws=p['yaw'].ravel(); lights=p['lighting'].ravel(); noise_ids=p['noise_ids'].ravel()
        else:
            val=p['validation_frames']; frames=np.concatenate([val,val])
            offsets=np.concatenate([np.zeros((len(val),2)),p['validation_offsets']])
            yaws=np.concatenate([np.zeros(len(val)),p['validation_yaw']])
            lights=np.concatenate([np.zeros(len(val),int),p['validation_lighting']]); noise_ids=np.zeros(len(frames),int)
        for i,f in enumerate(frames):
            center=np.array(site['center_m'])+np.r_[offsets[i],0.]
            result.append(dict(id=i,frame=int(f),seed=int(p['frame_seed'][f]),query=int(p['frame_query'][f]),
                center=center.tolist(),yaw_deg=float(yaws[i]),light=list(LIGHTS)[int(lights[i])],
                noise_id=int(noise_ids[i]),site=site_id,
                path=str(root/'sites'/site_id/'capture'/split/f'{i:06d}.npz'),
                clean_path=str(root/'sites'/site_id/'clean'/f'frame{int(f):03d}_{list(LIGHTS)[int(lights[i])]}.npz')))
    return result


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(DEFAULT))
    prepare(Path(parser.parse_args().root).resolve())
