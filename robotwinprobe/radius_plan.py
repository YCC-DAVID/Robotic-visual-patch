"""Immutable, paired Monte Carlo design for continuous position EOT."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import bootstrap as B

REFERENCE = B.OUT/'render_transfer'
RADII = [1, 3, 5]


def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def disk(rng, shape):
    u=rng.uniform(size=(*shape,2));r=np.sqrt(u[...,0]);a=2*np.pi*u[...,1]
    return np.stack([r*np.cos(a),r*np.sin(a)],-1)


def records(root, radius, split):
    with np.load(Path(root)/'plan.npz') as p:
        ids=p['batches'].reshape(-1) if split=='train' else p['validation_frames']
        xy=p['unit_train'].reshape(-1,2) if split=='train' else p['unit_validation']
        return [{'id':j,'frame':int(f),'seed':int(p['frame_seed'][f]),'query':int(p['frame_query'][f]),
                 'center':[float(xy[j,0]*radius/100),float(xy[j,1]*radius/100),.741],
                 'path':str(Path(root)/'capture'/f'r{radius:02d}'/split/f'{j:06d}.npz')}
                for j,f in enumerate(ids)]


def prepare(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():return
    assert not (root/'plan.npz').exists(), 'partial plan; preserve and inspect'
    seeds=[];queries=[];hashes={}
    for path in sorted((REFERENCE/'calibration').glob('seed*.npz')):
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as d:
            seeds.extend([int(d['seed'])]*len(d['query']));queries.extend(d['query'].tolist())
    seeds=np.array(seeds);queries=np.array(queries)
    ids=np.flatnonzero(seeds<10110);val=np.flatnonzero(seeds>=10110)
    rng=np.random.RandomState(20261007)
    batches=np.array([rng.choice(ids,8,replace=False) for _ in range(500)])
    unit=disk(np.random.RandomState(20261009),(500,8))
    vunit=disk(np.random.RandomState(20261010),(len(val),))
    test_seeds=np.arange(10400,10500)
    test_xy=.05*disk(np.random.RandomState(20261011),(100,10))
    np.savez_compressed(root/'plan.npz',frame_seed=seeds,frame_query=queries,batches=batches,
                        unit_train=unit,validation_frames=val,unit_validation=vunit,
                        test_seeds=test_seeds,test_xy=test_xy)
    # Independent check of pairing to the already trained nominal baseline.
    with np.load(REFERENCE/'train_proxy_full/checkpoint_0500.npz') as old:
        assert np.array_equal(batches,old['batches'])
    baseline=json.loads((REFERENCE/'train_proxy_full/config.json').read_text())
    assert hashes==baseline['pool_sha256']
    protocol={'task':'place_container_plate','task_config':'demo_clean','size_m':.20,
        'world_origin':[0,0,.741],'train_radii_cm':RADII,'light':'nominal fixed',
        'baseline':str(REFERENCE/'train_proxy_full/texture_0500.png'),
        'baseline_sha256':hashlib.sha256((REFERENCE/'train_proxy_full/texture_0500.png').read_bytes()).hexdigest(),
        'iters':500,'batch':8,'lr':.01,'parameterization':'Q128 full, shared by all three cameras',
        'loss':'maximize squared first t=1 velocity difference, all 32x32 dimensions',
        'noise_seed':20261007,'batch_seed':20261007,'train_transform_seed':20261009,
        'validation_transform_seed':20261010,'test_transform_seed':20261011,
        'sampling':'area-uniform disk: r=R*sqrt(U), theta=2*pi*V',
        'training_transforms':'4000 fresh continuous samples, frozen before calibration; each used once',
        'radius_pairing':'same states and unit-disk draws, offsets scaled by radius',
        'test_initial_states':50,'extra_clean_rule':'if first 50 have <30 clean successes, extend whole cohort to 100',
        'test_positions_per_state':10,'test_radius_m':.05,'also_test_origin':True,
        'test_position_rule':'independent positions per initial state; all methods share them; fixed within episode',
        'test_seeds':test_seeds.tolist(),'train_frames':len(ids),'validation_frames':len(val),
        'pool_sha256':hashes,'plan_sha256':hashlib.sha256((root/'plan.npz').read_bytes()).hexdigest(),
        'inference':'single training seed; paired scene-cluster uncertainty, not independent episodes'}
    dump(root/'protocol.json',protocol)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();prepare(args.root)
