"""Independent recomputation from saved velocities and rollout state arrays."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'out/proxy_factorial'


def main():
    protocol=json.loads((DEST/'protocol.json').read_text())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==s for n,s in protocol['source_sha256'].items())
    assert hashlib.sha256(Path(protocol['calibration']).read_bytes()).hexdigest()==protocol['calibration_sha256']
    velocity_errors={};physics_count=0
    for variant in sorted(protocol['variants']):
        directory=DEST/f'validation_{variant}'
        metrics=json.loads((directory/'metrics.json').read_text())
        with np.load(directory/'velocities.npz') as p:
            v=p['velocity'].astype(np.float64);c=p['clean_velocity'].astype(np.float64)
            conditions=p['conditions'].tolist()
            errors=[]
            for i,name in enumerate(conditions):
                l0=np.square(v[:,i]-c).reshape(len(v),3,-1).sum(-1).mean()
                errors.append(abs(l0-metrics['summary'][name]['L0']))
                if name.endswith('_proxy'):
                    r=conditions.index(name.replace('_proxy','_render'))
                    gap=np.square(v[:,i]-v[:,r]).reshape(len(v),3,-1).sum(-1).mean()
                    errors.append(abs(gap-metrics['summary'][name]['paired_L0_vs_render']))
            velocity_errors[variant]=max(errors)
            assert max(errors)<1e-5,(variant,errors)
        episodes=json.loads((DEST/f'rollout_{variant}/episodes.json').read_text())['per_episode']['optimized']
        assert len(episodes)==10
        for record in episodes:
            ep=record['init_index']
            with np.load(DEST/f'baseline/init{ep}_clean.npz') as clean,np.load(DEST/f'rollout_{variant}'/record['replay_record']) as attacked:
                assert np.array_equal(clean['initial_state'],attacked['initial_state'])
                for key in ['fixture_body_pos','fixture_body_quat']:
                    assert np.array_equal(clean[key],attacked[key][:len(clean[key])])
                noise=attacked['noise_seeds']
                assert np.array_equal(noise,20261006+ep*1000+np.arange(len(noise)))
                physics_count+=1
    result={'complete':True,'source_snapshot_hashes_match':True,'calibration_hash_matches':True,
            'paired_physics_and_noise_checked_episodes':physics_count,
            'velocity_metrics_max_abs_recompute_error':max(velocity_errors.values()),
            'velocity_errors_by_variant':velocity_errors,
            'limits':'KV tensors were not saved; KV distances cannot be independently recomputed from raw arrays here.'}
    (DEST/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
