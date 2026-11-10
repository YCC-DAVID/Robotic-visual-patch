"""Audit every RoboTwin episode boundary against the actual input transforms."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
import vision_freeze_ft as V


def main(root):
    start=time.monotonic()
    out=root/'data_audit'
    V.S.dump(out/'status.json',dict(stage='running',complete=False,pid=os.getpid()))
    manifest=json.loads((root/'data_cache/manifest.json').read_text())
    records=manifest['records']
    train={r['seed'] for r in records if r['split']=='train'}
    validation={r['seed'] for r in records if r['split']=='validation'}
    assert len(train)==100 and len(validation)==20 and not train & validation
    assert manifest['normalization_fit_split']=='train only'
    cfg=V.make_config(root,'data_check',202610230,20000)
    data=V.Data(root,cfg)
    checked=0
    for r in records:
        ids=sorted(set([r['start'],r['start']+r['frames']//2,r['start']+r['frames']-1]))
        for i in ids:
            expected=data.full_sample(i)
            obs,actions=data.batch([i])
            assert np.array_equal(obs.state[0],np.asarray(expected['state'],np.float32))
            assert np.array_equal(actions[0],np.asarray(expected['actions'],np.float32))
            assert np.array_equal(obs.tokenized_prompt[0],expected['tokenized_prompt'])
            assert np.array_equal(obs.tokenized_prompt_mask[0],expected['tokenized_prompt_mask'])
            assert np.isfinite(actions).all() and np.isfinite(obs.state).all()
            for key in V.S.MODEL:
                assert np.array_equal(obs.images[key][0],np.asarray(expected['image'][key],np.float32)/255*2-1)
            future=np.minimum(i+np.arange(32),data.index['end'][i])
            assert np.all(data.index['episode'][future]==r['seed'])
            assert bool(data.index['train_mask'][i])==(r['split']=='train')
            checked+=1
    assert V.S.sha(Path(manifest['norm_stats']))==manifest['norm_sha256']
    result=dict(passed=True,complete=True,stage='complete',episodes=len(records),
                checked_frames=checked,train_frames=len(data.train_ids),
                validation_frames=len(data.validation_ids),episode_splits_disjoint=True,
                action_chunks_within_episode=True,exact_upstream_transforms=True,
                normalization_sha256=manifest['norm_sha256'],elapsed_s=time.monotonic()-start)
    V.S.dump(out/'result.json',result);V.S.dump(out/'status.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    try:main(args.root.resolve())
    except Exception as exc:
        import traceback
        V.S.dump(args.root/'data_audit/status.json',dict(stage='failed',complete=False,
            error=repr(exc),traceback=traceback.format_exc()))
        raise
