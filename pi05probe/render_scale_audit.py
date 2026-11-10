"""Recompute saved evidence and pairing checks after the physical sweep."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'out/render_proxy_scale'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    baseline_path=BASE/'clean_baseline/episodes.json'
    baseline=json.loads(baseline_path.read_text())
    references={r['init_index']:np.load(baseline_path.parent/r['replay_record'],allow_pickle=False) for r in baseline['per_episode']['clean']}
    reference_train=np.load(ROOT/'out/render_proxy_eot/train_nominal/checkpoint_0500.npz',allow_pickle=False)
    audits=[];records_checked=0;max_error=0
    for run in sorted(BASE.glob('train_*')):
        config=json.loads((run/'config.json').read_text());result=json.loads((run/'results.json').read_text())
        assert config['complete'] and result['complete'] and result['iterations']==500
        assert result['gradient_frame_forwards']==4000
        snapshot=np.load(run/'checkpoint_0500.npz',allow_pickle=False)
        assert np.array_equal(snapshot['batches'],reference_train['batches'])
        assert np.array_equal(snapshot['noise'],reference_train['noise'])
        assert sha(config['calibration_path'])==config['calibration_sha256']
        assert sha(config['fit_path'])==config['fit_sha256']
        q=torch.from_numpy(snapshot['latent'])[None]
        if config['parameter_shape'][-1]==32:q=F.interpolate(q,(128,128),mode='bilinear',align_corners=False)
        assert np.array_equal(q.numpy()[0],snapshot['Q'])
        png=np.asarray(Image.open(run/'texture_0500.png').convert('RGB'))
        assert np.array_equal(png,np.rint(snapshot['Q'].transpose(1,2,0)*255).clip(0,255).astype(np.uint8))
        assert sha(run/'texture_0500.png')==result['texture_sha256']
        name=run.name[6:];stages=[]
        for stage in ['pilot','confirm']:
            path=BASE/(stage+'_'+name)/'episodes.json'
            if not path.exists():continue
            data=json.loads(path.read_text());ids=data['metadata']['indices']
            expected=list(range(40,45)) if stage=='pilot' else list(range(45,50))
            assert ids==expected
            for c in ['random','optimized']:
                rows=data['per_episode'][c];assert len(rows)==5
                for row in rows:
                    ep=row['init_index'];record=np.load(path.parent/row['replay_record'],allow_pickle=False);ref=references[ep]
                    assert np.array_equal(record['initial_state'],ref['initial_state'])
                    n=len(ref['fixture_body_pos'])
                    assert np.array_equal(record['fixture_body_pos'][:n],ref['fixture_body_pos'])
                    assert np.array_equal(record['fixture_body_quat'][:n],ref['fixture_body_quat'])
                    assert np.array_equal(record['noise_seeds'],20261006+ep*1000+np.arange(len(record['noise_seeds'])))
                    assert np.isfinite(record['actions']).all();records_checked+=1
                clean_success={r['init_index'] for r in data['per_episode']['clean'] if r['success']}
                paired=[r for r in rows if r['init_index'] in clean_success]
                assert data['summary'][c]['paired_ASR_failures']==sum(not r['success'] for r in paired)
                assert data['summary'][c]['paired_ASR_clean_success_n']==len(paired)
            stages.append(stage)
        vpath=BASE/('validation_'+name)
        metrics=json.loads((vpath/'metrics.json').read_text());v=np.load(vpath/'velocities.npz',allow_pickle=False)
        assert metrics['frames']==32 and len(metrics['records'])==128
        assert v['velocity'].shape==(32,4,3,1,10,32)
        for j,seed in enumerate([20260901,20260902,20260903]):
            assert np.array_equal(v['action_noise'][j],np.random.RandomState(seed).standard_normal((10,32)).astype(np.float32))
        for i,frame in enumerate(v['steps']):
            for j,c in enumerate(v['conditions']):
                row=next(r for r in metrics['records'] if r['frame_index']==int(frame) and r['condition']==c)
                diff=v['velocity'][i,j].astype(float)-v['clean_velocity'][i].astype(float)
                values={'L0':np.square(diff).reshape(3,-1).sum(1).mean(),
                    'L0_real7':np.square(diff[...,:7]).reshape(3,-1).sum(1).mean(),
                    'L0_exec7':np.square(diff[:,:,:5,:7]).reshape(3,-1).sum(1).mean()}
                for key,value in values.items():
                    error=abs(float(value)-row['vs_clean'][key]);max_error=max(max_error,error)
                    assert error<max(1e-4,abs(value)*1e-5),(name,key,error)
        audits.append({'name':name,'budget':4000,'same_batches_and_noise':True,'low_resolution_constraint_exact':True,
                       'PNG_equals_checkpoint':True,'physics_fixture_and_noise_pairing':True,'stages':stages,'velocity_records':128})
    report={'complete':True,'training_runs':audits,'physical_records_checked':records_checked,
        'maximum_velocity_metric_recompute_error':max_error,
        'notes':'Scientific evidence audit, no new model inference. Native calibration anchors and deterministic forward/cache checks were asserted by the capture/probe jobs.'}
    (BASE/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
