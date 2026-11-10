"""Small held-out, trajectory-split representation-distribution diagnostic."""
import argparse
import json
from pathlib import Path
import numpy as np
import checkpoint_study_common as S
import checkpoint_study_model as M


def distances(reference, query):
    """Three-neighbor distance in reference-standardized pooled features."""
    scale = reference.std(0)
    floor = max(float(np.median(scale[scale > 0]))*.1, 1e-6)
    scale = np.maximum(scale, floor)
    squared = ((query[:, None, :]-reference[None, :, :])/scale)**2
    d = np.sqrt(squared.mean(-1))
    return np.sort(d, axis=1)[:, :min(3, len(reference))].mean(1)


def main(args):
    import torch
    out = args.root/'representation_distribution'/args.model
    out.mkdir(parents=True, exist_ok=True)
    policy = M.load(args.root/'models'/args.model)
    def features(obs):
        captured = []
        def hook(module, inputs, output):
            v = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output[0]
            captured.append(v.detach().float())
        h = policy._model.paligemma_with_expert.paligemma.model.vision_tower.register_forward_hook(hook)
        try:
            with torch.no_grad():
                _, pad, kv = M.prefix(policy._model, obs)
        finally:
            h.remove()
        assert len(captured) == 3
        def pool(v):
            return torch.cat([v.mean(-2), v.std(-2, unbiased=False)], -1).cpu().numpy().ravel()
        result = {f'vision{i}': pool(v) for i, v in enumerate(captured)}
        valid = pad[0].bool()
        for i in sorted(set([0, len(kv.key_cache)//2, len(kv.key_cache)-1])):
            for label, cache in [('K', kv.key_cache), ('V', kv.value_cache)]:
                result[f'{label}{i}'] = pool(cache[i][:, :, valid, :].float())
        assert all(np.isfinite(v).all() for v in result.values())
        return result

    records = json.loads((args.root/'data_cache/manifest.json').read_text())['records']
    records = sorted([r for r in records if r['split'] == 'validation'], key=lambda r:r['seed'])
    assert len(records) == 20
    bank = []
    statebank = []
    bankmeta = []
    for ri, row in enumerate(records):
        source = args.root/'expert'/f'seed{row["seed"]}'/'demonstration.npz'
        if not source.exists():
            # The original study links validation trajectories with absolute
            # nnmc65 home paths. Use the shared study source on other hosts.
            source = args.root.parent/'checkpoint_study_20261023'/'expert'/f'seed{row["seed"]}'/'demonstration.npz'
        assert S.sha(source) == row['sha256']
        with np.load(source) as z:
            for phase, fraction in enumerate([0., 1/3, 2/3, 1.]):
                index = round((len(z['state14'])-1)*fraction)
                state = z['state14'][index].copy()
                obs = M.observation(policy, state, row['prompt'],
                                    {w:z[c][index].copy() for c,w in zip(S.CAMERAS,S.WIRE)})
                bank.append(features(obs)); statebank.append(state)
                bankmeta.append(dict(seed=row['seed'], frame=index, phase=phase,
                                     split='reference' if ri%2 == 0 else 'calibration'))
        print('[bank]', args.model, row['seed'], flush=True)
        S.dump(out/'status.json', dict(stage='building_bank', complete=False, trajectories=ri+1))
    keys = sorted(bank[0])
    matrices = {k:np.stack([r[k] for r in bank]) for k in keys}
    states = np.stack(statebank)
    ref = np.array([m['split']=='reference' for m in bankmeta])
    phases = np.array([m['phase'] for m in bankmeta])
    # Temporal bins are proxies, NOT semantic stages. The same clean joint
    # state selects a reference bin for F/U and every patch condition.
    state_scale = np.maximum(states[ref].std(0), .05)
    files = sorted((args.root/'diagnostic_frames').glob('*/manifest.json'))
    files = [files[i] for i in np.linspace(0, len(files)-1, min(12,len(files)), dtype=int)]
    assert files
    rows = []
    query_features = {}
    for file in files:
        meta = json.loads(file.read_text())
        assert meta['complete']
        clean = None; clean_action = None
        phase = None
        for condition in ['clean','clean_repeat']+[c['id'] for c in meta['conditions']]:
            with np.load(file.parent/(condition+'.npz')) as z:
                state = z['state14'].copy()
                obs = M.observation(policy, state, meta['prompt'], {w:z[w].copy() for w in S.WIRE})
            f = features(obs)
            query_features[file.parent.name+'__'+condition] = f
            noise = np.random.default_rng(202610290).standard_normal((1,32,32)).astype(np.float32)
            with torch.no_grad():
                _, action, raw = M.response(policy, obs, torch.tensor(noise,device='cuda'))
            if condition == 'clean':
                clean = f; clean_action = action
                nearest = np.argmin((((states[ref]-state)/state_scale)**2).mean(1))
                phase = int(phases[ref][nearest])
            for k in keys:
                r = matrices[k][ref & (phases==phase)]
                calibration = matrices[k][~ref & (phases==phase)]
                cal_d = distances(r, calibration)
                d, clean_d = distances(r, np.stack([f[k],clean[k]]))
                rows.append(dict(frame=file.parent.name, seed=meta['seed'], query=meta['query'],
                    condition=condition, feature=k, temporal_bin=phase,
                    distance=float(d), clean_distance=float(clean_d),
                    distance_over_calibration_median=float(d/max(np.median(cal_d),1e-8)),
                    calibration_percentile=float(np.mean(cal_d <= d)),
                    relative_feature_shift=float(np.linalg.norm(f[k]-clean[k])/max(np.linalg.norm(clean[k]),1e-8)),
                    action_MSE14=float(((action[...,:14]-clean_action[...,:14])**2).mean())))
        print('[frame]',args.model,file.parent.name,flush=True)
        S.dump(out/'status.json',dict(stage='paired_frames',complete=False,frames_done=len(query_features)//9))
    repeats = [r for r in rows if r['condition']=='clean_repeat']
    assert all(r['relative_feature_shift']==0 and r['action_MSE14']==0 for r in repeats), 'Clean repeat differs'
    np.savez_compressed(out/'features.npz', **{'bank__'+k:v for k,v in matrices.items()},
                        **{frame+'__'+k:v for frame,f in query_features.items() for k,v in f.items()})
    S.dump(out/'bank_manifest.json',dict(samples=bankmeta,reference_trajectories=10,calibration_trajectories=10))
    S.dump(out/'results.json',dict(model=args.model,rows=rows,
        limitations=['Pooled features discard spatial detail; exploratory diagnostic only',
                     'Temporal bins are not semantic stages',
                     'Captured query states originate from F clean rollouts',
                     'No object-position probe or causal feature replacement yet',
                     'Distances calibrated within each model; no cross-model coordinate subtraction']))
    S.dump(out/'status.json',dict(complete=True,frames=len(files),bank_samples=len(bank),rows=len(rows),clean_repeat_exact=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model',required=True);main(p.parse_args())
