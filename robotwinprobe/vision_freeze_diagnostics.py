"""Actual-render shared-state L0 and full-action cross-checkpoint matrices."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
import checkpoint_study_common as S
import checkpoint_study_model as M


def main(args):
    import torch
    policy=M.load(args.checkpoint or args.root/'models'/args.model)
    out=args.root/'diagnostics'/args.model;out.mkdir(parents=True,exist_ok=True)
    rows=[];directions=[];representation_rows=[]
    def representations(obs):
        captured=[]
        vision=policy._model.paligemma_with_expert.paligemma.model.vision_tower
        def hook(module,inputs,output):
            value=output.last_hidden_state if hasattr(output,'last_hidden_state') else output[0]
            captured.append(value.detach().float().cpu().numpy())
        handle=vision.register_forward_hook(hook)
        try:
            with torch.no_grad(): _,pad,kv=M.prefix(policy._model,obs)
        finally:handle.remove()
        values={f'vision_view{i}':v.ravel() for i,v in enumerate(captured)}
        assert len(captured)==3,'Expected three camera vision outputs'
        valid=pad[0].bool()
        for i,(key,value) in enumerate(zip(kv.key_cache,kv.value_cache)):
            for label,tensor in [('K',key),('V',value)]:
                values[f'{label}_layer{i:02d}']=tensor[:,:,valid,:].detach().float().cpu().numpy().ravel()
        return values

    files=sorted((args.root/'diagnostic_frames').glob('*/manifest.json'))
    assert files,'No actual-render diagnostic frames captured'
    for file in files:
        meta=json.loads(file.read_text());assert meta['complete'];dest=file.parent
        conditions=['clean','clean_repeat']+[r['id'] for r in meta['conditions']]
        values={};clean_repr=None
        for condition in conditions:
            with np.load(dest/(condition+'.npz')) as d:
                obs=M.observation(policy,d['state14'].copy(),meta['prompt'],{w:d[w].copy() for w in S.WIRE})
            rep=representations(obs)
            if condition=='clean':clean_repr=rep
            else:
                for key,current in rep.items():
                    clean=clean_repr[key];delta=current-clean
                    norm=np.linalg.norm(clean);other=np.linalg.norm(current)
                    representation_rows.append(dict(model=args.model,frame=dest.name,condition=condition,
                        representation=key,relative_L2=float(np.linalg.norm(delta)/max(norm,1e-12)),
                        cosine=float(np.dot(current,clean)/max(norm*other,1e-12))))
            values[condition]=[]
            for ni in range(3):
                # Different frames and models use the same three noise draws.
                noise=np.random.default_rng(202610290+ni).standard_normal((1,32,32)).astype(np.float32)
                with torch.no_grad():v,a,raw=M.response(policy,obs,torch.tensor(noise,device='cuda'))
                assert np.isfinite(v).all() and np.isfinite(a).all() and np.isfinite(raw).all()
                values[condition].append((v,a,raw))
                if condition=='clean':continue
                cv,ca,cr=values['clean'][ni];dv=v-cv
                arm=[0,1,2,3,4,5,7,8,9,10,11,12]
                rows.append(dict(model=args.model,frame=dest.name,seed=meta['seed'],query=meta['query'],
                    noise_id=ni,condition=condition,L0_32=float((dv**2).sum()),L0_14=float((dv[...,:14]**2).sum()),
                    action_MSE14=float(((a[...,:14]-ca[...,:14])**2).mean()),
                    joint_RMSE_rad=float(np.sqrt(((raw[:,arm]-cr[:,arm])**2).mean())),
                    gripper_RMSE=float(np.sqrt(((raw[:,[6,13]]-cr[:,[6,13]])**2).mean()))))
        # Functional similarity is computed within a target model, never across
        # two incompatible model latent coordinate systems.
        for left,right in itertools.combinations(conditions[2:],2):
            for ni in range(3):
                a=(values[left][ni][0]-values['clean'][ni][0])[...,:14].ravel()
                b=(values[right][ni][0]-values['clean'][ni][0])[...,:14].ravel()
                denom=np.linalg.norm(a)*np.linalg.norm(b)
                directions.append(dict(model=args.model,frame=dest.name,seed=meta['seed'],noise_id=ni,
                    left=left,right=right,delta_v_cosine=float(np.dot(a,b)/denom) if denom>1e-12 else None))
        print('[diagnostic]',args.model,dest.name,flush=True)
    for name,data in [('responses',rows),('directions',directions),('representations',representation_rows)]:
        with (out/(name+'.csv')).open('w') as h:
            fields=list(data[0]) if data else ['model','frame','seed','noise_id','left','right','delta_v_cosine']
            w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(data)
    S.dump(out/'status.json',dict(complete=True,frames=len(files),rows=len(rows),noise_count=3,
        actual_render=True,reference_models=sorted({json.loads(f.read_text())['reference_model'] for f in files}),
        reference_frames=[f.parent.name for f in files]))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=S.DEFAULT)
    ap.add_argument('--model',required=True);ap.add_argument('--checkpoint',type=Path)
    main(ap.parse_args())
