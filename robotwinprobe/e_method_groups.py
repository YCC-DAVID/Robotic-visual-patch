"""Training-state conflict diagnostics and matched group-balanced L0 updates."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from e_method_common import draw_plan,dump,load,request,sha

GROUP_METHODS=('velocity_group_balanced','velocity_pcgrad')


def training_groups(root):
    rows=json.loads((root/'collection_manifest.json').read_text())['records']
    groups={}
    for row in rows:
        with np.load(row['path']) as z:
            label=('bowl' if 'bowl' in str(z['prompt']) else 'cup')+('_right' if int(z['active_start'])==7 else '_left')
        groups.setdefault(label,{}).setdefault(row['seed'],{}).setdefault(row['query'],{})[row['draw']]=row['path']
    assert set(groups)=={'bowl_left','bowl_right','cup_left','cup_right'}
    return groups


def group_draw_plan(root,seed):
    groups=training_groups(root);rng=np.random.default_rng(seed);result=[]
    for _ in range(load(root)['method_study']['steps']):
        batch=[]
        for group in sorted(groups):
            for _ in range(2):
                source=int(rng.choice(sorted(groups[group])))
                query=int(rng.choice(sorted(groups[group][source])))
                draw=int(rng.random()>=.75)
                batch.append(dict(record=groups[group][source][query][draw],seed=int(rng.integers(1,2**30)),group=group))
        result.append(batch)
    return result


def merge_gradients(gradients,project=False,seed=0):
    keys=sorted(gradients);original=[np.asarray(gradients[k],np.float64) for k in keys]
    norms=np.array([np.linalg.norm(g) for g in original])
    cos=[]
    for i in range(len(keys)):
        for j in range(i):
            if norms[i]>0 and norms[j]>0:cos.append(float(np.sum(original[i]*original[j])/(norms[i]*norms[j])))
    projected=[g.copy() for g in original];rng=np.random.default_rng(seed)
    if project:
        for i,g in enumerate(projected):
            for j in rng.permutation(len(keys)):
                if i==j:continue
                norm=float(np.sum(original[j]*original[j]));dot=float(np.sum(g*original[j]))
                if norm>1e-30 and dot<0:g-=dot/norm*original[j]
    merged=np.mean(projected,axis=0)
    diagnostics=dict(negative_pair_fraction=float(np.mean(np.array(cos)<0)) if cos else 0,
        mean_pair_cosine=float(np.mean(cos)) if cos else 0,
        raw_survival_ratio=float(np.linalg.norm(np.sum(original,axis=0))/max(float(norms.sum()),1e-30)),
        projected_gradient_norm=float(np.linalg.norm(merged)))
    return merged.astype(np.float32),diagnostics


def conflict_audit(root):
    output=root/'group_round/conflict_summary.json'
    if output.exists():return json.loads(output.read_text())
    groups=training_groups(root);out=root/'group_round/conflicts';out.mkdir(parents=True,exist_ok=True)
    nominal=[]
    for group in sorted(groups):
        for source in sorted(groups[group]):
            for query in sorted(groups[group][source]):
                nominal.append(dict(group=group,seed=source,query=query,record=groups[group][source][query][0]))
    noises=[202610280,202610281]
    config=dict(base_protocol_sha256=sha(root/'protocol.json'),collection_sha256=sha(root/'collection_manifest.json'),
        texture_sha256=sha(root/'patches/velocity14/texture_0500.png'),records=nominal,noises=noises,
        scope='training-only proxy derivatives at frozen velocity14 texture; not actual-render derivatives, not test ASR')
    protocol=out/'protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())==config
    else:dump(protocol,config)
    texture=np.asarray(Image.open(root/'patches/velocity14/texture_0500.png').convert('RGB')).transpose(2,0,1)[None].astype(np.float32)/255
    items=[];started=time.monotonic()
    for index,row in enumerate(nominal):
        artifact=out/f's{row["seed"]}_q{row["query"]:03d}.npz'
        if not artifact.exists():
            gradients=[];values=[]
            for noise in noises:
                r=request(dict(command='method_gradient',kind='velocity14',texture=texture,record=row['record'],seed=noise),root)
                gradients.append(r['gradient']);values.append(r['loss'])
            temp=artifact.with_suffix('.tmp.npz')
            np.savez_compressed(temp,gradients=np.array(gradients),losses=values,diagnostic_protocol_sha256=sha(protocol));temp.replace(artifact)
        with np.load(artifact) as z:
            assert str(z['diagnostic_protocol_sha256'])==sha(protocol)
            gradient=z['gradients'].mean(0).astype(np.float64)
            a,b=z['gradients'].reshape(2,-1).astype(np.float64)
            noise_cosine=float(np.dot(a,b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-30))
        items.append(dict(**row,gradient=gradient,noise_cosine=noise_cosine))
        dump(root/'group_round/progress.json',dict(stage='conflict_diagnostic',completed_states=index+1,total_states=len(nominal),elapsed_s=time.monotonic()-started))
    # Correlated query states are averaged within each initial seed first.
    by_seed={}
    for row in items:by_seed.setdefault(row['seed'],[]).append(row)
    seed_gradients={str(seed):np.mean([r['gradient'] for r in rows],axis=0) for seed,rows in by_seed.items()}
    _,seed_stats=merge_gradients(seed_gradients)
    group_gradients={group:np.mean([seed_gradients[str(seed)] for seed,rows in by_seed.items() if rows[0]['group']==group],axis=0) for group in groups}
    _,group_stats=merge_gradients(group_gradients)
    _,projected_stats=merge_gradients(group_gradients,project=True,seed=202610270)
    stats=dict(passed=True,scope=config['scope'],independent_training_seeds=len(by_seed),states=len(items),
        groups={group:dict(initial_states=len(groups[group]),states=sum(len(v) for v in groups[group].values()),
                           mean_gradient_norm=float(np.linalg.norm(group_gradients[group]))) for group in groups},
        across_initial_states=seed_stats,across_groups=group_stats,projected_groups=projected_stats,
        median_within_state_noise_cosine=float(np.median([r['noise_cosine'] for r in items])),
        test_used=False,diagnostic_protocol_sha256=sha(protocol))
    dump(output,stats);return stats


def prepare_group_round(runner):
    root=runner.root;path=root/'group_round/protocol.json'
    if path.exists():
        config=json.loads(path.read_text())
        assert config['base_protocol_sha256']==sha(root/'protocol.json')
        assert config['collection_sha256']==sha(root/'collection_manifest.json')
        return config
    runner.status('training_gradient_conflict')
    stats=conflict_audit(root)
    groups=training_groups(root)
    config=dict(round=3,methods=list(GROUP_METHODS),loss_kind='velocity14',
        base_protocol_sha256=sha(root/'protocol.json'),collection_sha256=sha(root/'collection_manifest.json'),
        conflict_summary_sha256=sha(root/'group_round/conflict_summary.json'),
        grouping='training prompt bowl/cup x active left/right arm, four nonempty groups',
        group_seed_counts={k:len(v) for k,v in groups.items()},steps=500,batch=8,lr=.01,initialization='gray .5',
        draws='two per group; uniform seed then query inside group; 75% nominal/25% EOT; same draws and noises for both methods',
        velocity_group_balanced='ordinary mean of four two-sample group gradients',
        velocity_pcgrad='same group means; sequentially remove negative dot-product components against original other-group gradients; deterministic shuffled projection order; then mean',
        projection_seed='training_seed + update index',diagnostic_support=stats,
        comparison='same32 development initial states/noises and GPU assignments; old records preserved',
        independent_test='untouched; test does not select methods, samples, or checkpoints')
    dump(path,config);return config
