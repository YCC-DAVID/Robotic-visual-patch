"""E30 audit: direct-v orthogonality, fixed-target feasibility and true rollouts."""
from pathlib import Path
import json
import shutil
import datetime
import numpy as np
from direction_loss_report import angles, sha, read

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'out'
DEST = OUT/'orthogonal_research'
MODES = ['distance_pgd','orthogonal_pgd','orthogonal_adam','orthogonal_target']
LABELS = ['Distance PGD','Orthogonal PGD','Orthogonal Adam','Orthogonal target']


def angle_metrics(a, c):
    result = angles(a,c)
    a,c = np.asarray(a,dtype=float),np.asarray(c,dtype=float)
    a,c = a.reshape(*a.shape[:-2],-1),c.reshape(*c.shape[:-2],-1)
    relative = np.linalg.norm(a-c,axis=-1)/np.linalg.norm(c,axis=-1)
    result.update(relative_delta_norm_mean=float(relative.mean()),
                  relative_delta_norm_max=float(relative.max()),
                  relative_delta_norm_min=float(relative.min()))
    return result


def main():
    from PIL import Image
    import torch
    assert read(DEST/'progress.json')['stage'] in ['measurements_complete','complete']
    protocol = read(DEST/'protocol.json')
    for key in ['source_sha256','calibration_fit_sha256']:
        assert all(sha(ROOT/n)==v for n,v in protocol[key].items())
    for mode,path in protocol['initial_checkpoints'].items():
        assert sha(path)==protocol['initial_checkpoint_sha256'][mode]
    cal = np.load(OUT/'dual_view/calibration/base.npz')
    train = np.flatnonzero(cal['train_mask']);rng=np.random.RandomState(20261005)
    expected = np.array([rng.choice(train,8,replace=False) for _ in range(1000)])[500:]
    monitors = {};ranges = {};checkpoints={}
    for mode,loss,optimizer,parent in protocol['runs']:
        run=DEST/('train_'+mode);cfg=read(run/'config.json');ck=np.load(run/'checkpoint_1000.npz')
        initial=np.load(protocol['initial_checkpoints'][mode]);checkpoints[mode]=ck
        assert cfg['complete'] and cfg['objective']==loss and cfg['optimizer']==optimizer
        assert cfg['total_lineage_updates']==1000 and cfg['iters']==500 and int(ck['offset'])==500
        assert np.array_equal(ck['batches'],expected) and np.array_equal(ck['noise'],initial['noise'])
        assert all(sha(ROOT/n)==v for n,v in cfg['source_sha256'].items())
        assert np.array_equal(np.array(Image.open(run/'texture_1000.png')),
                              np.rint(ck['Q'].transpose(1,2,0)*255).clip(0,255).astype(np.uint8))
        history=read(run/'training_metrics.json');assert len(history)==500
        assert [r['iteration'] for r in history]==list(range(501,1001))
        samples=[m for r in history for m in r['per_frame']];assert len(samples)==4000
        assert min(m['norm_ratio'] for m in samples)>.001
        ranges[mode]={'minimum_cosine':min(m['cosine'] for m in samples),
                      'minimum_norm_ratio':min(m['norm_ratio'] for m in samples),
                      'maximum_norm_ratio':max(m['norm_ratio'] for m in samples)}
        rows=read(run/'monitoring.json');assert [r['iteration'] for r in rows]==[500,600,700,800,900,1000]
        monitors[mode]=[]
        for r in rows:
            item={'iteration':r['iteration']}
            for split,n in [('train_diag',8),('validation',32)]:
                vals=r[split];assert len(vals)==n
                cs=np.array([v['cosine'] for v in vals]);deg=np.degrees(np.arccos(cs.clip(-1,1)))
                item[split]={key:float(np.mean([v[key] for v in vals])) for key in ['objective','distance','cosine','norm_ratio']}
                item[split].update(angle_deg_mean=float(deg.mean()),angle_deg_max=float(deg.max()),
                                   near_orthogonal_fraction=float(((deg>=80)&(deg<=100)).mean()))
            monitors[mode].append(item)
        # First monitoring must reproduce the exact frozen parent checkpoint.
        old=read(OUT/'direction_loss'/('train_'+parent)/'monitoring.json')[-1]
        for split in ['train_diag','validation']:
            for a,b in zip(rows[0][split],old[split]):
                for key in ['distance','cosine','norm_ratio']:
                    assert abs(a[key]-b[key])<1e-6,(mode,key,a[key],b[key])
    targets=np.load(DEST/'train_orthogonal_target/orthogonal_targets.npz')
    tc,tt=targets['clean'].astype(float).reshape(187,-1),targets['target'].astype(float).reshape(187,-1)
    cn,tn=np.linalg.norm(tc,axis=1),np.linalg.norm(tt,axis=1)
    target_audit={'maximum_absolute_cosine':float(np.max(np.abs((tc*tt).sum(1)/(cn*tn)))),
                  'maximum_norm_ratio_error':float(np.max(np.abs(tn/cn-1))), 'states':187}
    assert target_audit['maximum_absolute_cosine']<1e-6 and target_audit['maximum_norm_ratio_error']<1e-6
    adam=torch.load(DEST/'train_orthogonal_adam/optimizer_1000.pt',map_location='cpu',weights_only=True)
    assert len(adam['state'])==1
    optimizer_state=next(iter(adam['state'].values()))
    assert int(optimizer_state['step'])==500
    for key in ['exp_avg','exp_avg_sq']:
        assert optimizer_state[key].shape==(1,3,128,128) and torch.isfinite(optimizer_state[key]).all()
    clean_path=OUT/'render_proxy_scale/clean_baseline';clean=read(clean_path/'episodes.json')['per_episode']['clean']
    refs={r['init_index']:r for r in clean};good={i for i,r in refs.items() if r['success']};assert len(good)==9
    checked=0
    def count(rows,path):
        nonlocal checked
        assert len(rows)==10 and {r['init_index'] for r in rows}==set(range(40,50))
        for r in rows:
            a=np.load(path/r['replay_record']);b=np.load(clean_path/refs[r['init_index']]['replay_record'])
            assert np.array_equal(a['initial_state'],b['initial_state'])
            n=min(len(a['fixture_body_pos']),len(b['fixture_body_pos']))
            for k in ['fixture_body_pos','fixture_body_quat']:assert np.array_equal(a[k][:n],b[k][:n])
            assert np.array_equal(a['noise_seeds'],20261006+r['init_index']*1000+np.arange(len(a['noise_seeds'])))
            checked+=1
        return {'task_successes':sum(r['success'] for r in rows),'episodes':10,
                'attack_successes':sum(not r['success'] for r in rows if r['init_index'] in good),
                'denominator':9,'failed_clean_success_indices':[r['init_index'] for r in rows if not r['success'] and r['init_index'] in good]}
    ep=read(DEST/'rollout/episodes.json');counts={}
    for mode in MODES:
        counts[mode]=count(ep['per_episode'][mode],DEST/'rollout')
        assert ep['metadata']['textures'][mode]['sha256']==sha(DEST/('train_'+mode)/'texture_1000.png')
    old=read(OUT/'direction_loss/rollout/episodes.json')
    for mode in ['distance','orthogonal']:
        counts[mode+'_500_reused']=count(old['per_episode'][mode],OUT/'direction_loss/rollout')
    counts['random_reused']=count(read(OUT/'dual_view/rollout_random/episodes.json')['per_episode']['random'],OUT/'dual_view/rollout_random')
    inputs=np.load(DEST/'validation/inputs.npz');previous=np.load(OUT/'direction_loss/validation/inputs.npz')
    for k in ['frames','clean_base','clean_wrist','state8']:assert np.array_equal(inputs[k],previous[k])
    capture=read(DEST/'validation/capture.json');assert capture['both_camera_native_anchors_exact']
    probe=read(DEST/'validation/summary.json');v=np.load(DEST/'validation/velocities.npz');early={};errors=[]
    for j,name in enumerate(v['conditions']):
        a,c=v['velocity'][:,j].astype(float),v['clean_velocity'].astype(float)
        l0=float(np.square(a-c).sum(axis=(-1,-2,-3)).mean());errors.append(abs(l0-probe[str(name)]['L0']))
        early[str(name)]={'all10x32':angle_metrics(a,c),'real10x7':angle_metrics(a[...,:7],c[...,:7]),
                          'exec5x7':angle_metrics(a[...,:5,:7],c[...,:5,:7])}
    actions=np.load(DEST/'actions/actions.npz');assert actions['actions'].shape==(32,5,3,10,7)
    act_summary=read(DEST/'actions/summary.json');assert act_summary['verification']['standard_policy_max_abs_error']<1e-6
    final={}
    for j,mode in enumerate(actions['conditions'][1:],1):
        a,c=actions['raw_actions'][:,j].astype(float),actions['raw_actions'][:,0].astype(float)
        value=float(np.square(a[:,:,:5,:7]-c[:,:,:5,:7]).sum(axis=(-1,-2)).mean())
        errors.append(abs(value-act_summary['conditions'][str(mode)]['final_raw_exec5_real7_squared']))
        final[str(mode)]={'raw_exec5x7':angles(a[:,:,:5,:7],c[:,:,:5,:7]),
                          'output_translation_exec5x3':angles(actions['actions'][:,j,:,:5,:3],actions['actions'][:,0,:,:5,:3])}
    assert max(errors)<1e-4 and (DEST/'service_stopped.json').exists()
    audit={'complete':True,'new_training_runs':4,'new_gradient_frame_forwards':16000,
           'total_updates_per_lineage':1000,'next500_batches_identical':True,'noise_identical':True,
           'initial_monitoring_matches_parent':True,'source_calibration_initial_hashes_match':True,
           'adam_new_updates_verified':500,
           'new_closedloop_episodes':40,'paired_records_checked':checked,'native256_anchors_exact':True,
           'target_geometry':target_audit,'early_records':256,'full_denoise_count':480,
           'numeric_recompute_max_error':max(errors),'standard_policy_check':act_summary['verification'],
           'service_stopped':True}
    summary={'complete':True,'closed_loop':counts,'monitoring':monitors,'training_ranges':ranges,
             'early_angles':early,'final_angles':final,'validation':probe,'actions':act_summary,'audit':audit,
             'notes':'Direct v_patch versus v_clean at t1 all10x32. Targets are exactly orthogonal, learned velocities may not be. One seed, reused init40–49, exploratory.'}
    for name,data in [('summary.json',summary),('audit.json',audit)]:
        (DEST/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    figures(summary,inputs)
    progress=read(DEST/'progress.json');progress.update(stage='complete',audit_complete=True)
    (DEST/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    snap=DEST/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in list(protocol['source_sha256'])+['orthogonal_research_report.py','direction_loss_report.py']:
        shutil.copy2(ROOT/name,snap/name)
    files={str(p.relative_to(DEST)):{'sha256':sha(p),'bytes':p.stat().st_size}
           for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='provenance.json' and p.suffix!='.log'}
    (DEST/'provenance.json').write_text(json.dumps({'complete':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files},indent=2)+'\n')
    print(json.dumps({'counts':counts,'early':early,'audit':audit}),flush=True)


def figures(s,inputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#31688e','#d07025','#21918c','#8c4c9f']
    fig,ax=plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
    for mode,label,color in zip(MODES,LABELS,colors):
        rows=s['monitoring'][mode];x=[r['iteration'] for r in rows]
        ax[0].plot(x,[r['validation']['angle_deg_mean'] for r in rows],'-o',label=label,color=color)
        ax[1].plot(x,[r['validation']['norm_ratio'] for r in rows],'-o',label=label,color=color)
    for j,(mode,color) in enumerate(zip(MODES,colors)):
        n=s['closed_loop'][mode]['attack_successes'];ax[2].bar(j,n,color=color);ax[2].text(j,n+.15,str(n)+'/9',ha='center')
    ax[0].set_xlabel('Total PGD/Adam updates');ax[0].set_ylabel('Mean angle(v_patch, v_clean), degrees');ax[0].legend(fontsize=8)
    ax[1].set_xlabel('Total updates');ax[1].set_ylabel('Mean norm(v_patch)/norm(v_clean)')
    ax[2].set_xticks(range(4),['Distance\nPGD','Orthogonal\nPGD','Orthogonal\nAdam','Orthogonal\ntarget'])
    ax[2].set_ylim(0,10);ax[2].set_ylabel('Task failures / 9 clean-success initial states')
    fig.suptitle('Continue500 to1000: direct velocity orthogonality. Curves: fixed proxy validation; counts: true render.')
    fig.savefig(DEST/'orthogonality_and_rollout.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4.4),constrained_layout=True)
    for j,(mode,label,color) in enumerate(zip(MODES,LABELS,colors)):
        r=s['early_angles'][mode+'/both/render']['all10x32']
        ax[0].bar(j,r['angle_deg_mean'],color=color)
        ax[0].plot(j,r['angle_deg_max'],'k_',markersize=12)
        ax[1].bar(j,r['relative_delta_norm_mean'],color=color)
        ax[1].plot(j,r['relative_delta_norm_max'],'k_',markersize=12)
    for a in ax:a.set_xticks(range(4),['Distance\nPGD','Orthogonal\nPGD','Orthogonal\nAdam','Orthogonal\ntarget'])
    ax[0].axhspan(80,100,alpha=.12,color='green',label='Predefined near-orthogonal band')
    ax[0].axhline(90,color='green',ls='--');ax[0].set_ylim(0,105);ax[0].set_ylabel('Angle(v_patch, v_clean), degrees');ax[0].legend(fontsize=8)
    ax[1].axhline(1,color='red',ls='--',label='Necessary lower bound for exact orthogonality')
    ax[1].set_ylabel('norm(v_patch - v_clean) / norm(v_clean)');ax[1].legend(fontsize=8)
    fig.suptitle('Actual render: bars=mean, black marks=max over32 states x3noise; not an impossibility proof')
    fig.savefig(DEST/'achieved_vs_required.png',dpi=160);plt.close(fig)
    k=next(i for i in range(32) if inputs['ep_id'][i]==10 and inputs['env_step'][i]==45)
    fig,ax=plt.subplots(4,3,figsize=(10,12),constrained_layout=True)
    for r,(mode,label) in enumerate(zip(MODES,LABELS)):
        for c,img in enumerate([inputs[mode+'__texture'],inputs[mode+'__base'][k],inputs[mode+'__wrist'][k]]):
            ax[r,c].imshow(img,interpolation='nearest');ax[r,c].set_xticks([]);ax[r,c].set_yticks([])
            if r==0:ax[r,c].set_title(['Shared physical Q128','Actual base224','Actual wrist224'][c])
        ax[r,0].set_ylabel(label)
    fig.suptitle('Same physical state: init10 step45. Camera columns are true MuJoCo render.')
    fig.savefig(DEST/'textures_views.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
