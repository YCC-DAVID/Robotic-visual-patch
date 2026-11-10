"""Audit matched low-frequency factorial and summarize mechanistic diagnostics."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'out';DEST=OUT/'dual_view_lowfreq'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    assert read(DEST/'progress.json')['stage']=='complete'
    run_paths={n:DEST/('train_'+n) for n in ['bilinear_base','bilinear_both','block_base']}
    run_paths['block_both']=OUT/'dual_view_both_low/train'
    source_hashes=read(DEST/'source_hashes_before_evaluation.json')
    assert all(sha(ROOT/name)==expected for name,expected in source_hashes.items())
    protocol=read(DEST/'protocol.json')
    assert all(sha(ROOT/name)==expected for name,expected in protocol['calibration_fit_sha256'].items())
    ref=np.load(OUT/'render_proxy_scale/train_13_101_band32/checkpoint_0500.npz')
    gradients={};parameter_errors={};checkpoints={}
    for name,run in run_paths.items():
        c=read(run/'config.json');ck=np.load(run/'checkpoint_0500.npz');checkpoints[name]=ck
        assert c['complete'] and ck['batches'].shape==(500,8)
        assert np.array_equal(ck['batches'],ref['batches']) and np.array_equal(ck['noise'],ref['noise'])
        assert all(sha(ROOT/n)==v for n,v in c['source_sha256'].items())
        for key in ['calibration_sha256','fit_sha256','wrist_calibration_sha256','wrist_fit_sha256']:
            assert c[key]==read(run_paths['bilinear_base']/'config.json')[key]
        q=ck['Q'];png=np.array(Image.open(run/'texture_0500.png'))
        assert np.array_equal(png,np.rint(q.transpose(1,2,0)*255).clip(0,255).astype(np.uint8))
        if name.startswith('bilinear'):
            expanded=F.interpolate(torch.from_numpy(ck['latent'])[None],(128,128),mode='bilinear',align_corners=False).numpy()[0]
            err=float(np.max(np.abs(expanded-q)));assert err<1e-6
        else:
            b=q.reshape(3,32,4,32,4);err=float(np.max(np.abs(b-b[:,:,0:1,:,0:1])));assert err==0
        parameter_errors[name]=err
        rows=read(run/'view_gradients.json');assert len(rows)==21
        if name.endswith('both'):
            ratios=[r['wrist_norm']/r['base_norm'] for r in rows]
            gradients[name]={'wrist_to_base_norm_median':float(np.median(ratios)),
                'cosine_median':float(np.median([r['cosine'] for r in rows])),
                'cosine_min':min(r['cosine'] for r in rows),'cosine_max':max(r['cosine'] for r in rows)}
            for key in ['joint_sign_matches_base','joint_sign_matches_wrist','coordinate_opposite_fraction']:
                if key in rows[0]:gradients[name][key+'_median']=float(np.median([r[key] for r in rows]))
        if 'gradient_sum_max_error' in rows[0]:
            maxerr=max(r['gradient_sum_max_error'] for r in rows);assert maxerr<1e-3,maxerr
    clean_path=OUT/'render_proxy_scale/clean_baseline';clean=read(clean_path/'episodes.json')['per_episode']['clean']
    refs={r['init_index']:r for r in clean};good={k for k,v in refs.items() if v['success']};assert len(good)==9
    audit_count=0
    def count(rows,path):
        nonlocal audit_count
        assert len(rows)==10 and {r['init_index'] for r in rows}==set(range(40,50))
        for row in rows:
            a=np.load(path/row['replay_record']);b=np.load(clean_path/refs[row['init_index']]['replay_record'])
            assert np.array_equal(a['initial_state'],b['initial_state'])
            n=min(len(a['fixture_body_pos']),len(b['fixture_body_pos']))
            for k in ['fixture_body_pos','fixture_body_quat']:assert np.array_equal(a[k][:n],b[k][:n])
            assert np.array_equal(a['noise_seeds'],20261006+row['init_index']*1000+np.arange(len(a['noise_seeds'])))
            audit_count+=1
        return {'task_successes':sum(r['success'] for r in rows),'episodes':10,
                'attack_successes':sum(not r['success'] for r in rows if r['init_index'] in good),
                'denominator':9,'failed_clean_success_indices':[r['init_index'] for r in rows if not r['success'] and r['init_index'] in good]}
    episodes=read(DEST/'rollout/episodes.json');counts={}
    for name in ['bilinear_base','bilinear_both','block_base']:
        counts[name]=count(episodes['per_episode'][name],DEST/'rollout')
        assert episodes['metadata']['textures'][name]['sha256']==sha(run_paths[name]/'texture_0500.png')
    oldblock=read(OUT/'dual_view_both_low/rollout/episodes.json')
    counts['block_both']=count(oldblock['per_episode']['optimized'],OUT/'dual_view_both_low/rollout')
    counts['random_block']=count(oldblock['per_episode']['random'],OUT/'dual_view_both_low/rollout')
    historical={condition:[] for condition in ['optimized','random']}
    for phase in ['pilot','confirm']:
        path=OUT/'render_proxy_scale'/(phase+'_13_101_band32')
        ep=read(path/'episodes.json')
        for condition in historical:
            for row in ep['per_episode'][condition]:
                # Absolute replay paths preserve two source directories for the paired audit.
                row=dict(row);row['replay_record']=str(path/row['replay_record']);historical[condition].append(row)
    counts['old_bilinear_base']=count(historical['optimized'],OUT)
    counts['random_bilinear']=count(historical['random'],OUT)
    # Confirm reused random controls match the exact two declared frequency structures.
    latent=np.random.RandomState(20261008).randint(0,256,(1,3,32,32)).astype(np.float32)/255
    for mode,image in [('bilinear',OUT/'render_proxy_scale/eval_13_101_band32/random.png'),
                       ('block',OUT/'dual_view_both_low/manifest/random.png')]:
        x=torch.from_numpy(latent)
        q=F.interpolate(x,(128,128),mode='bilinear',align_corners=False) if mode=='bilinear' else x.repeat_interleave(4,2).repeat_interleave(4,3)
        expected=np.rint(q.numpy()[0].transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
        assert np.array_equal(expected,np.array(Image.open(image)))
    validation=read(DEST/'validation/summary.json');v=np.load(DEST/'validation/velocities.npz');errors=[]
    for j,name in enumerate(v['conditions']):
        value=float(np.square(v['velocity'][:,j].astype(float)-v['clean_velocity'].astype(float)).sum(axis=(-1,-2,-3)).mean())
        errors.append(abs(value-validation[str(name)]['L0']))
    assert max(errors)<1e-4
    records=read(DEST/'validation/metrics.json')['records'];assert len(records)==768
    actions=read(DEST/'actions/summary.json');assert actions['complete']
    a=np.load(DEST/'actions/actions.npz');assert a['actions'].shape==(32,7,3,10,7)
    action_errors=[]
    for j,name in enumerate(a['conditions'][1:],1):
        delta=a['raw_actions'][:,j].astype(float)-a['raw_actions'][:,0].astype(float)
        value=float(np.square(delta[:,:,:5,:7]).sum(axis=(-1,-2)).mean())
        action_errors.append(abs(value-actions['conditions'][str(name)]['final_raw_exec5_real7_squared']))
    assert max(action_errors)<1e-4
    # Exploratory time/direction summaries; these clean-state windows are not pick labels.
    delta=a['raw_actions'].astype(float)-a['raw_actions'][:,0:1].astype(float)
    names=list(a['conditions']);steps=a['env_step'];windows={}
    for name in names[1:]:
        x=delta[:,names.index(name),:,:5,:7];y=delta[:,names.index('full_both'),:,:5,:7]
        cos=(x*y).sum((-1,-2))/(np.linalg.norm(x,axis=(-1,-2))*np.linalg.norm(y,axis=(-1,-2))+1e-20)
        dist=np.square(x).sum((-1,-2)).mean(1)
        windows[str(name)]={'delta_cos_to_full_both_mean':float(cos.mean()),
                           'delta_cos_to_full_both_min':float(cos.min()),
                           'final_exec_by_window':{label:float(dist[mask].mean()) for label,mask in
                             [('10-25',steps<30),('30-55',(steps>=30)&(steps<60)),('60-85',steps>=60)]}}
    (DEST/'actions/direction_windows.json').write_text(json.dumps({'summary':windows,
        'notes':'Exploratory fixed-clean-state diagnostic. Windows are environment steps, not pick phases. Cosine per state/noise over normalized executed5x7 coordinates.'},indent=2)+'\n')
    # Physical images for old Q should match earlier real-render validation exactly.
    inputs=np.load(DEST/'validation/inputs.npz');old=np.load(OUT/'render_proxy_scale/validation_13_101_band32/inputs.npz')
    old_keys=old.files
    # Generic E24 capture key names recorded for provenance; shared base rows checked if available.
    old_match={}
    for key in ['frames','clean_base','clean_wrist','state8']:
        if key in old:old_match[key]=bool(np.array_equal(inputs[key],old[key]));assert old_match[key]
    old_match['old_texture_real_base_exact']=bool(np.array_equal(inputs['old_bilinear_base__base'],old['optimized__render']))
    assert old_match['old_texture_real_base_exact']
    audit={'complete':True,'new_training_runs':3,'new_gradient_frame_forwards':12000,
           'batches_noise_match_E24_and_E26':True,'parameterization_max_errors':parameter_errors,
           'source_and_calibration_hashes_match':True,'new_closedloop_episodes':30,
           'new_and_reused_paired_records_checked':audit_count,'frequency_matched_random_PNGs_exact':True,
           'validation_records':len(records),'velocity_recompute_max_error':max(errors),
           'full_action_recompute_max_error':max(action_errors),'standard_policy_check':actions['verification'],
           'historical_input_alignment':old_match,'historical_input_keys':old_keys}
    summary={'complete':True,'closed_loop':counts,'gradients':gradients,'validation':validation,
             'actions':actions,'audit':audit,
             'notes':'Matched2x2 on13cm@101,one seed/task. block_both and controls reused.30new rollouts on10previouslyused states. Historical bilinear has different proxy and is not a camera ablation.'}
    (DEST/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    figures(counts,gradients,run_paths,inputs,a,actions)
    lines=['# Matched low-frequency × camera experiment','','Attack successes = failed tasks among9clean-success initial states.','',
           '| Condition | Task successes /10 | Attack successes /9 | Failed clean-success init |','|---|---:|---:|---|']
    for name,c in counts.items():lines.append(f"| {name} | {c['task_successes']} | {c['attack_successes']} | {c['failed_clean_success_indices']} |")
    lines+=['','Gradient diagnostics:',json.dumps(gradients,indent=2),'',summary['notes']]
    (DEST/'results.md').write_text('\n'.join(lines)+'\n')
    snap=DEST/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in list(source_hashes)+['dual_view_lowfreq_report.py']:
        shutil.copy2(ROOT/name,snap/name)
    files={str(p.relative_to(DEST)):{'sha256':sha(p),'bytes':p.stat().st_size}
           for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='provenance.json' and p.suffix!='.log'}
    (DEST/'provenance.json').write_text(json.dumps({'complete':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files},indent=2)+'\n')
    print(json.dumps({'counts':counts,'gradients':gradients,'audit':audit}),flush=True)


def figures(counts,gradients,runs,inputs,actions,action_summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    x=np.arange(2);width=.32
    for i,view in enumerate(['base','both']):
        vals=[counts[low+'_'+view]['attack_successes'] for low in ['bilinear','block']]
        bars=ax[0].bar(x+(i-.5)*width,vals,width,label=view)
        for bar,n in zip(bars,vals):ax[0].text(bar.get_x()+bar.get_width()/2,n+.15,str(n)+'/9',ha='center')
    ax[0].set_xticks(x,['Bilinear latent32','Block-gradient32']);ax[0].set_ylim(0,10)
    ax[0].set_ylabel('Attack successes / 9 clean-success states');ax[0].legend()
    for name in ['bilinear_both','block_both']:
        rows=read(runs[name]/'view_gradients.json')
        ax[1].plot([r['iteration'] for r in rows],[r['wrist_norm']/r['base_norm'] for r in rows],label=name)
    ax[1].axhline(1,color='gray',ls=':');ax[1].set_yscale('log');ax[1].set_xlabel('PGD update')
    ax[1].set_ylabel('Wrist/base gradient norm (parameter space)');ax[1].legend()
    fig.suptitle('Same imaging proxy, geometry, seed and 500 x 8 budget')
    fig.savefig(DEST/'factorial.png',dpi=160);plt.close(fig)
    oldblock=np.load(OUT/'dual_view_both_low/validation/inputs.npz')
    k=next(i for i in range(len(inputs['frames'])) if inputs['ep_id'][i]==10 and inputs['env_step'][i]==45)
    fig,ax=plt.subplots(4,3,figsize=(10,12),constrained_layout=True)
    for row,name in enumerate(['bilinear_base','bilinear_both','block_base','block_both']):
        d,key=(oldblock,'both_low') if name=='block_both' else (inputs,name)
        for col,img in enumerate([d[key+'__texture'],d[key+'__base'][k],d[key+'__wrist'][k]]):
            ax[row,col].imshow(img,interpolation='nearest');ax[row,col].set_xticks([]);ax[row,col].set_yticks([])
            if row==0:ax[row,col].set_title(['Shared Q128','Actual base224','Actual wrist224'][col])
        ax[row,0].set_ylabel(name)
    fig.suptitle('Same clean state: init10, step45. Both camera images are real render.')
    fig.savefig(DEST/'textures_views.png',dpi=150);plt.close(fig)
    names=['bilinear_base','bilinear_both','block_base','block_both','old_bilinear_base','full_both']
    fig,ax=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for name in names:
        s=action_summary['conditions'][name];style='--' if name in ['old_bilinear_base','full_both'] else '-'
        ax[0].plot(range(10),s['denoise_velocity_delta_squared_by_step'],style,label=name)
        vals=np.array(s['perframe_final_raw_exec5_real7_squared']);steps=actions['env_step']
        xs=sorted(set(steps.tolist()));ys=[float(vals[steps==x].mean()) for x in xs]
        ax[1].plot(xs,ys,style,label=name)
        ax[2].scatter(s['early_L0'],s['final_raw_exec5_real7_squared'],label=name)
    ax[0].set_xlabel('Denoising step (0 = t1)');ax[0].set_ylabel('Real7 velocity difference squared')
    ax[1].set_xlabel('Environment step (clean trajectory)');ax[1].set_ylabel('Final normalized exec5 x real7 delta squared')
    ax[2].set_xlabel('Early L0 (all10 x32)');ax[2].set_ylabel('Final normalized exec5 x real7 delta squared')
    ax[0].legend(fontsize=8);ax[2].legend(fontsize=8)
    fig.suptitle('Open-loop diagnostics: 32 correlated states, three common noise samples; not ASR')
    fig.savefig(DEST/'denoise_actions.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
