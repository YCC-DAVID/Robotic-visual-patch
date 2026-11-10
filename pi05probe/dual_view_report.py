"""E26 completed-run audit, paired task counts, attribution and actual render figures."""
from pathlib import Path
import json,hashlib,shutil,datetime
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    from PIL import Image
    root=Path(__file__).resolve().parent;dest=root/'out/dual_view'
    assert json.loads((dest/'progress.json').read_text())['stage']=='complete'
    cleanpath=root/'out/render_proxy_scale/clean_baseline/episodes.json'
    clean=json.loads(cleanpath.read_text())['per_episode']['clean'];good={r['init_index'] for r in clean if r['success']};assert len(good)==9
    counts={};rawchecks=0
    for name in ['random','base','both','routed']:
        folder=dest/('rollout_'+name);data=json.loads((folder/'episodes.json').read_text());cond='random' if name=='random' else 'optimized';rows=data['per_episode'][cond]
        assert len(rows)==10 and {r['init_index'] for r in rows}==set(range(40,50))
        ref={r['init_index']:r for r in clean}
        for r in rows:
            a=np.load(folder/r['replay_record']);b=np.load(cleanpath.parent/ref[r['init_index']]['replay_record'])
            assert np.array_equal(a['initial_state'],b['initial_state'])
            n=min(len(a['fixture_body_pos']),len(b['fixture_body_pos']))
            for key in ['fixture_body_pos','fixture_body_quat']:assert np.array_equal(a[key][:n],b[key][:n])
            assert np.array_equal(a['noise_seeds'],20261006+r['init_index']*1000+np.arange(len(a['noise_seeds'])))
            rawchecks+=1
        counts[name]={'robot_task_successes':sum(r['success'] for r in rows),'episodes':10,'attack_successes':sum(not r['success'] for r in rows if r['init_index'] in good),'clean_success_denominator':9,
                      'failed_clean_success_indices':[r['init_index'] for r in rows if r['init_index'] in good and not r['success']]}
    training={};reference=np.load(root/'out/render_proxy_scale/train_13_101_full/checkpoint_0500.npz')
    for mode in ['base','both','routed']:
        folder=dest/('train_'+mode);ck=np.load(folder/'checkpoint_0500.npz');cfg=json.loads((folder/'config.json').read_text())
        assert cfg['complete'] and cfg['mode']==mode
        assert ck['batches'].shape==(500,8) and np.array_equal(ck['batches'],reference['batches']) and np.array_equal(ck['noise'],reference['noise'])
        assert np.array_equal(np.array(Image.open(folder/'texture_0500.png')),np.rint(ck['Q'].transpose(1,2,0)*255).clip(0,255).astype(np.uint8))
        assert all(sha(root/k)==v for k,v in cfg['source_sha256'].items())
        training[mode]=json.loads((folder/'results.json').read_text())
    validation=json.loads((dest/'validation/summary.json').read_text());v=np.load(dest/'validation/velocities.npz')
    error=0
    for j,name in enumerate(v['conditions']):
        value=float(np.square(v['velocity'][:,j].astype(float)-v['clean_velocity'].astype(float)).sum(axis=(-1,-2,-3)).mean())
        error=max(error,abs(value-validation[str(name)]['L0']))
    assert error<1e-4
    records=json.loads((dest/'validation/metrics.json').read_text())['records'];assert len(records)==32*4*3*2
    audit={'complete':True,'three_modes_all500_batches_and_noise_match_E24':True,'all_finalPNGs_matchQ':True,'training_sources_match_hashes':True,
           'raw_paired_rollouts_checked':rawchecks,'validation_model_records':len(records),'velocity_recompute_max_error':error,'validation_frames_from_two_episodes':32}
    report={'complete':True,'closed_loop':counts,'training':training,'validation':validation,'preflight':json.loads((dest/'preflight/summary.json').read_text()),'audit':audit,
            'notes':'One seed/task,13cm@101,sharedQ128.40new episodes on10previouslyused init40-49; clean9/10 reused. Validation32correlated frames from2episodes. Routed is gradient routing, not strict final texture bandlimit. Per-camera diagnostic replacements are not standalone physical experiments.'}
    (dest/'summary.json').write_text(json.dumps(report,indent=2)+'\n');(dest/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for mode,color in zip(['base','both','routed'],['#64748b','#2563eb','#d97706']):
        hist=np.load(dest/('train_'+mode)/'checkpoint_0500.npz')['history']
        axes[0].plot(np.arange(20,501),np.convolve(hist,np.ones(20)/20,mode='valid'),label=mode,color=color)
    axes[0].set_xlabel('PGD update');axes[0].set_ylabel('Training early L0 (20-update mean)');axes[0].legend()
    values=[counts[n]['attack_successes'] for n in counts];axes[1].bar(list(counts),values,color=['#cbd5e1','#64748b','#2563eb','#d97706']);axes[1].set_ylim(0,10)
    axes[1].set_ylabel('任务失败数（攻击成功数）',fontproperties=font)
    for i,n in enumerate(values):axes[1].text(i,n+.15,str(n)+'/9',ha='center')
    fig.suptitle('13 cm@101：同一Q的双相机联合训练与真实闭环',fontproperties=font,fontsize=14);fig.savefig(dest/'training_rollout.png',dpi=160);plt.close(fig)
    inputs=np.load(dest/'validation/inputs.npz');cal=np.load(dest/'calibration/wrist.npz');sizes=cal['mask'][inputs['frames']].sum((1,2));k=int(np.argsort(sizes)[len(sizes)//2])
    fig,axes=plt.subplots(4,3,figsize=(11,13),constrained_layout=True)
    for row,name in enumerate(['random','base','both','routed']):
        for col,img in enumerate([inputs[name+'__texture'],inputs[name+'__base'][k],inputs[name+'__wrist'][k]]):
            axes[row,col].imshow(img,interpolation='nearest');axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(['矩形世界Q128','真实base输入224','真实wrist输入224'][col],fontproperties=font)
        axes[row,0].set_ylabel(name,fontsize=12)
    fig.suptitle('同一纹理的两路真实render：init{} step{}'.format(int(inputs['ep_id'][k]),int(inputs['env_step'][k])),fontproperties=font,fontsize=14)
    fig.savefig(dest/'textures_real_views.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for mode,color in zip(['both','routed'],['#2563eb','#d97706']):
        rows=json.loads((dest/('train_'+mode)/'view_gradients.json').read_text());it=[r['iteration'] for r in rows]
        axes[0].plot(it,[r['wrist_norm']/max(r['base_norm'],1e-12) for r in rows],label=mode,color=color)
        axes[1].plot(it,[r['cosine'] for r in rows],label=mode,color=color)
    axes[0].set_yscale('log');axes[0].set_ylabel('Wrist / base batch gradient norm');axes[1].set_ylabel('Base / wrist gradient cosine');axes[1].axhline(0,color='gray',ls='--')
    for ax in axes:ax.set_xlabel('PGD update');ax.legend()
    fig.savefig(dest/'view_gradients.png',dpi=160);plt.close(fig)
    lines=['# E26 shared-world-texture dual-camera result','','Task failure on a clean-success init = attack success.','',
           '| Mode | Robot successes /10 | Attack successes /9 |','|---|---:|---:|']
    for mode,row in counts.items():lines.append('| {} | {} | {} |'.format(mode,row['robot_task_successes'],row['attack_successes']))
    lines+=['','| Texture, actual render | Base-only L0 | Wrist-only L0 | Both L0 | Both proxy L0 |','|---|---:|---:|---:|---:|']
    for mode in counts:
        vals=[validation[mode+'/'+view+'/render']['L0'] for view in ['base','wrist','both']]+[validation[mode+'/both/proxy']['L0']]
        lines.append('| {} | {} |'.format(mode,' | '.join('{:.6f}'.format(x) for x in vals)))
    lines+=['',report['notes'],'','See summary.json, audit.json and per-episode replay records.']
    (dest/'results.md').write_text('\n'.join(lines)+'\n')
    source_names=['dual_view_capture.py','dual_view_proxy.py','dual_view_fit.py','dual_view_probe.py','dual_view_train.py','dual_view_validation_capture.py','dual_view_pipeline.py','dual_view_gradient_check.py','dual_view_preflight_figure.py','dual_view_report.py','render_scale_rollout.py','render_scale_manifest.py','render_proxy_policy_server.py']
    snap=dest/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in source_names:shutil.copy2(root/name,snap/name)
    files={str(p.relative_to(dest)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='provenance.json' and p.suffix!='.log'}
    (dest/'provenance.json').write_text(json.dumps({'complete':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files},indent=2)+'\n')
    print(json.dumps({'closed_loop':counts,'audit':audit}))

if __name__=='__main__':main()
