"""Audit and visualize the frozen E26 both-low extension, preserving old results."""
from pathlib import Path
import json,hashlib,shutil,datetime
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def main():
    from PIL import Image
    root=Path(__file__).resolve().parent;dest=root/'out/dual_view_both_low';old=root/'out/dual_view'
    assert json.loads((dest/'progress.json').read_text())['stage']=='complete'
    run=dest/'train';cfg=json.loads((run/'config.json').read_text());ck=np.load(run/'checkpoint_0500.npz');ref=np.load(old/'train_routed/checkpoint_0500.npz')
    assert cfg['complete'] and cfg['mode']=='both_low' and ck['batches'].shape==(500,8)
    assert np.array_equal(ck['batches'],ref['batches']) and np.array_equal(ck['noise'],ref['noise'])
    assert all(sha(root/k)==v for k,v in cfg['source_sha256'].items())
    q=ck['Q'];png=np.array(Image.open(run/'texture_0500.png'));assert np.array_equal(png,np.rint(q.transpose(1,2,0)*255).clip(0,255).astype(np.uint8))
    blocks=q.reshape(3,32,4,32,4);spread=float(np.max(np.abs(blocks-blocks[:,:,0:1,:,0:1])));assert spread==0
    random_png=np.array(Image.open(dest/'manifest/random.png'));random_blocks=random_png.reshape(32,4,32,4,3)
    assert np.array_equal(random_blocks,np.broadcast_to(random_blocks[:,0:1,:,0:1,:],random_blocks.shape))
    episodes=json.loads((dest/'rollout/episodes.json').read_text());rows=episodes['per_episode']['optimized'];clean=episodes['per_episode']['clean'];cleanpath=Path(episodes['metadata']['clean_reference']).parent
    assert len(rows)==10 and {r['init_index'] for r in rows}==set(range(40,50))
    good={r['init_index'] for r in clean if r['success']};assert len(good)==9
    refs={r['init_index']:r for r in clean}
    random_rows=episodes['per_episode']['random'];assert len(random_rows)==10
    for row in rows+random_rows:
        a=np.load(dest/'rollout'/row['replay_record']);b=np.load(cleanpath/refs[row['init_index']]['replay_record'])
        assert np.array_equal(a['initial_state'],b['initial_state'])
        n=min(len(a['fixture_body_pos']),len(b['fixture_body_pos']))
        for key in ['fixture_body_pos','fixture_body_quat']:assert np.array_equal(a[key][:n],b[key][:n])
        assert np.array_equal(a['noise_seeds'],20261006+row['init_index']*1000+np.arange(len(a['noise_seeds'])))
    counts={'robot_task_successes':sum(r['success'] for r in rows),'episodes':10,'attack_successes':sum(not r['success'] for r in rows if r['init_index'] in good),'clean_success_denominator':9,'failed_clean_success_indices':[r['init_index'] for r in rows if r['init_index'] in good and not r['success']]}
    validation=json.loads((dest/'validation/summary.json').read_text());v=np.load(dest/'validation/velocities.npz');error=0
    for j,name in enumerate(v['conditions']):
        value=float(np.square(v['velocity'][:,j].astype(float)-v['clean_velocity'].astype(float)).sum(axis=(-1,-2,-3)).mean())
        error=max(error,abs(value-validation[str(name)]['L0']))
    assert error<1e-4
    inputs=np.load(dest/'validation/inputs.npz');prev=np.load(old/'validation/inputs.npz')
    for key in ['frames','clean_base','clean_wrist','state8']:assert np.array_equal(inputs[key],prev[key]),key
    oldreport=json.loads((old/'summary.json').read_text());comparison=oldreport['closed_loop'];comparison['random_low']={'robot_task_successes':sum(r['success'] for r in random_rows),'episodes':10,'attack_successes':sum(not r['success'] for r in random_rows if r['init_index'] in good),'clean_success_denominator':9}
    comparison['both_low']=counts
    records=json.loads((dest/'validation/metrics.json').read_text())['records'];assert len(records)==384
    audit={'complete':True,'matched500_batches_noise_with_E26':True,'PNG_matches_finalQ':True,'source_hashes_match':True,'raw_paired_rollouts_checked':20,'random32_blocks_match_optimized_constraint':True,'validation_records':len(records),'velocity_recompute_max_error':error,'Q_within4x4_block_max_difference':spread}
    summary={'complete':True,'closed_loop':counts,'comparison':comparison,'training':json.loads((run/'results.json').read_text()),'validation':validation,'audit':audit,'notes':'Same sharedQ13cm@101,500x8,one seed. Both paths lowpass adjoint; from gray init Q remains4x4constant blocks, different from bilinear band32.20new episodes on previouslyused40-49; matchedrandom32 newly evaluated; clean/E26comparators reused.'}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(dest/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for name,color in [('both','#2563eb'),('routed','#d97706'),('both_low','#059669')]:
        data=ck if name=='both_low' else np.load(old/('train_'+name)/'checkpoint_0500.npz');hist=data['history']
        axes[0].plot(np.arange(20,501),np.convolve(hist,np.ones(20)/20,mode='valid'),label=name,color=color)
    axes[0].set_xlabel('PGD update');axes[0].set_ylabel('Training early L0 (20-update mean)');axes[0].legend()
    vals=[r['attack_successes'] for r in comparison.values()];axes[1].bar(list(comparison),vals,color=['#cbd5e1','#64748b','#2563eb','#d97706','#a3a3a3','#059669']);axes[1].set_ylim(0,10);axes[1].set_ylabel('任务失败数（攻击成功数）',fontproperties=font)
    for i,n in enumerate(vals):axes[1].text(i,n+.15,str(n)+'/9',ha='center')
    fig.suptitle('13 cm@101：新增两路同时低频梯度约束',fontproperties=font,fontsize=14);fig.savefig(dest/'comparison.png',dpi=160);plt.close(fig)
    k=next(i for i in range(len(inputs['frames'])) if inputs['ep_id'][i]==10 and inputs['env_step'][i]==45)
    fig,axes=plt.subplots(3,3,figsize=(11,10),constrained_layout=True)
    for row,name in enumerate(['both','routed','both_low']):
        d=inputs if name=='both_low' else prev
        for col,img in enumerate([d[name+'__texture'],d[name+'__base'][k],d[name+'__wrist'][k]]):
            axes[row,col].imshow(img,interpolation='nearest');axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(['共享矩形Q128','真实base输入224','真实wrist输入224'][col],fontproperties=font)
        axes[row,0].set_ylabel(name)
    fig.suptitle('相同init10 step45：同一世界纹理在两路真实render中',fontproperties=font,fontsize=14);fig.savefig(dest/'textures_views.png',dpi=150);plt.close(fig)
    lines=['# Both-camera low-frequency routing extension','','Attack success = task failure among9clean-success initial states.','', '| Mode | Robot task successes /10 | Attack successes /9 |','|---|---:|---:|']
    for name,c in comparison.items():lines.append('| {} | {} | {} |'.format(name,c['robot_task_successes'],c['attack_successes']))
    lines+=['','| New texture | Real L0 | Proxy L0 | KV relative render |','|---|---:|---:|---:|']
    for view in ['base','wrist','both']:
        r=validation['both_low/'+view+'/render'];p=validation['both_low/'+view+'/proxy'];lines.append('| {} | {:.6f} | {:.6f} | {:.6f} |'.format(view,r['L0'],p['L0'],p['KV_relative_render']))
    lines+=['',summary['notes']];(dest/'results.md').write_text('\n'.join(lines)+'\n')
    snap=dest/'source_snapshot';snap.mkdir(exist_ok=True)
    names=['dual_view_both_low_train.py','dual_view_both_low_manifest.py','dual_view_both_low_capture.py','dual_view_both_low_evaluate.py','dual_view_both_low_report.py','dual_view_proxy.py','dual_view_probe.py','render_scale_manifest.py','render_scale_rollout.py','render_proxy_policy_server.py']
    for name in names:shutil.copy2(root/name,snap/name)
    files={str(p.relative_to(dest)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='provenance.json' and p.suffix!='.log'}
    (dest/'provenance.json').write_text(json.dumps({'complete':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files},indent=2)+'\n')
    print(json.dumps({'closed_loop':counts,'audit':audit}))

if __name__=='__main__':main()
