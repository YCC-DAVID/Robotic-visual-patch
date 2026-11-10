"""Aggregate and audit the completed geometry-only training and render test."""
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    from PIL import Image
    root=Path(__file__).resolve().parent
    dest=root/'out/perspective_only'
    assert json.loads((dest/'evaluation_complete.json').read_text())['complete']
    run=dest/'train_13_101_full'
    train=json.loads((run/'results.json').read_text())
    cfg=json.loads((run/'config.json').read_text())
    ck=np.load(run/'checkpoint_0500.npz',allow_pickle=False)
    ref=np.load(root/'out/render_proxy_scale/train_13_101_full/checkpoint_0500.npz',allow_pickle=False)
    assert train['iterations']==500 and train['gradient_frame_forwards']==4000
    assert np.array_equal(ck['batches'],ref['batches']) and np.array_equal(ck['noise'],ref['noise'])
    png=np.array(Image.open(run/'texture_0500.png'))
    assert np.array_equal(png,np.rint(ck['Q'].transpose(1,2,0)*255).clip(0,255).astype(np.uint8))
    combined={c:[] for c in ['clean','random','optimized']}
    raw_checks=0
    for stage in ['development','confirmation']:
        folder=dest/('rollout_'+stage)
        d=json.loads((folder/'episodes.json').read_text())
        for c in combined:combined[c]+=d['per_episode'][c]
        clean={r['init_index']:r for r in d['per_episode']['clean']}
        clean_root=Path(d['metadata']['clean_reference']).parent
        for condition in ['random','optimized']:
            for row in d['per_episode'][condition]:
                a=np.load(folder/row['replay_record'],allow_pickle=False)
                b=np.load(clean_root/clean[row['init_index']]['replay_record'],allow_pickle=False)
                assert np.array_equal(a['initial_state'],b['initial_state'])
                n=min(len(a['fixture_body_pos']),len(b['fixture_body_pos']))
                assert np.array_equal(a['fixture_body_pos'][:n],b['fixture_body_pos'][:n])
                assert np.array_equal(a['fixture_body_quat'][:n],b['fixture_body_quat'][:n])
                expected=20261006+row['init_index']*1000+np.arange(len(a['noise_seeds']))
                assert np.array_equal(a['noise_seeds'],expected)
                raw_checks+=1
    good={r['init_index'] for r in combined['clean'] if r['success']}
    assert len(good)==9 and all(len(v)==10 for v in combined.values())
    closed={c:{'n':len(rows),'robot_successes':sum(r['success'] for r in rows),
               'task_failures_on_clean_success':sum(not r['success'] for r in rows if r['init_index'] in good),
               'clean_success_denominator':len(good),
               'failed_clean_success_indices':[r['init_index'] for r in rows if r['init_index'] in good and not r['success']]}
            for c,rows in combined.items()}
    validation=json.loads((dest/'validation/summary.json').read_text())
    metrics=json.loads((dest/'validation/metrics.json').read_text())
    velocities=np.load(dest/'validation/velocities.npz',allow_pickle=False)
    vs=velocities['velocity'].astype(float);clean=velocities['clean_velocity'].astype(float)
    recompute={str(c):float(np.square(vs[:,i]-clean).sum(axis=(-1,-2,-3)).mean())
               for i,c in enumerate(velocities['conditions'])}
    maxerr=max(abs(recompute[c]-validation[c]['L0']) for c in recompute)
    assert maxerr<1e-4,maxerr
    assert len(metrics['records'])==128
    old=json.loads((root/'out/render_proxy_scale/validation_13_101_full/summary.json').read_text())
    summary={'complete':True,'closed_loop':closed,'validation':validation,'full_proxy_E24_validation':old,
             'training':train,'audit':{'matched_E24_all_500_batches':True,'matched_action_noise':True,
              'checkpoint_matches_uploaded_PNG':True,'paired_rollout_files_checked':raw_checks,
              'velocity_records_checked':128,'velocity_L0_max_recompute_error':maxerr},
             'notes':'13cm@101, one seed, 500x8. Pure perspective training uses geometry-only bilinear224; '
                     'E24 uses full calibrated imaging. Both were truly rendered in closed loop. '
                     '20 new episodes on 10 previously tested initial states, clean reference reused; paired ASR denominator9. '
                     '32 validation frames are correlated samples from two trajectories. proxy keys here mean pure perspective, not full E24 proxy.'}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (dest/'audit.json').write_text(json.dumps(summary['audit'],indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    axes[0].plot(np.arange(1,501),ck['history'],alpha=.35,color='#64748b')
    smooth=np.convolve(ck['history'],np.ones(20)/20,mode='valid')
    axes[0].plot(np.arange(20,501),smooth,color='#0369a1')
    axes[0].set_xlabel('PGD update');axes[0].set_ylabel('Training early L0')
    labels=['随机Q','纯透视训练Q','完整代理训练Q（E24）']
    counts=[closed['random']['task_failures_on_clean_success'],closed['optimized']['task_failures_on_clean_success'],3]
    axes[1].bar(np.arange(3),counts,color=['#94a3b8','#0f766e','#d97706'])
    axes[1].set_xticks(np.arange(3));axes[1].set_xticklabels(labels,fontproperties=font,fontsize=9)
    axes[1].set_ylim(0,9.7);axes[1].set_ylabel('任务失败数（攻击成功数）',fontproperties=font)
    for i,n in enumerate(counts):axes[1].text(i,n+.15,str(n)+'/9',ha='center')
    fig.suptitle('13 cm@101：训练与真实render闭环',fontproperties=font,fontsize=14)
    fig.savefig(dest/'training_and_rollout.png',dpi=160);plt.close(fig)
    inp=np.load(dest/'validation/inputs.npz',allow_pickle=False)
    images=np.load(dest/'validation/model_inputs.npz',allow_pickle=False)
    y,x=np.where(inp['mask'][0]);sy=slice(max(0,y.min()-3),min(224,y.max()+4));sx=slice(max(0,x.min()-3),min(224,x.max()+4))
    fig,axes=plt.subplots(2,5,figsize=(14,6),constrained_layout=True)
    for row,c in enumerate(['random','optimized']):
        p=images[c+'_proxy'][0];r=images[c+'_render'][0]
        for col,im in enumerate([inp[c+'__texture'],p,r,p[sy,sx],r[sy,sx]]):
            axes[row,col].imshow(im,interpolation='nearest');axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(['矩形世界Q128','同Q透视数字输入','同Q真实render','透视输入局部','真实render局部'][col],fontproperties=font,fontsize=11)
        axes[row,0].set_ylabel(['随机Q','优化Q'][row],fontproperties=font,fontsize=12)
    fig.savefig(dest/'learned_Q_and_images.png',dpi=160);plt.close(fig)
    lines=['# 13cm@101 pure perspective training result','','Robot task failure among clean successes = attack success.','',
           '| Condition, actual render | Robot task successes /10 | Task failures among clean-success /9 |',
           '|---|---:|---:|']
    for c in ['clean','random','optimized']:
        v=closed[c];lines.append('| {} | {}/10 | {}/9 |'.format(c,v['robot_successes'],v['task_failures_on_clean_success']))
    lines+=['','| Input | Early L0 vs clean |','|---|---:|']
    for c,v in validation.items():lines.append('| {} | {:.6f} |'.format(c,v['L0']))
    lines+=['','proxy means geometry-only perspective in this experiment. No full optical proxy in the training forward.',
            'E24 full-proxy reference on the same initial states:3/9 attack successes. Single seed, exploratory paired retest.',
            '','![Training and rollout](training_and_rollout.png)','','![Q and real images](learned_Q_and_images.png)']
    (dest/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
