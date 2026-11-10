"""Audit all eight frozen runs and report paired actual-render outcomes."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'out/proxy_factorial'
ORDER=['000','100','010','001','110','101','011','111']
LABELS=['skeleton','+M','+A','+C','full-C','full-A','full-M','full']


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert read(DEST/'complete.json')['complete']
    baseline=read(DEST/'baseline/episodes.json')
    clean={r['init_index']:r['success'] for r in baseline['per_episode']['clean']}
    assert len(clean)==10
    eligible=sorted(k for k,v in clean.items() if v)
    first=None;rows={};textures=[];source_hashes={};random_hash=None
    same_fields=['batches','noise']
    old=np.load(ROOT/'out/render_proxy_scale/train_13_101_full/checkpoint_0500.npz')
    for variant in ORDER:
        folder=DEST/('train_'+variant)
        cfg=read(folder/'config.json');result=read(folder/'results.json')
        assert cfg['complete'] and result['complete'] and result['iterations']==500
        assert result['gradient_frame_forwards']==4000 and cfg['variant']==variant
        with np.load(folder/'checkpoint_0500.npz') as p:
            if first is None:first={k:p[k].copy() for k in same_fields}
            assert all(np.array_equal(first[k],p[k]) for k in same_fields)
            assert np.array_equal(p['batches'],old['batches']) and np.array_equal(p['noise'],old['noise'])
            q=p['Q'].copy();history=p['history'].copy()
            if variant=='111':old_repeat={'Q_exact':bool(np.array_equal(q,old['Q'])),
                                         'history_exact':bool(np.array_equal(history,old['history'])),
                                         'Q_max_abs_diff':float(abs(q-old['Q']).max())}
        from PIL import Image
        texture=np.asarray(Image.open(folder/'texture_0500.png').convert('RGB'))
        assert np.array_equal(texture,np.rint(q.transpose(1,2,0)*255).clip(0,255).astype(np.uint8))
        textures.append(texture)
        manifest=read(DEST/f'manifest_{variant}/manifest.json')
        h=sha(manifest['textures']['random'])
        if random_hash is None:random_hash=h
        assert h==random_hash
        assert sha(folder/'texture_0500.png')==manifest['texture_sha256']['optimized']==result['texture_sha256']
        rollout=read(DEST/f'rollout_{variant}/episodes.json')
        episodes=rollout['per_episode']['optimized'];assert sorted(e['init_index'] for e in episodes)==list(range(40,50))
        outcomes={e['init_index']:not e['success'] for e in episodes if e['init_index'] in eligible}
        metric=read(DEST/f'validation_{variant}/metrics.json');assert metric['complete'] and metric['variant']==variant
        summary=metric['summary'];proxy=summary['optimized_proxy'];real=summary['optimized_render']
        rows[variant]={'task_successes':sum(e['success'] for e in episodes),
                      'Q_changed_texel_fraction':float(np.any(q != .5,axis=0).mean()),
                      'paired_failures':sum(outcomes.values()),'clean_success_n':len(eligible),
                      'paired_ASR':sum(outcomes.values())/len(eligible),
                      'paired_outcomes':outcomes,'real_L0':real['L0'],'proxy_L0':proxy['L0'],
                      'proxy_render_KV_relative':proxy['KV_relative_render'],
                      'proxy_render_velocity_sqdiff':proxy['paired_L0_vs_render'],
                      'train_seconds':result['elapsed_training_s']}
        source_hashes[variant]={'config':sha(folder/'config.json'),'texture':sha(folder/'texture_0500.png'),
                                'rollout':sha(DEST/f'rollout_{variant}/episodes.json'),
                                'model_metrics':sha(DEST/f'validation_{variant}/metrics.json')}
    effects={}
    for index,name in enumerate('MAC'):
        alone=['0']*3;alone[index]='1';alone=''.join(alone)
        missing=['1']*3;missing[index]='0';missing=''.join(missing)
        deltas=[]
        for other in itertools.product('01',repeat=2):
            low=list(other);low.insert(index,'0');low=''.join(low)
            high=low[:index]+'1'+low[index+1:]
            deltas.append(rows[high]['paired_ASR']-rows[low]['paired_ASR'])
        effects[name]={'add_alone_count_delta':rows[alone]['paired_failures']-rows['000']['paired_failures'],
                       'add_to_other_two_count_delta':rows['111']['paired_failures']-rows[missing]['paired_failures'],
                       'average_ASR_difference_across_other_factors':float(np.mean(deltas))}
    random_summary=baseline['summary']['random']
    result={'complete':True,'rows':rows,'effects_descriptive':effects,'eligible_initial_states':eligible,
            'clean_task_successes':sum(clean.values()),'random':random_summary,
            'old_full_repeat':old_repeat,'checks':{'all_batches_and_noise_equal':True,'match_old_E24_batches_noise':True,
            'PNG_matches_Q':True,'random_identical_all_groups':True,'all_iterations_500':True},
            'provenance':source_hashes,'notes':'One training seed/task/geometry; reused10test states. Factor averages are descriptive, not confidence intervals or proof of universal necessity.'}
    (DEST/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 固定透视骨架的代理单加／单减结果','',
           f'Clean任务成功{sum(clean.values())}/10；random配对攻击成功{random_summary["paired_ASR_failures"]}/{len(eligible)}。',
           '', '| MAC | 含义 | 任务成功/10 | 配对攻击成功 | 真实L0 | 训练代理L0 | 代理/真实KV距离 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for v,label in zip(ORDER,LABELS):
        r=rows[v];lines.append(f'| {v} | {label} | {r["task_successes"]} | {r["paired_failures"]}/{len(eligible)} | {r["real_L0"]:.5f} | {r["proxy_L0"]:.5f} | {r["proxy_render_KV_relative"]:.5f} |')
    lines+=['','单加增量=只加该项−000；补齐增量=111−缺该项，单位为配对失败集数。',
            '', '| 因素 | 单加增量 | 补齐增量 |', '|---|---:|---:|']
    for k,v in effects.items():lines.append(f'| {k} | {v["add_alone_count_delta"]:+d} | {v["add_to_other_two_count_delta"]:+d} |')
    lines+=['','L0为第一步速度平方差，不是KV或任务失败率。只报告当前单seed探索性对照，不作普遍必要性结论。']
    (DEST/'results.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    x=np.arange(8)
    axes[0].bar(x,[rows[v]['paired_failures'] for v in ORDER])
    axes[0].set_xticks(x,LABELS,rotation=30);axes[0].set_ylim(0,len(eligible)+.6)
    axes[0].set_ylabel(f'Task failures / {len(eligible)} clean successes')
    axes[0].set_title('Actual MuJoCo closed loop')
    axes[1].plot(x,[rows[v]['proxy_L0'] for v in ORDER],'o-',label='training proxy')
    axes[1].plot(x,[rows[v]['real_L0'] for v in ORDER],'s-',label='actual render')
    axes[1].set_xticks(x,LABELS,rotation=30);axes[1].set_ylabel('Early velocity squared difference')
    axes[1].set_title('Same texture, paired model inputs');axes[1].legend()
    fig.tight_layout();fig.savefig(DEST/'fig_factorial_results.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(10,5))
    for ax,v,label,texture in zip(axes.flat,ORDER,LABELS,textures):
        ax.imshow(texture,interpolation='nearest');ax.set_title(v+' '+label);ax.axis('off')
    fig.tight_layout();fig.savefig(DEST/'fig_factorial_textures.png',dpi=160);plt.close(fig)
    print(json.dumps({v:rows[v]['paired_failures'] for v in ORDER}),flush=True)


if __name__=='__main__':main()
