"""Post-hoc paired origin-to-moved attack retention from existing closed loops."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from radius_plan import dump


def analyze(root):
    seeds=json.loads((root/'cohort.json').read_text())['seeds']
    sources={}
    def read(path):
        raw=path.read_bytes();sources[str(path.relative_to(root))]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    clean={s:read(root/'eval/clean'/f'seed{s}_p00.json') for s in seeds}
    methods=['baseline','r01','r03','r05','random'];rows=[];per_seed={}
    indices=np.random.RandomState(2026101203).randint(len(seeds),size=(20000,len(seeds)))
    for method in methods:
        counts=[]
        for seed in seeds:
            origin=read(root/'eval'/method/f'seed{seed}_p00.json')
            moved=[read(root/'eval'/method/f'seed{seed}_p{j:02d}.json') for j in range(1,11)]
            for episode in [origin]+moved:
                for key in ['initial_rgb_sha256','initial_state_sha256','first_noise_seed','prompt']:
                    assert episode[key]==clean[seed][key],(method,seed,key)
                assert episode['texture_sha256']==origin['texture_sha256']
            assert origin['radius_cm']==0 and all(0<r['radius_cm']<=5+1e-8 for r in moved)
            if not clean[seed]['success']:
                counts.append([0,0,0,0]);continue
            a=not origin['success'];b=np.array([not r['success'] for r in moved])
            # Both attack; origin-only attack lost; moved-only attack gained; neither.
            counts.append([int(b.sum()) if a else 0,int((~b).sum()) if a else 0,
                           int(b.sum()) if not a else 0,int((~b).sum()) if not a else 0])
        counts=np.array(counts);kept,lost,gained,neither=counts.sum(0);den=counts.sum()
        origin_den=kept+lost
        boot=counts[indices].sum(1);total=boot.sum(1);bd=boot[:,0]+boot[:,1]
        delta=(gained-lost)/den
        row=dict(method=method,origin_attacks=int(origin_den//10),clean_successes=int(den//10),
            origin_ASR=float(origin_den/den),moved_ASR=float((kept+gained)/den),paired_ASR_change=float(delta),
            kept=int(kept),lost=int(lost),gained=int(gained),neither=int(neither),
            origin_attack_position_pairs=int(origin_den),retention=float(kept/origin_den) if origin_den else None,
            delta_cluster_95=np.quantile((boot[total>0,2]-boot[total>0,1])/total[total>0],[.025,.975]).tolist(),
            retention_cluster_95=np.quantile(boot[bd>0,0]/bd[bd>0],[.025,.975]).tolist() if (bd>0).any() else None)
        assert np.isclose(row['moved_ASR']-row['origin_ASR'],delta)
        assert kept+lost+gained+neither==den
        rows.append(row);per_seed[method]=counts.tolist()
    result=dict(rows=rows,initial_states=len(seeds),positions_per_seed=10,
        definitions=dict(ASR_change='ASR_moved - ASR_origin = (gained - lost) / clean_success_position_pairs',
                         retention='P(attack at moved position | attack at origin, clean success)',
                         transition_order=['kept','lost','gained','neither']),
        uncertainty='20000 paired bootstrap draws over initial states, retaining all ten positions. Pointwise post-hoc intervals; one training seed.',
        limitation='Retention uses a different origin-attack subset for each method; interpret with absolute ASR and matched Random.',
        per_seed=per_seed,seeds=seeds,source_sha256=sources)
    dump(root/'origin_retention.json',result)
    lines=['# 从原位置到随机位置：攻击效果保留','',
        '使用原有 50 个初态、每个初态 10 个共同 5 cm 圆盘随机位置，不新增训练或 rollout。Clean 成功 30/50。',
        '', '`ΔASR = 移动 ASR − 原点 ASR`。攻击保留率仅在“clean 成功且原点攻击成功”的初态中，统计移动后仍攻击成功的比例。',
        '', '|方法|原点 ASR|移动 ASR|ΔASR（百分点）|移动后保留攻击 / 原点攻击位置对|保留率|',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['method']}|{r['origin_ASR']:.1%}|{r['moved_ASR']:.1%}|{100*r['paired_ASR_change']:+.1f}|{r['kept']}/{r['origin_attack_position_pairs']}|{r['retention']:.1%}|")
    lines+=['', '例如 baseline 原点攻击成功的 15 个初态，各移动 10 次，150 个配对中只有 41 个仍攻击成功；另有 14 个原点攻击未成功的配对在移动后成功。因此移动 ASR 为 (41+14)/300，而保留率为 41/150。',
        '', 'Random 的平均 ASR 维持 20%，但仍有 24 个原点有效的配对失效、另外 24 个配对新生效；平均值不变不等于逐场景稳定。',
        '', '各方法保留率的条件子集不同；高保留率需要与移动后的绝对 ASR、同条件 Random 一起解释。按初态聚类的逐项 95% 区间及逐初态转换计数保存在 `origin_retention.json`，不包含重新训练的不确定性。']
    (root/'origin_retention.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    analyze(Path(parser.parse_args().root).resolve())
