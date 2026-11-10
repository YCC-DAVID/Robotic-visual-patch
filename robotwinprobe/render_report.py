"""Readable progress/results with an explicit paired clean-success denominator."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='robotwinprobe/out/render_transfer')
    args=ap.parse_args();root=Path(args.root)
    clean=json.loads((root/'clean/results.json').read_text())
    baseline={e['seed']:e for e in clean['episodes'] if 10200<=e['seed']<10210}
    assert len(baseline)==10 and clean['complete']
    denominator=sum(e['success'] for e in baseline.values())
    labels=['random_digital','random_render','digital_full_digital','digital_full_render',
            'digital_band32_digital','digital_band32_render','proxy_full_render','proxy_band32_render']
    rows=[]
    for name in labels:
        p=root/name/'results.json'
        if not p.exists():rows.append({'condition':name,'complete':False,'episodes':0});continue
        data=json.loads(p.read_text());episodes=data['episodes'];fails=[];rescues=[]
        for e in episodes:
            seed=e['seed'];ref=baseline[seed]
            if ref['success'] and not e['success']:fails.append(seed)
            if not ref['success'] and e['success']:rescues.append(seed)
            with np.load(root/name/f'seed{seed}.npz') as record, np.load(root/'clean'/f'seed{seed}.npz') as reference:
                assert np.array_equal(record['states14'][0],reference['states14'][0]),(name,seed,'initial state')
                for camera in ('head_camera','left_camera','right_camera'):
                    assert np.array_equal(record['initial_clean_'+camera],reference[camera][0]),(name,seed,camera)
                assert np.array_equal(record['noise_seeds'],20261006+seed*1000+np.arange(len(record['noise_seeds'])))
        rows.append({'condition':name,'complete':data['complete'],'episodes':len(episodes),
                     'task_successes':sum(e['success'] for e in episodes),'attack_successes':len(fails),
                     'paired_clean_success_denominator':denominator,'attack_failure_seeds':fails,
                     'rescued_clean_failure_seeds':rescues})
    summary={'complete':all(r['complete'] for r in rows),'clean_test_successes':denominator,'clean_test_episodes':10,'rows':rows,
             'caution':'One task, one training seed, ten paired test episodes. Incomplete rows are not final ASR.'}
    (root/'rollout_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['| 条件 | 已完成轨迹 | 任务成功 | 攻击成功（clean 成功→任务失败） |','|---|---:|---:|---:|',f'| clean | 10/10 | {denominator}/10 | — |']
    for r in rows:
        done=r['complete'];n=r['episodes']
        lines.append(f"| {r['condition']} | {n}/10 | {str(r.get('task_successes'))+'/'+str(n) if n else '待跑'} | {str(r['attack_successes'])+'/'+str(denominator) if done else '未完成，不报最终比例'} |")
    (root/'rollout_results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
