"""Audit paired clean references, summarize completed robustness cells only."""
import argparse, json
import numpy as np
import bootstrap as B
from eot_conditions import evaluation_conditions
from render_geometry import CAMERAS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args()
    root=B.Path(args.root).resolve();rows=[]
    for name,c in evaluation_conditions().items():
        clean_name='light_dim' if c['light_gain']==[.85]*3 else 'light_bright' if c['light_gain']==[1.15]*3 else 'nominal'
        clean_path=root/'eval'/clean_name/'clean'
        if not (clean_path/'results.json').exists():continue
        clean=json.loads((clean_path/'results.json').read_text())
        if not clean['complete']:continue
        cs={e['seed']:e['success'] for e in clean['episodes']}
        for mode in ['random','baseline','position','light','joint']:
            path=root/'eval'/name/mode
            if not (path/'results.json').exists():continue
            result=json.loads((path/'results.json').read_text())
            if not result['complete']:continue
            assert set(cs)=={e['seed'] for e in result['episodes']}
            for e in result['episodes']:
                with np.load(path/f"seed{e['seed']}.npz") as a,np.load(clean_path/f"seed{e['seed']}.npz") as b:
                    assert np.array_equal(a['states14'][0],b['states14'][0])
                    for n in CAMERAS:assert np.array_equal(a['initial_clean_'+n],b['initial_clean_'+n]), (name,mode,e['seed'],n)
                    assert a['noise_seeds'][0]==b['noise_seeds'][0]
            failed=[e['seed'] for e in result['episodes'] if cs[e['seed']] and not e['success']]
            rows.append({'condition':name,'mode':mode,'task_successes':result['task_successes'],
                'episodes':len(cs),'clean_successes':sum(cs.values()),'attack_successes':len(failed),'failed_seeds':failed})
    report={'complete':len(rows)==35,'rows':rows,'paired_initial_RGB_state_noise_audited':True,
            'scope':'one training seed, five fresh test initial states, seven held-out transform settings'}
    (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# RoboTwin 位置／光照 EOT 闭环结果','',
        '攻击成功：同光照 clean 成功、贴图后失败。未完成的单元不计入结果。','',
        '| 测试条件 | 训练方式 | 任务成功 | clean 成功 | 攻击成功 |',
        '|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['condition']} | {r['mode']} | {r['task_successes']}/{r['episodes']} | {r['clean_successes']}/{r['episodes']} | {r['attack_successes']}/{r['clean_successes']} |")
    (root/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
