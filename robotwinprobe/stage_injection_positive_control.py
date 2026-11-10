"""Predeclare the one prior trained-only failure as a separate case study.

This cohort is selected using the frozen earlier pilot, before delayed-onset
outcomes are measured. It must not be pooled into the independent new cohort.
"""
import argparse
import json
from pathlib import Path

import bootstrap as B
from radius_plan import dump
import stage_injection as S


def prepare(root):
    if (root/'protocol.json').exists():
        return S.load_protocol(root)
    prior = B.OUT/'table9_closedloop_10cm_20261013'
    eligible = []
    evidence = {}
    for seed in range(10500,10510):
        paths = [prior/'episodes'/name for name in [f'clean_{seed}_nominal.json',
            f'site07_{seed}_nominal_0_trained.json',f'site07_{seed}_nominal_0_random.json']]
        clean,trained,random = [json.loads(p.read_text()) for p in paths]
        if clean['success'] and not trained['success'] and random['success']:
            eligible.append(seed)
            evidence.update({str(p):S.sha(p) for p in paths})
    assert eligible == [10504], 'prior pilot changed; inspect selection before rerunning'
    args = S.parser().parse_args(['prepare','--root',str(root),'--seeds',','.join(map(str,eligible))])
    p = S.prepare(args)
    prior_protocol = json.loads((prior/'protocol.json').read_text())
    assert prior_protocol['noise_rule'] == '20261013 + seed*1000 + query'
    assert p['textures_sha256']['trained'] == prior_protocol['texture_sha256'][str(S.DEFAULT_TEXTURE)]
    assert p['textures_sha256']['random'] == prior_protocol['texture_sha256'][str(prior/'random.png')]
    p['config']['noise_base'] = 20261013
    p['config']['cohort_type'] = 'prior trained-only attack-success case; illustrative positive control, not independent validation'
    p['config']['selection_rule'] = 'all prior site07 nominal clean successes with trained failure and Random success; selected before delayed-onset measurements'
    p['config']['prior_evidence_sha256'] = evidence
    p['config']['prior_protocol_sha256'] = S.sha(prior/'protocol.json')
    name = Path(__file__).name
    p['source_sha256'][name] = S.sha(__file__)
    (root/'source_snapshot'/name).write_bytes(Path(__file__).read_bytes())
    dump(root/'protocol.json',p)
    return p


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default=str(B.OUT/'stage_injection_positive_control_20261016'))
    args = ap.parse_args()
    p = prepare(Path(args.root).resolve())
    print(json.dumps(dict(seeds=p['config']['seeds'],noise_base=p['config']['noise_base'],
                          cohort_type=p['config']['cohort_type'])),flush=True)
