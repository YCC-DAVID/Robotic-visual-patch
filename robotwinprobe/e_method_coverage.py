"""Noninitial attacked-state coverage and independent training-seed diagnostics."""
import json

import numpy as np

from e_method_common import dump,load,sha
from e_method_onpolicy import grouped_records,continuation_draw_plan

COVERAGE_METHODS=('velocity_onpolicy_noninitial',)


def coverage_draw_plan(root,seed):
    rows=json.loads((root/'onpolicy_round/collection_manifest.json').read_text())['records']
    groups=grouped_records(rows);lookup={row['path']:row for row in rows}
    draws=continuation_draw_plan(root,seed,'velocity_onpolicy_mix')
    rng=np.random.default_rng(seed+4_000_000)
    for batch in draws:
        for slot in range(4,8):
            row=lookup[batch[slot]['record']];pool=groups[row['seed']]
            eligible=sorted(q for q in pool if q>0)
            # Retain every training initial seed even if only query0 exists.
            query=int(rng.choice(eligible if eligible else sorted(pool)))
            batch[slot]['record']=pool[query][row['draw']]
    return draws


def prepare_coverage(runner):
    root=runner.root;path=root/'coverage_round/protocol.json'
    rows=json.loads((root/'onpolicy_round/collection_manifest.json').read_text())['records']
    groups=grouped_records(rows)
    config=dict(round=5,methods=list(COVERAGE_METHODS),loss_kind='velocity14',
        base_protocol_sha256=sha(root/'protocol.json'),onpolicy_protocol_sha256=sha(root/'onpolicy_round/protocol.json'),
        attacked_collection_sha256=sha(root/'onpolicy_round/collection_manifest.json'),
        parent_state_sha256=sha(root/'training/velocity14/state.npz'),
        clean_collection_sha256=sha(root/'collection_manifest.json'),
        hypothesis='half the attacked-state bank is query0; test concentrating the four attacked slots on post-action states, without filtering any source initial seed',
        budget='same parent500 + continuation500x8; same first14 L0, lr .01 and fixed final checkpoint',
        sampling='reuse fourth-round mix first4 clean slots, all noises/EOT slots, and source seeds; attacked slots prefer query>0 uniformly within each seed; fallback to existing queries if no noninitial query',
        training_initial_seeds=runner.settings['training_seeds'],
        noninitial_queries_per_seed={str(s):len([q for q in queries if q>0]) for s,queries in groups.items()},
        replication='after candidate development comparison, retrain the best method using two independent gray-initialized parent500 runs and continuation500 runs; pilot attacked bank remains fixed',
        diagnostic_replication='run even when seed0 misses the development gate; do not lower the gate or open held-out test for that diagnostic',
        heldout_gate='original seed0 gate AND both training replicas meet unchanged ASR/excess thresholds on development before any independent test',
        development='same32 initial states/noises and preserved GPU assignments; all previous records retained',
        independent_test='sealed; no test-driven training data, checkpoint, denominator, or threshold changes')
    if path.exists():assert json.loads(path.read_text())==config
    else:dump(path,config)
    return config


def development_replicas_pass(data,conditions,settings):
    rows={r['condition']:r for r in data['rows']};target=settings['target']
    return all(rows[c]['completed_pairs']==data['expected_seeds'] and rows[c]['n'] and
        rows[c]['ASR']>=target['ASR'] and rows[c]['excess_ASR']>=target['excess_ASR'] for c in conditions)
