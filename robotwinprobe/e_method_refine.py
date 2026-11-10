"""Controlled actuator-weighting and training-group E refinement rounds."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

from e_method_common import (DEFAULT,ROOT,SOURCE,PY,OBJECTIVES,REFINE_OBJECTIVES,BASELINES,SCREEN,
    prepare,load,dump,sha,draw_plan,request,results)
from e_method_pipeline import Pipeline,acceptance
from e_method_groups import GROUP_METHODS,prepare_group_round
from e_method_onpolicy import ONPOLICY_METHODS,prepare_onpolicy
from e_method_coverage import COVERAGE_METHODS,prepare_coverage,development_replicas_pass


def prepare_refinement(runner):
    root=runner.root;path=root/'refinement/protocol.json'
    if path.exists():
        protocol=json.loads(path.read_text())
        assert protocol['base_protocol_sha256']==sha(root/'protocol.json')
        assert protocol['collection_sha256']==sha(root/'collection_manifest.json')
        return protocol
    runner.status('refinement_training_scale')
    from render_geometry import patterns
    random=patterns()['random'].transpose(2,0,1)[None].astype(np.float32)/255
    draws=[row for batch in draw_plan(root,runner.settings['training_random_seeds'][0])[:4] for row in batch]
    values=[]
    for row in draws:
        response=request(dict(command='method_gradient',kind='velocity14',texture=random,**row),root)
        values.append(dict(arm6=response['active_arm6_mse'],gripper=response['active_gripper_mse']))
    scales={name:float(np.mean([v[name] for v in values])) for name in ('arm6','gripper')}
    assert all(np.isfinite(v) and v>1e-16 for v in scales.values()),scales
    protocol=dict(version=1,round=2,objectives=list(REFINE_OBJECTIVES),
        hypothesis='Initial real-render L0 already transfers to effective actions; gripper response grows much less. Test task-sensitive actuator emphasis, not proved causality.',
        base_protocol_sha256=sha(root/'protocol.json'),collection_sha256=sha(root/'collection_manifest.json'),
        diagnostic_sha256=sha(root/'transfer_diagnostic/summary.json'),
        initialization='same gray .5',steps=500,batch=8,lr=.01,
        fixed_training_random_scales=scales,scale_draws=draws,scale_values=values,
        velocity_gripper='mean squared first-step velocity difference of active gripper over native32 horizon',
        velocity_joint_balanced='0.5 * active six-joint MSE / fixed arm6 Random mean + 0.5 * active-gripper MSE / fixed gripper Random mean',
        matching='identical500x8 state/noise/EOT draws and initialization to first-round seed202610270',
        development='same32 seeds, paired baseline records and GPU assignments reused',
        final_test='still sealed; no final-test results used for refinement or target selection',
        stopping='fixed500 checkpoint; development must reach existing ASR/excess threshold before replica and independent-test stages')
    dump(path,protocol)
    return protocol


def audit_objectives(runner):
    root=runner.root;output=root/'audit/refinement_gradients.json'
    if output.exists():return
    record=draw_plan(root,runner.settings['training_random_seeds'][0])[0][0]
    q=np.full((1,3,128,128),.5,np.float32);rows={}
    for kind in REFINE_OBJECTIVES:
        r=request(dict(command='method_gradient',kind=kind,texture=q,**record),root)
        assert np.isfinite(r['loss']) and np.isfinite(r['gradient']).all() and r['gradient_norm']>0
        rows[kind]={k:v for k,v in r.items() if k!='gradient'}
    dump(output,dict(passed=True,record=record,rows=rows,
        objective_protocol_sha256=sha(root/'refinement/protocol.json')))


def select(root,settings,candidates):
    data=results(root,'development',candidates)
    assert not data['errors']
    rows={r['condition']:r for r in data['rows']}
    assert all(rows[name]['completed_pairs']==32 and rows[name]['n'] for name in candidates)
    best=max(candidates,key=lambda name:(rows[name]['excess_ASR'],rows[name]['ASR'],-candidates.index(name)))
    row=rows[best]
    passed=row['ASR']>=settings['target']['ASR'] and row['excess_ASR']>=settings['target']['excess_ASR']
    return dict(objective=best,development_result=row,development_target_met=passed,
        replica_conditions=[best+'_replica1',best+'_replica2'],
        criterion='development excess ASR, then ASR, then predeclared candidate order; no test used')


def run(args):
    prepare(args.root)
    with (args.root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runner=Pipeline(args)
        try:
            runner.snapshot_sources()
            original=json.loads((SOURCE/'provenance.json').read_text())['sha256']
            assert sha(Path(load(args.root)['policies']['pi'])/'model.safetensors')==original
            assert json.loads((args.root/'audit/development.json').read_text())['passed']
            assert json.loads((args.root/'transfer_diagnostic/summary.json').read_text())['complete']
            first_audit=args.root/'audit/development_round1.json'
            if not first_audit.exists():shutil.copy2(args.root/'audit/development.json',first_audit)
            if args.group_round:
                second_audit=args.root/'audit/development_round2.json'
                if not second_audit.exists():shutil.copy2(args.root/'audit/development.json',second_audit)
            if args.onpolicy_round:
                third_audit=args.root/'audit/development_round3.json'
                if not third_audit.exists():shutil.copy2(args.root/'audit/development.json',third_audit)
            if args.coverage_round:
                fourth_audit=args.root/'audit/development_round4.json'
                if not fourth_audit.exists():shutil.copy2(args.root/'audit/development.json',fourth_audit)
            runner.backend()
            if args.coverage_round:
                prepare_coverage(runner)
                methods=COVERAGE_METHODS
                conditions=(*SCREEN,*REFINE_OBJECTIVES,*GROUP_METHODS,*ONPOLICY_METHODS,*COVERAGE_METHODS)
                selection_path=args.root/'coverage_round/selection.json'
            elif args.onpolicy_round:
                prepare_onpolicy(runner)
                manifest=args.root/'onpolicy_round/collection_manifest.json'
                if not manifest.exists():
                    while args.sim_gpu not in runner.gpus():
                        runner.status('waiting_onpolicy_collection_gpu');time.sleep(10)
                    runner.call('onpolicy_state_collection',runner.command('e_method_onpolicy.py',sim=True),args.sim_gpu)
                data=json.loads(manifest.read_text());assert data['complete'] and data['no_outcome_filter']
                assert data['seeds']==runner.settings['training_seeds']
                assert data['collection_protocol_sha256']==sha(args.root/'onpolicy_round/protocol.json')
                for record in data['records']:assert sha(Path(record['path']))==record['sha256']
                dump(args.root/'audit/onpolicy_collection.json',dict(passed=True,initial_states=len(data['seeds']),
                    records=len(data['records']),collection_sha256=sha(manifest),test_used=False))
                methods=ONPOLICY_METHODS
                conditions=(*SCREEN,*REFINE_OBJECTIVES,*GROUP_METHODS,*ONPOLICY_METHODS)
                selection_path=args.root/'onpolicy_round/selection.json'
            elif args.group_round:
                prepare_group_round(runner)
                methods=GROUP_METHODS
                conditions=(*SCREEN,*REFINE_OBJECTIVES,*GROUP_METHODS)
                selection_path=args.root/'group_round/selection.json'
            else:
                prepare_refinement(runner);audit_objectives(runner)
                methods=REFINE_OBJECTIVES
                conditions=(*SCREEN,*REFINE_OBJECTIVES)
                selection_path=args.root/'refinement/selection.json'
            runner.snapshot_sources()
            for name in methods:
                runner.train(name,'velocity14' if name in (*GROUP_METHODS,*ONPOLICY_METHODS,*COVERAGE_METHODS) else name,runner.settings['training_random_seeds'][0])
            runner.evaluate('development',conditions);runner.audit('development',conditions)
            selection=select(args.root,runner.settings,tuple(c for c in conditions if c not in BASELINES))
            dump(selection_path,selection)
            if not selection['development_target_met'] and not args.coverage_round:
                runner.status(('onpolicy' if args.onpolicy_round else 'group' if args.group_round else 'refinement')+'_round_complete_needs_method_iteration',complete=False,
                    next_action='Analyze controlled results and continue method work; target not met. Held-out test remains untouched.')
                return
            winner=selection['objective'];replicas=selection['replica_conditions']
            loss_kind='velocity14' if winner in (*GROUP_METHODS,*ONPOLICY_METHODS,*COVERAGE_METHODS) else winner
            for index,(name,seed) in enumerate(zip(replicas,runner.settings['training_random_seeds'][1:]),1):
                if winner in (*ONPOLICY_METHODS,*COVERAGE_METHODS):runner.train('velocity14_parent_replica'+str(index),'velocity14',seed)
                runner.train(name,loss_kind,seed)
            runner.evaluate('development',(*conditions,*replicas));runner.audit('development',(*conditions,*replicas))
            if args.coverage_round:
                data=results(args.root,'development',(winner,*replicas))
                passed=selection['development_target_met'] and development_replicas_pass(data,(winner,*replicas),runner.settings)
                dump(args.root/'coverage_round/replication.json',dict(selection=selection,results=data,
                    all_development_targets_met=passed,test_used=False))
                if not passed:
                    runner.status('coverage_round_complete_needs_method_iteration',complete=False,
                        next_action='Analyze independent-parent training seed diagnostics and noninitial-state coverage; unchanged target not met; held-out test still sealed.')
                    return
            dump(args.root/'frozen_selection.json',dict(selection=selection,
                replica_textures={name:sha(args.root/'patches'/name/'texture_0500.png') for name in (winner,*replicas)},
                frozen_before_test_unix=time.time(),test_used_for_selection=False))
            final_conditions=(*BASELINES,winner,*replicas)
            runner.evaluate('test',final_conditions);runner.status('final_audit');runner.audit('test',final_conditions)
            decision=acceptance(results(args.root,'test',final_conditions),(winner,*replicas),runner.settings)
            dump(args.root/'acceptance.json',decision)
            runner.status('complete' if decision['passed'] else 'refinement_round_complete_needs_method_iteration',
                complete=decision['passed'],independent_test_complete=True,
                next_action='Report independent results; if below target keep test cohort sealed for future iterations.')
        except BaseException as error:
            runner.status('stopped_on_error',complete=False,error=repr(error),traceback=traceback.format_exc())
            raise
        finally:runner.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--model-gpu',default='4');parser.add_argument('--sim-gpu',default='2')
    parser.add_argument('--max-workers',type=int,default=2);parser.add_argument('--launch',action='store_true')
    rounds=parser.add_mutually_exclusive_group()
    rounds.add_argument('--group-round',action='store_true')
    rounds.add_argument('--onpolicy-round',action='store_true')
    rounds.add_argument('--coverage-round',action='store_true')
    args=parser.parse_args();args.root=args.root.resolve()
    if args.launch:
        prepare(args.root);(args.root/'logs').mkdir(exist_ok=True)
        with (args.root/'logs/refinement_pipeline.log').open('a') as log:
            argv=[v for v in sys.argv if v!='--launch']
            proc=subprocess.Popen([sys.executable,'-u',*argv],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        dump(args.root/'refinement_launch.json',dict(pid=proc.pid,argv=argv,started_unix=time.time()))
        print(json.dumps(dict(pid=proc.pid,root=str(args.root))))
    else:run(args)
