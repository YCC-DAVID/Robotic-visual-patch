"""Method semantics, complete-flow derivatives, matching and held-out selection."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import e_method_common as C
from e_method_backend import EBackend,objective
from e_method_pipeline import choose,acceptance
import e_method_monitor as Monitor


class MethodTests(unittest.TestCase):
    def test_noninitial_coverage_preserves_sources_noises_and_clean_control(self):
        from e_method_coverage import coverage_draw_plan,development_replicas_pass
        from e_method_onpolicy import continuation_draw_plan
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);C.dump(root/'protocol.json',dict(method_study={'steps':3}))
            for prefix,manifest in [('old','collection_manifest.json'),('new','onpolicy_round/collection_manifest.json')]:
                rows=[dict(seed=s,query=q,draw=d,path=f'{prefix}_{s}_{q}_{d}') for s in [1,2] for q in ([0] if s==2 else [0,1]) for d in [0,1]]
                C.dump(root/manifest,dict(records=rows))
            a=continuation_draw_plan(root,42,'velocity_onpolicy_mix');b=coverage_draw_plan(root,42)
            self.assertEqual(b,coverage_draw_plan(root,42))
            for old,new in zip(a,b):
                self.assertEqual(old[:4],new[:4])
                self.assertEqual([r['seed'] for r in old],[r['seed'] for r in new])
                for x,y in zip(old[4:],new[4:]):
                    sx,qx,dx=x['record'].split('_')[1:];sy,qy,dy=y['record'].split('_')[1:]
                    self.assertEqual((sx,dx),(sy,dy));self.assertEqual(qy,'1' if sy=='1' else '0')
            data=dict(expected_seeds=32,rows=[dict(condition=c,completed_pairs=32,n=22,ASR=.36,excess_ASR=.23) for c in ['seed0','seed1','seed2']])
            settings={'target':dict(ASR=.3,excess_ASR=.2)}
            self.assertTrue(development_replicas_pass(data,['seed0','seed1','seed2'],settings))
            data['rows'][1]['ASR']=.27
            self.assertFalse(development_replicas_pass(data,['seed0','seed1','seed2'],settings))

    def test_continuations_match_shared_states_noises_and_eot_slots(self):
        from e_method_onpolicy import continuation_draw_plan,ONPOLICY_METHODS
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);C.dump(root/'protocol.json',dict(method_study={'steps':500}))
            for prefix,manifest in [('old','collection_manifest.json'),('new','onpolicy_round/collection_manifest.json')]:
                rows=[dict(seed=s,query=q,draw=d,path=f'{prefix}_{s}_{q}_{d}') for s in [1,2] for q in [0,1] for d in [0,1]]
                C.dump(root/manifest,dict(records=rows))
            control,mixed=[continuation_draw_plan(root,42,name) for name in ONPOLICY_METHODS]
            self.assertEqual(control,continuation_draw_plan(root,42,ONPOLICY_METHODS[0]))
            for a,b in zip(control,mixed):
                self.assertEqual(a[:4],b[:4])
                self.assertEqual([r['seed'] for r in a],[r['seed'] for r in b])
                self.assertEqual([r['record'].rsplit('_',1)[1] for r in a],[r['record'].rsplit('_',1)[1] for r in b])
                self.assertTrue(all(r['record'].startswith('old_') for r in a))
                self.assertTrue(all(r['record'].startswith('new_') for r in b[4:]))

    def test_continuation_uses_own_parent_for_each_replica(self):
        from e_method_onpolicy import parent_state_path,parent_texture
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);C.dump(root/'protocol.json',dict(method_study={'training_random_seeds':[10,11,12]}))
            for name,value in [('velocity_onpolicy_mix',.2),('velocity_onpolicy_mix_replica1',.7)]:
                path=parent_state_path(root,name);path.parent.mkdir(parents=True)
                np.savez(path,iteration=500,texture=np.full((1,3,128,128),value,np.float32))
                if '_replica' in name:C.dump(path.parent/'complete.json',dict(kind='velocity14',training_seed=11))
                else:C.dump(root/'onpolicy_round/protocol.json',dict(parent_state_sha256=C.sha(path)))
            self.assertAlmostEqual(float(parent_texture(root,'velocity_onpolicy_mix').mean()),.2,places=5)
            self.assertAlmostEqual(float(parent_texture(root,'velocity_onpolicy_mix_replica1').mean()),.7,places=5)
            p=parent_state_path(root,'velocity_onpolicy_mix_replica1')
            C.dump(p.parent/'complete.json',dict(kind='velocity14',training_seed=12))
            with self.assertRaises(AssertionError):parent_texture(root,'velocity_onpolicy_mix_replica1')

    def test_group_gradient_projection_and_mean_control(self):
        from e_method_groups import merge_gradients
        gradients={'a':np.array([1.,0.]),'b':np.array([-1.,1.])}
        mean,stats=merge_gradients(gradients)
        np.testing.assert_allclose(mean,[0,.5])
        self.assertEqual(stats['negative_pair_fraction'],1)
        projected,_=merge_gradients(gradients,project=True,seed=42)
        np.testing.assert_allclose(projected,[.25,.75])
        repeated,_=merge_gradients(gradients,project=True,seed=42)
        np.testing.assert_array_equal(projected,repeated)
        for original in gradients.values():self.assertGreaterEqual(float(projected@original),0)
        aligned={'a':np.array([1.,0.]),'b':np.array([1.,2.])}
        projected,_=merge_gradients(aligned,project=True)
        np.testing.assert_allclose(projected,[1.,1.])

    def test_group_draws_balance_groups_and_match_noise(self):
        from collections import Counter
        from e_method_groups import group_draw_plan
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);C.dump(root/'protocol.json',dict(method_study={'steps':500}))
            rows=[]
            for seed,(object_name,active) in enumerate([('bowl',0),('bowl',7),('cup',0),('cup',7)]):
                for query in [0,1]:
                    for draw in [0,1]:
                        path=root/f'{seed}_{query}_{draw}.npz'
                        np.savez(path,prompt=f'place the {object_name}',active_start=active)
                        rows.append(dict(seed=seed,query=query,draw=draw,path=str(path)))
            C.dump(root/'collection_manifest.json',dict(records=rows))
            a=group_draw_plan(root,42);b=group_draw_plan(root,42)
            self.assertEqual(a,b)
            self.assertTrue(all(len(batch)==8 and set(Counter(r['group'] for r in batch).values())=={2} for batch in a))
            nominal=sum(r['record'].endswith('_0.npz') for batch in a for r in batch)
            self.assertTrue(2800<nominal<3200)

    def test_balancing_scales_load_and_validate_actual_file(self):
        from e_method_backend import fixed_scales
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            C.dump(root/'refinement/protocol.json',dict(fixed_training_random_scales={'arm6':.02,'gripper':.003}))
            self.assertEqual(fixed_scales(root),{'arm6':.02,'gripper':.003})
            C.dump(root/'refinement/protocol.json',dict(fixed_training_random_scales={'arm6':.02,'gripper':0}))
            with self.assertRaises(AssertionError):fixed_scales(root)

    def test_actuator_losses_use_active_gripper_and_fixed_scales(self):
        x=torch.tensor(.4,requires_grad=True)
        clean=torch.zeros(1,2,32);v=clean.clone()
        v[...,7:13]=2*x;v[...,13]=3*x
        gripper=objective(v,clean,clean,clean,'velocity_gripper',7)
        self.assertAlmostEqual(float(gripper.detach()),9*.4**2,places=6)
        balanced=objective(v,clean,clean,clean,'velocity_joint_balanced',7,{'arm6':2,'gripper':3})
        self.assertAlmostEqual(float(balanced.detach()),2.5*.4**2,places=6)
        self.assertAlmostEqual(float(torch.autograd.grad(balanced,x)[0]),5*.4,places=6)
        self.assertEqual(float(objective(v,clean,clean,clean,'velocity_gripper',0)),0)

    def test_real_render_metrics_separate_discarded_and_active_dimensions(self):
        from e_method_probe import metrics
        clean={'v':np.zeros((1,2,32)),'action':np.zeros((1,2,32))}
        patched={'v':np.zeros((1,2,32)),'action':np.zeros((1,2,32))}
        patched['v'][...,20]=3;patched['v'][...,1]=4
        patched['action'][...,13]=2
        row=metrics(clean,patched,7)
        self.assertAlmostEqual(row['discarded_velocity_energy_fraction'],9/25)
        self.assertAlmostEqual(row['first14_mse'],16/14)
        self.assertEqual(row['active_first7_mse'],0)
        self.assertAlmostEqual(row['active_final7_mse'],4/7)
        self.assertEqual(row['active_gripper_mse'],4)

    def test_objective_selects_declared_dimensions(self):
        vc=torch.zeros(1,2,32);ac=torch.zeros_like(vc)
        v=vc.clone();v[...,20]=8
        a=ac.clone();a[...,2]=2;a[...,9]=3
        self.assertEqual(float(objective(v,a,vc,ac,'velocity14')),0)
        self.assertEqual(float(objective(v,a,vc,ac,'velocity32')),2)
        self.assertAlmostEqual(float(objective(v,a,vc,ac,'action14')),13/14,places=6)
        self.assertAlmostEqual(float(objective(v,a,vc,ac,'action_active7',7)),9/7,places=6)
        self.assertAlmostEqual(float(objective(v,a,vc,ac,'action_active7',0)),4/7,places=6)

    def test_checkpointed_ten_steps_forward_and_derivative(self):
        backend=EBackend.__new__(EBackend)
        backend.noise=lambda seed:torch.ones(1,2,32)
        backend.M=SimpleNamespace(prefix=lambda model,prepared:(prepared,torch.ones(1,1),None))
        backend.model=SimpleNamespace(denoise_step=lambda state,pad,kv,x,t:.1*x+.05*state*t)
        x=torch.tensor(.4,requires_grad=True)
        _,a=backend.response(x,42,True)
        gradient=torch.autograd.grad(a.square().mean(),x)[0]
        def value(number):
            with torch.no_grad():return float(backend.response(torch.tensor(number),42,True)[1].square().mean())
        h=.002
        self.assertAlmostEqual(float(gradient),(value(.4+h)-value(.4-h))/(2*h),places=4)
        manual=torch.ones(1,2,32);t=torch.tensor(1.)
        for _ in range(10):manual=manual-.1*(.1*manual+.05*x*t);t=t-.1
        torch.testing.assert_close(a,manual)

    def test_draws_shared_and_three_cohorts_disjoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=C.prepare(root);groups=p['method_study']
            tr,dev,test=[set(groups[k+'_seeds']) for k in ['training','development','test']]
            self.assertFalse(tr&dev or tr&test or dev&test)
            rows=[dict(seed=s,query=q,draw=d,path=f'{s}_{q}_{d}') for s in [1,2] for q in [0,1] for d in [0,1]]
            C.dump(root/'collection_manifest.json',dict(records=rows))
            a=C.draw_plan(root,42);b=C.draw_plan(root,42)
            self.assertEqual(a,b);self.assertEqual(len(a),500)
            self.assertTrue(all(len(batch)==8 for batch in a))
            nominal=sum(r['record'].endswith('_0') for batch in a for r in batch)
            self.assertTrue(2800<nominal<3200)

    def test_real_pairing_excludes_natural_failure_and_runtime_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);C.dump(root/'protocol.json',dict(method_study={'development_seeds':[1,2,3,4]}))
            for seed in [1,2,3,4]:
                for condition in ['clean','random','candidate']:
                    row=dict(job=dict(seed=seed,condition=condition),status='complete',
                        success=(seed!=3 if condition=='clean' else seed!=1 if condition=='random' else False))
                    if seed==4 and condition=='candidate':row.update(status='runtime_error',error='OOM')
                    C.dump(root/'cohorts/development/episodes'/f'{seed}_{condition}.json',row)
            data=C.results(root,'development',['candidate']);row=data['rows'][0]
            self.assertEqual(row['n'],2);self.assertEqual(row['ASR'],1)
            self.assertEqual(row['random_ASR'],.5);self.assertEqual(row['excess_ASR'],.5)
            self.assertEqual(len(data['errors']),1)

    def test_selection_and_all_replicas_must_pass(self):
        settings=dict(target=dict(ASR=.3,excess_ASR=.2,positive_paired_CI=True,min_test_clean_successes=50))
        rows=[dict(condition=name,completed_pairs=32,n=20,ASR=.4,excess_ASR=.2) for name in C.OBJECTIVES]
        selected=choose(dict(rows=rows,expected_seeds=32),settings)
        self.assertEqual(selected['objective'],'velocity14');self.assertTrue(selected['development_target_met'])
        rows=[dict(condition=name,completed_pairs=100,n=66,ASR=.4,excess_ASR=.25,excess_CI95=[.1,.4])
              for name in ['winner','replica1','replica2']]
        rows[-1]['excess_CI95']=[-.01,.4]
        decision=acceptance(dict(rows=rows,expected_seeds=100),['winner','replica1','replica2'],settings)
        self.assertFalse(decision['passed'])
        self.assertFalse(decision['replicas']['replica2']['passed'])

    def test_monitor_calls_for_next_method_work_below_target(self):
        row=dict(root='/tmp/study',summary={},status={'stage':'round_complete_below_target'},
                 checked_at_local='now',health='needs_method_iteration',problems=[],stage_remaining_s=None)
        message=Monitor.message(row,'test')
        self.assertIn('继续设计和执行下一轮',message)
        self.assertIn('不得根据最终测试结果',message)

    def test_monitor_reports_paired_asr_and_group_round(self):
        result=dict(condition='velocity_pcgrad',n=22,failures=6,ASR=6/22,random_ASR=3/22,excess_ASR=3/22)
        row=dict(root='/tmp/study',summary={'group_round':{},'results':{'development':{'rows':[result]}}},
                 status={'stage':'training_gradient_conflict'},completed=4,total=49,
                 checked_at_local='now',health='running',problems=[],stage_remaining_s=None)
        message=Monitor.message(row,'test')
        self.assertIn('当前第三轮',message)
        self.assertIn('ASR 6/22=27.3%',message)
        self.assertIn('+13.6个百分点',message)
        self.assertIn('独立测试集尚无',message)
        row['summary']['onpolicy_round']={'round':4}
        message=Monitor.message(row,'test')
        self.assertIn('当前第四轮',message)
        self.assertIn('训练轨迹结果不作为闭环ASR',message)
        row['summary']['coverage_round']={'round':5}
        message=Monitor.message(row,'test')
        self.assertIn('当前第五轮',message)
        self.assertIn('不降低门槛',message)


if __name__=='__main__':unittest.main()
