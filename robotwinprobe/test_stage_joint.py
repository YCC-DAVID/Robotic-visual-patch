"""CPU regression tests for log gradients, event onset, replay, and pulse removal."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import stage_joint_common as C
import stage_joint_rollout as R
from stage_joint_train import objective_weight


class RecipeTests(unittest.TestCase):
    def test_balanced_gradient_matches_finite_difference(self):
        scale=.07;x=.4
        def loss(x):
            return np.mean([np.log1p((a*x+b)**2/scale) for a,b in [(1,.2),(5,1),(-2,.3),(.3,.01)]])
        gradient=0
        for a,b in [(1,.2),(5,1),(-2,.3),(.3,.01)]:
            l0=(a*x+b)**2;_,weight=objective_weight('joint_balanced',l0,scale)
            gradient+=weight*2*a*(a*x+b)/4
        h=1e-6
        self.assertAlmostEqual(gradient,(loss(x+h)-loss(x-h))/(2*h),places=6)
        self.assertEqual(objective_weight('joint_mean',2,4),(.5,.25))

    def test_predeclared_budget_and_noise_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);C.dump(root/'protocol.json',{'evaluation':{
                'seeds':list(range(22000,22100)),'pulse_seeds':list(range(22000,22020))}})
            js=C.jobs(root)
            self.assertEqual(len(js),2400)
            self.assertEqual(len({C.job_id(j) for j in js}),2400)
            self.assertEqual(sum(j['suite']=='native_full' for j in js),500)
            self.assertEqual(sum(j['suite']=='delayed' for j in js),1500)
            self.assertEqual(sum(j['suite']=='pulse' for j in js),400)
        self.assertEqual(R.noise_seed(10,7,'grasp','delayed',0),R.noise_seed(10,7,'grasp','pulse',0))
        self.assertEqual(R.noise_seed(10,7,'approach','native_full',0),R.noise_seed(10,7,'approach','pulse',0))

    def test_phase_lift_stability_and_no_regression(self):
        t=C.PhaseTracker([.2,0,.75]);s=np.ones(14)
        t.update(s,[.2,0,.8],[0,0,.75],1)
        self.assertEqual(t.phase,'approach')
        s[13]=.5
        for step,z in [(2,.8),(3,.75),(4,.8),(5,.8)]:t.update(s,[.2,0,z],[0,0,.75],step)
        self.assertEqual(t.phase,'grasp')
        t.update(s,[.2,0,.8],[0,0,.75],6)
        self.assertEqual(t.phase,'transport')
        t.update(s,[.01,0,.8],[0,0,.75],7)
        self.assertEqual(t.phase,'place')
        t.update(np.ones(14),[.2,0,.75],[0,0,.75],8)
        self.assertEqual(t.phase,'place')

    def test_report_uses_native_and_stage_clean_intersection(self):
        import stage_joint_report as report
        with tempfile.TemporaryDirectory() as tmp,patch.object(report,'ROOT',Path(tmp)):
            root=Path(tmp)/'run';root.mkdir();(Path(tmp)/'result').mkdir()
            C.dump(root/'protocol.json',{'evaluation':{
                'seeds':list(range(22000,22100)),'pulse_seeds':list(range(22000,22020))}})
            ph=C.sha(root/'protocol.json')
            for seed in [22000,22001,22002]:
                for suite,stage in [('native_full','approach'),('delayed','grasp')]:
                    for condition in C.CONDITIONS:
                        job=dict(seed=seed,suite=suite,stage=stage,condition=condition)
                        success=condition!='early'
                        if condition=='clean':success=seed!=22001 if suite=='native_full' else seed!=22002
                        C.dump(R.output_path(root,job),dict(job=job,protocol_sha256=ph,status='complete',
                            success=success,steps=100,initial_clean_sha256=str(seed)))
            error=dict(seed=22000,suite='pulse',stage='approach',condition='early')
            C.dump(R.output_path(root,error),dict(job=error,protocol_sha256=ph,status='runtime_error',error='fake runtime'))
            summary=report.report(root)
            row=next(r for r in summary['rows'] if r['suite']=='delayed' and r['stage']=='grasp' and r['condition']=='early')
            self.assertEqual(row['eligible_seeds'],[22000])
            self.assertEqual(row['n'],1);self.assertEqual(row['failures'],1)
            self.assertEqual(row['excess_ASR'],1)
            self.assertEqual(len(summary['errors']),1)
            self.assertFalse(summary['complete'])
            self.assertGreater(report.wilson([0]*10)[1],.2)


class FakeTask:
    instances=[]
    def __init__(self,seed):
        self.seed=seed;self.take_action_cnt=0;self.step_lim=64;self.eval_success=False
        self.actor_name='002_bowl';self.active_patch=False;self.closed=False
        self.container=SimpleNamespace(get_pose=lambda:SimpleNamespace(p=self.object_position()))
        self.plate=SimpleNamespace(get_pose=lambda:SimpleNamespace(p=np.array([0,0,.75])))
        self.robot=SimpleNamespace(get_left_arm_jointState=lambda:self.state()[:7].tolist(),
                                   get_right_arm_jointState=lambda:self.state()[7:].tolist())
        self.scene=SimpleNamespace(remove_actor=lambda actor:setattr(self,'active_patch',False))
        FakeTask.instances.append(self)
    def object_position(self):
        return np.array([.01 if self.take_action_cnt>=(45 if self.seed==2 else 20) else .2,
                         0,.79 if self.take_action_cnt>=6 else .75])
    def state(self):
        s=np.zeros(14,dtype=np.float32);s[0]=self.take_action_cnt/100
        s[6]=1;s[13]=.2 if self.take_action_cnt>=5 else 1
        return s
    def get_obs(self):
        v=self.take_action_cnt+(80 if self.active_patch else 0)
        return {'joint_action':{'vector':self.state().astype(np.float64)},
            'observation':{n:{'rgb':np.full((3,4,3),v,np.uint8)} for n in C.CAMERAS}}
    def take_action(self,action,action_type):
        assert action_type=='qpos'
        self.take_action_cnt+=1;self.eval_success=self.take_action_cnt>=45
    def set_instruction(self,prompt):pass
    def close_env(self):self.closed=True


class ReplayTests(unittest.TestCase):
    def test_mid_chunk_onset_and_one_query_pulse(self):
        import bootstrap
        scene=ModuleType('scene_smoke');scene.make_task=lambda task,seed,assets:(FakeTask(seed),None,None)
        geometry=ModuleType('render_geometry')
        class Patch:
            def __init__(self,task,*args,**kw):self.task=task;self.actor=SimpleNamespace(per_scene_id=9)
            def texture(self,texture):self.task.active_patch=True
        geometry.Patch=Patch
        geometry.camera_objects=lambda task:{n:SimpleNamespace(get_picture=lambda key:
            np.full((3,4,2),9 if task.active_patch else 0,np.uint32)) for n in C.CAMERAS}
        def infer(payload,root):
            v=float(payload['state'][0])+float(payload['images'][C.WIRE[0]][0,0,0])/100
            return {'actions':np.full((32,14),v,np.float32)}
        with tempfile.TemporaryDirectory() as tmp,patch.dict('sys.modules',{'scene_smoke':scene,'render_geometry':geometry}),\
                patch.object(bootstrap,'simulator_paths',lambda:None),patch.object(R,'request',infer):
            root=Path(tmp)
            C.dump(root/'protocol.json',dict(task='fake',patch=dict(center_m=[0,0,.741],size_m=.1),evaluation=dict(noise_base=10)))
            dest=root/'patches/random';dest.mkdir(parents=True)
            Image.fromarray(np.full((8,8,3),80,np.uint8)).save(dest/'texture.png')
            baseline=R.run_episode(root,R.native_job(1))
            self.assertEqual(baseline['steps'],45)
            self.assertEqual(baseline['events']['grasp'],5)
            for suite in ['delayed','pulse']:
                control=dict(seed=1,suite=suite,stage='grasp',condition='clean')
                attack={**control,'condition':'random'}
                c=R.run_episode(root,control);a=R.run_episode(root,attack)
                self.assertEqual(a['prefix_verified_actions'],5)
                self.assertEqual(a['injection']['step'],5)
                with np.load(R.output_path(root,attack).with_suffix('.npz')) as d,\
                        np.load(R.output_path(root,control).with_suffix('.npz')) as z:
                    self.assertEqual(d['query_steps'].tolist(),[5,37])
                    self.assertTrue(np.array_equal(d['noise_seeds'],z['noise_seeds']))
                    self.assertEqual(d['patch_active'].tolist(),[True,suite=='delayed'])
                    self.assertEqual(len(d['executed_actions']),45)
                    self.assertTrue(np.array_equal(d['step_states'][:6],z['step_states'][:6]))
            R.run_episode(root,R.native_job(2))
            skipped=R.run_episode(root,dict(seed=2,suite='delayed',stage='place',condition='random'))
            self.assertEqual(skipped['status'],'no_opportunity')
            self.assertIsNone(skipped['success'])
            self.assertTrue(all(t.closed for t in FakeTask.instances))


if __name__=='__main__':unittest.main()
