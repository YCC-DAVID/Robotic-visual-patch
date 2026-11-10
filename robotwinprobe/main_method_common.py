"""First ready RoboTwin task for the frozen render-proxy + texture-EOT method."""
import hashlib
import json
from pathlib import Path
import numpy as np
import bootstrap as B

TASK = 'place_container_plate'
DEFAULT = B.OUT/'main_method'
SOURCE = B.OUT/'frequency_bands_13cm_20261030'
CHECKPOINT = B.ROOT/'checkpoints/pi05_robotwin2_pytorch'
CAMERAS = ('head_camera','left_camera','right_camera')
WIRE = ('cam_high','cam_left_wrist','cam_right_wrist')
MODEL = ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')
CONDITIONS = ('clean','random','method')
SEEDS = list(range(26000,26010))
CENTER = [-0.0400000000000002,0.2400000000000001,0.741]
SIZE = .13


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def records():
    manifest=SOURCE/'collection_manifest.json'
    source=json.loads(manifest.read_text())
    rows=[]
    for row in source['records']:
        if row['draw']!=0:continue
        path=B.PROBE/str(row['path']).split('/robotwinprobe/',1)[1]
        assert path.is_file() and sha(path)==row['sha256'],path
        with np.load(path,allow_pickle=False) as z:
            assert float(z['size_m'])==SIZE and np.array_equal(z['center_m'],CENTER)
            assert float(z['yaw_deg'])==0 and str(z['light'])=='nominal'
            assert z['state14'].shape==(14,)
            for c in CAMERAS:
                for suffix in ['clean','black','gray64','gray','gray192','white','random','mask','intrinsic','extrinsic']:
                    assert c+'_'+suffix in z.files,(path,c,suffix)
                assert z[c+'_clean'].shape==(240,320,3)
        rows.append(dict(path=str(path),sha256=row['sha256'],seed=int(row['seed']),query=int(row['query']),draw=0))
    assert len(rows)==49 and len({r['seed'] for r in rows})==24
    assert not ({r['seed'] for r in rows}&set(SEEDS))
    return rows


def training_spec(rows):
    from texture_eot import RANGES
    return dict(schema='robotwin-main-texture-eot-train-v1',task=TASK,
        checkpoint=str(CHECKPOINT),checkpoint_sha256=sha(CHECKPOINT/'model.safetensors'),
        source_manifest=str(SOURCE/'collection_manifest.json'),source_manifest_sha256=sha(SOURCE/'collection_manifest.json'),
        records=rows,data='49 nominal calibrated policy approach states from 24 clean-policy training seeds; approach only; no expert-success filtering',
        train_seeds=sorted({r['seed'] for r in rows}),independent_validation=False,
        patch_m=SIZE,center_m=CENTER,resolution=128,steps=500,batch=8,lr=.01,
        initialization='gray .5',seed_batch=20261005,noise_seed=20260901,eot_seed=2026102901,
        objective='full 32x32 first-t=1 velocity SSE against same-noise clean',
        eot=RANGES,eot_scope='one texture-plane draw per sampled state, shared before all three camera proxies; fixed world footprint',
        proxy='existing native SAPIEN five-gray response, linear-RGB 4x4 footprint quadrature and global response',
        proxy_fidelity='random-interior RMSE is measured and reported, not asserted as <0.02; this SAPIEN proxy has a known larger fidelity gap',
        checkpoint_selection='fixed final step500; no test-driven selection',
        source_sha256={n:sha(B.PROBE/n) for n in ['main_method_train.py','main_method_common.py','crossarch_proxy.py','crossarch_backend.py','render_model.py','texture_eot.py']})


def prepare(root,tasks=None):
    root=Path(root).resolve();tasks=[TASK] if tasks is None else list(tasks)
    if tasks!=[TASK]:raise ValueError('Only place_container_plate currently has this method calibration; other RoboTwin task preparation is pending')
    rows=records();spec=training_spec(rows)
    protocol=dict(schema='robotwin-main-method-v1',benchmark='RoboTwin',tasks=tasks,first_ready_batch=True,
        ready_tasks=[TASK],remaining_task_scope='other49 require their own pools, position selection and calibration; no coverage claim',
        method=spec,evaluation=dict(seeds=SEEDS,conditions=list(CONDITIONS),task_config='demo_clean',
            patch_m=SIZE,center_m=CENTER,action_horizon=32,replan=32,denoise_steps=10,
            noise_seed_rule='20261006+seed*1000+query',expert_prefilter=False,
            prompt='actual bowl/cup on plate, same historical task prompt; all conditions share prompt',
            ASR='clean-success and method-failure / matched clean successes',
            endpoint='native task eval_success or official step limit; runtime errors never count as attack successes',
            source_sha256={n:sha(B.PROBE/n) for n in ['main_method_rollout.py','main_method_runtime.py','render_geometry.py','scene_smoke.py','policy_server.py']}))
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'protocol changed; preserve prior run'
    else:dump(path,protocol)
    return protocol


def paired_summary(rows,seeds=SEEDS):
    complete={}
    errors=[]
    for row in rows:
        key=(int(row['seed']),row['condition'])
        assert key[0] in seeds and key[1] in CONDITIONS, ('unknown evaluation row',key)
        assert key not in complete and not any((int(r['seed']),r['condition'])==key for r in errors), ('duplicate evaluation row',key)
        if row.get('status')!='complete':errors.append(row);continue
        assert type(row['success']) is bool, ('invalid success type',key)
        complete[key]=row
    paired=[s for s in seeds if all((s,c) in complete for c in CONDITIONS)]
    eligible=[s for s in paired if complete[(s,'clean')]['success']]
    result=dict(expected_seeds=len(seeds),completed_seeds=len(paired),complete=len(paired)==len(seeds) and not errors,
        clean_successes=sum(complete[(s,'clean')]['success'] for s in paired),
        clean_evaluated=len(paired),paired_clean_success_n=len(eligible),runtime_errors=errors,conditions={})
    for c in ['random','method']:
        failures=sum(not complete[(s,c)]['success'] for s in eligible)
        result['conditions'][c]=dict(failures=failures,n=len(eligible),ASR=failures/len(eligible) if eligible and result['complete'] else None,
            robot_successes=sum(complete[(s,c)]['success'] for s in paired),robot_evaluated=len(paired))
    return result
