"""Matched clean/random/method true-SAPIEN episodes for a fixed trained patch."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
import numpy as np
import bootstrap as B
from main_method_common import TASK,CONDITIONS,SEEDS,CENTER,SIZE,CAMERAS,WIRE,dump,sha,paired_summary


def digest(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pci(value):
    domain,bus,tail=str(value).lower().split(':')
    return '%04x:%02x:%s'%(int(domain,16),int(bus,16),tail)


def pin_sapien_device(expected_uuid,out):
    """Bind both legacy renderer and scene RenderSystem to authorized CUDA0."""
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=expected_uuid:
        raise RuntimeError('RoboTwin render requires sole visible GPU UUID')
    expected=subprocess.check_output(['nvidia-smi','--id='+expected_uuid,
        '--query-gpu=pci.bus_id','--format=csv,noheader'],text=True).strip()
    import sapien
    import sapien.core as core
    device=sapien.Device('cuda:0')
    assert device.cuda_id==0 and device.can_render() and device.is_cuda()
    assert device.pci_string and pci(device.pci_string)==pci(expected),('SAPIEN selected wrong PCI device',device.pci_string,expected)
    native_renderer=sapien.pysapien.render.SapienRenderer
    def renderer(*args,**kw):
        if args or kw:raise ValueError('unexpected legacy renderer options')
        return native_renderer(device)
    def scene(engine,config=None):
        if config is not None:sapien.physx.set_scene_config(config)
        selected=sapien.render.RenderSystem(device)
        assert selected.device.cuda_id==0 and pci(selected.device.pci_string)==pci(expected)
        return sapien.Scene([sapien.physx.PhysxCpuSystem(),selected])
    sapien.SapienRenderer=renderer;core.SapienRenderer=renderer
    sapien.Engine.create_scene=scene
    report=dict(authorized_uuid=expected_uuid,cuda_visible_devices=expected_uuid,
        sapien_device=str(device),cuda_logical_index=device.cuda_id,pci_string=device.pci_string,
        nvidia_smi_pci=expected,binding='explicit Device(cuda:0) for renderer and RenderSystem; PCI checked before render context',
        render_can_present=device.can_present(),render_can_render=device.can_render())
    dump(Path(out)/'sapien_device.json',report)
    return report


def actors(task):
    return sorted([dict(name=a.get_name(),p=a.get_pose().p.tolist(),q=a.get_pose().q.tolist())
                   for a in task.scene.get_all_actors()],key=lambda a:(a['name'],a['p'],a['q']))


def image_delta(a,b):
    d=np.abs(a.astype(np.int16)-b.astype(np.int16))
    return dict(max_abs=int(d.max()),mean_abs=float(d.mean()),changed_pixels=int(np.any(d,axis=-1).sum()))


def main(args):
    B.simulator_paths()
    from main_method_runtime import install_path_normalization
    install_path_normalization(Path(args.root)/'eval'/TASK)
    from PIL import Image
    root=Path(args.root).resolve();out=root/'eval'/TASK;out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    assert protocol['tasks']==[TASK] and protocol['evaluation']['conditions']==list(CONDITIONS)
    assert protocol['evaluation']['seeds']==SEEDS
    for name,row in manifest['textures'].items():
        if row is not None:assert sha(row['path'])==row['sha256'],name
    binding=pin_sapien_device(args.gpu_uuid,out)
    from scene_smoke import make_task
    from render_geometry import Patch,camera_objects
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    client=WebsocketClientPolicy('127.0.0.1',args.port)
    artifacts=out/'episodes';artifacts.mkdir(exist_ok=True)
    def save():
        rows=[json.loads(p.read_text()) for p in sorted(artifacts.glob('*.json'))]
        summary=paired_summary(rows);dump(root/'summary.json',dict(task=TASK,**summary))
        return summary
    try:
        for seed in SEEDS:
            for condition in CONDITIONS:
                path=artifacts/f'seed{seed}_{condition}.json';artifact=path.with_suffix('.npz')
                if path.exists():
                    row=json.loads(path.read_text())
                    assert row['protocol_sha256']==sha(root/'protocol.json') and row['manifest_sha256']==sha(root/'manifest.json')
                    if row['status']!='complete':raise RuntimeError('preserved runtime error requires review: '+str(path))
                    assert sha(artifact)==row['artifact_sha256'];continue
                task=None;started=time.monotonic()
                try:
                    assets=out/'assets'/f'seed{seed}_{condition}';assets.mkdir(parents=True,exist_ok=True)
                    random.seed(seed);task,config,embodiment=make_task(TASK,seed,assets)
                    actual=task.scene.render_system.device
                    assert pci(actual.pci_string)==pci(binding['nvidia_smi_pci'])
                    prompt='Place the '+('bowl' if task.actor_name=='002_bowl' else 'cup')+' on the plate.'
                    task.set_instruction(prompt)
                    initial=task.get_obs()
                    initial_state=initial['joint_action']['vector'].astype(np.float32).copy()
                    initial_rgb={c:initial['observation'][c]['rgb'].copy() for c in CAMERAS}
                    repeat=task.get_obs()
                    row=dict(status='running',seed=seed,condition=condition,prompt=prompt,
                        protocol_sha256=sha(root/'protocol.json'),manifest_sha256=sha(root/'manifest.json'),
                        initial_state_sha256=digest(initial_state),initial_actors=actors(task),embodiment=embodiment,
                        initial_rgb_sha256={c:digest(initial_rgb[c]) for c in CAMERAS},
                        initial_clean_repeat={c:image_delta(initial_rgb[c],repeat['observation'][c]['rgb']) for c in CAMERAS})
                    if condition!='clean':
                        clean_path=artifacts/f'seed{seed}_clean.json';reference=json.loads(clean_path.read_text())
                        assert reference['status']=='complete'
                        for key in ['prompt','initial_state_sha256','initial_actors','embodiment']:
                            assert row[key]==reference[key],('unpaired initial physical state/prompt',seed,condition,key)
                        with np.load(clean_path.with_suffix('.npz'),allow_pickle=False) as z:
                            row['paired_clean_rgb_difference']={c:image_delta(initial_rgb[c],z['initial_'+c]) for c in CAMERAS}
                        # Ray tracing / OIDN can be stochastic. RGB drift is diagnosed
                        # against same-scene repeats; physical pairing remains exact.
                        texture=np.asarray(Image.open(manifest['textures'][condition]['path']).convert('RGB'))
                        patch=Patch(task,assets,center=CENTER,size=SIZE);patch.texture(texture)
                        after=task.get_obs()
                        assert np.array_equal(after['joint_action']['vector'].astype(np.float32),initial_state)
                        assert [a for a in actors(task) if a['name']!='wam_visual_patch']==row['initial_actors'],'visual patch changed physics'
                    states=[];actions=[];noise_seeds=[];visibility=[];hashes=[]
                    while task.take_action_cnt<task.step_lim and not task.eval_success:
                        obs=task.get_obs();state=obs['joint_action']['vector'].astype(np.float32)
                        raw={w:obs['observation'][c]['rgb'] for c,w in zip(CAMERAS,WIRE)}
                        noise_seed=20261006+seed*1000+len(states)
                        noise=np.random.RandomState(noise_seed).standard_normal((32,32)).astype(np.float32)
                        chunk=np.asarray(client.infer(dict(state=state,prompt=prompt,images=raw,noise=noise))['actions'])
                        assert chunk.shape==(32,14) and np.isfinite(chunk).all()
                        states.append(state);actions.append(chunk);noise_seeds.append(noise_seed)
                        hashes.append([digest(raw[w]) for w in WIRE])
                        if condition!='clean':
                            cameras=camera_objects(task)
                            visibility.append([int((cameras[c].get_picture('Segmentation')[...,1]==patch.actor.per_scene_id).sum()) for c in CAMERAS])
                        else:visibility.append([0,0,0])
                        for action in chunk:
                            task.take_action(action,action_type='qpos')
                            if task.eval_success or task.take_action_cnt>=task.step_lim:break
                    assert states,'initial state unexpectedly complete'
                    if condition!='clean':assert any(any(v) for v in visibility),'patch never visible to policy'
                    temporary=artifact.with_suffix('.tmp.npz')
                    np.savez_compressed(temporary,states14=states,actions=actions,noise_seeds=noise_seeds,
                        visible_pixels=visibility,input_rgb_sha256=hashes,initial_state14=initial_state,
                        **{'initial_'+c:im for c,im in initial_rgb.items()})
                    temporary.replace(artifact)
                    row.update(status='complete',success=bool(task.eval_success),final_predicate=bool(task.check_success()),
                        steps=task.take_action_cnt,queries=len(states),step_limit=task.step_lim,
                        artifact_sha256=sha(artifact),elapsed_s=time.monotonic()-started,
                        texture_sha256=None if condition=='clean' else manifest['textures'][condition]['sha256'],
                        visible_pixels_max=np.max(visibility,axis=0).astype(int).tolist())
                    dump(path,row);save();print('[episode]',seed,condition,row['success'],row['steps'],flush=True)
                except BaseException as error:
                    dump(path,dict(status='runtime_error',seed=seed,condition=condition,error=repr(error),
                        traceback=traceback.format_exc(),protocol_sha256=sha(root/'protocol.json'),manifest_sha256=sha(root/'manifest.json')))
                    save();raise
                finally:
                    if task is not None:task.close_env()
        summary=save();assert summary['complete']
        dump(out/'complete.json',dict(complete=True,protocol_sha256=sha(root/'protocol.json'),manifest_sha256=sha(root/'manifest.json')))
    finally:client._ws.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--port',required=True,type=int);ap.add_argument('--gpu-uuid',required=True)
    main(ap.parse_args())
