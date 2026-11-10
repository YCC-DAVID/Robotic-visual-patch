"""Train/validation-only flow diagnostic; never trains or selects on test rollouts.

Run only after GPU allocation. This entry point independently repeats the
project fleet/selected-card preflight before importing the GPU model.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TASKS = ('put_the_bowl_on_the_plate', 'put_the_bowl_on_top_of_the_cabinet',
         'put_the_wine_bottle_on_the_rack', 'turn_on_the_stove',
         'put_the_bowl_on_the_stove', 'put_the_cream_cheese_in_the_bowl',
         'open_the_middle_drawer_of_the_cabinet')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def breakdown(delta):
    """First denoising velocity, all 10 horizon positions and 32 dimensions."""
    x = np.asarray(delta, dtype=np.float64).reshape(10, 32)
    q = x * x
    total = float(q.sum())
    active = float(q[:, :7].sum())
    executed = float(q[:5, :7].sum())
    return dict(full_SSE=total, active7_SSE=active,
                padded25_SSE=float(q[:, 7:].sum()),
                executed5x7_SSE=executed,
                later5x7_SSE=float(q[5:, :7].sum()),
                first5_full32_SSE=float(q[:5].sum()),
                active7_fraction=active / total if total else None,
                executed5x7_fraction=executed / total if total else None)


def check_resource(args, out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == args.gpu_uuid
    skill = ROOT / 'skills/gpu-preflight/SKILL.md'
    assert skill.is_file()
    _ = skill.read_text()
    script = skill.parent / 'scripts/check_gpus.py'
    host = socket.gethostname().split('.')[0]
    for name, extra in [('fleet', []), ('selected', ['--hosts', host])]:
        cmd = [sys.executable, str(script), '--json', '--min-free-mib', '32000', *extra]
        r = subprocess.run(cmd, text=True, capture_output=True, timeout=90)
        assert r.returncode in (0, 2), (r.returncode, r.stderr)
        data = json.loads(r.stdout)
        dump(out / ('preflight_' + name + '.json'), dict(command=cmd, **data))
        node = next(n for n in data['nodes'] if n['host'] == host)
        assert node['status'] == 'ok'
        gpu = next(g for g in node['gpus'] if g['uuid'] == args.gpu_uuid and g['index'] == args.gpu_index)
        assert gpu['idle'] and gpu['eligible'] and not gpu['faulty'], gpu
        assert not (host == 'nnmc63' and args.gpu_index == 0), 'historical ECC exclusion'


def run(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'diagnostic.json').exists(), 'use a new output directory'
    check_resource(args, out)
    import dataclasses
    import torch
    import jax
    from PIL import Image
    import drift_patch_train as T
    from dual_view_proxy import CameraProxy
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.models import model as M
    from openpi_client import image_tools
    torch.set_num_threads(4)
    cfg = C.get_config('pi05_libero')
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = PC.create_trained_policy(cfg, T.TORCH_CKPT)
    model, device = policy._model, policy._pytorch_device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    seeds = [20260901] + list(range(2026110901, 2026110901 + args.noise_draws - 1))
    report = dict(schema='libero-low-asr-flow-diagnostic-v1', complete=False,
                  started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  gpu_uuid=args.gpu_uuid, gpu_index=args.gpu_index,
                  tasks=args.tasks.split(','), noise_seeds=seeds,
                  noise_rule='training fixed seed plus independent diagnostic draws; no test noise',
                  frame_rule='uniform deterministic indices within train ep0-9 / validation ep10-11 only',
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                      [Path(__file__), ROOT/'pi05probe/multitask_train.py',
                       ROOT/'pi05probe/dual_view_proxy.py', ROOT/'pi05probe/drift_patch_train.py']},
                  rows=[], gradient_rows=[], proxy_native_rows=[], input_provenance={})
    result_path = out / 'diagnostic.json'
    for task in report['tasks']:
        assert task in TASKS
        folder = ROOT / 'pi05probe/out/visible_pos/tasks' / task
        trained = ROOT / 'pi05probe/out/visible_pos/train' / task / 'eot'
        with np.load(folder/'base.npz', allow_pickle=False) as z:
            data = {k: z[k] for k in ['clean_base', 'clean_wrist', 'state8', 'prompt',
                                     'ep_id', 'train_mask', 'validation_mask', 'random__texture']}
        proxies = {v: CameraProxy(folder/(v+'.npz'), device,
                                 ROOT/'pi05probe/out/dual_view/fit'/(v+'.json'))
                   for v in ['base', 'wrist']}
        arrays = {'gray': np.full((128, 128, 3), .5, np.float32),
                  'random': np.asarray(Image.open(ROOT/'pi05probe/out/dual_view/manifest_both/random.png').convert('RGB'), dtype=np.float32)/255,
                  'trained': np.asarray(Image.open(trained/'texture_0500.png').convert('RGB'), dtype=np.float32)/255,
                  'calibration_random': data['random__texture'].astype(np.float32)/255}
        qs = {name: torch.from_numpy(a.transpose(2, 0, 1)[None].copy()).to(device)
              for name, a in arrays.items()}
        report['input_provenance'][task] = dict(
            calibration={v: sha(folder/(v+'.npz')) for v in ['base', 'wrist']},
            trained_texture=sha(trained/'texture_0500.png'),
            trained_config=sha(trained/'config.json'), prompt=str(data['prompt']))

        def obs(k, images=None):
            x = policy._input_transform(dict(
                **{'observation/image': data['clean_base'][k],
                   'observation/wrist_image': data['clean_wrist'][k],
                   'observation/state': data['state8'][k]}, prompt=str(data['prompt'])))
            x = jax.tree.map(lambda a: torch.from_numpy(np.array(a)).to(device)[None], x)
            value = M.Observation.from_dict(x)
            if images is not None:
                value.images['base_0_rgb'] = images['base'] * 2 - 1
                value.images['left_wrist_0_rgb'] = images['wrist'] * 2 - 1
            return value

        selected = []
        for split, key in [('train', 'train_mask'), ('validation', 'validation_mask')]:
            indices = np.flatnonzero(data[key])
            count = min(args.frames_per_split, len(indices))
            offsets = [len(indices)//2] if count == 1 else np.linspace(0, len(indices)-1, count, dtype=int)
            for k in np.unique(indices[offsets]):
                assert (int(data['ep_id'][k]) < 10) == (split == 'train')
                selected.append((split, int(k)))
        for split, k in selected:
            with torch.no_grad():
                rendered = {name: {v: proxies[v](q, k) for v in proxies}
                            for name, q in qs.items()}
                oracle = {}
                pixel_rows = {}
                for v in proxies:
                    with np.load(folder/(v+'.npz'), allow_pickle=False) as z:
                        native = z['random256'][k]
                    raw224 = image_tools.convert_to_uint8(image_tools.resize_with_pad(native, 224, 224))
                    oracle[v] = torch.from_numpy(raw224.transpose(2, 0, 1)[None].copy()).to(device).float()/255
                    d = (rendered['calibration_random'][v] - oracle[v]).float()
                    pixel_rows[v] = dict(full224_RMSE=float(d.square().mean().sqrt()),
                                         max_abs224=float(d.abs().max()))
                native_obs = obs(k, oracle)
                clean_obs = obs(k)
                for seed in seeds:
                    noise = torch.from_numpy(np.random.RandomState(seed).standard_normal((1, 10, 32)).astype(np.float32)).to(device)
                    clean = T.velocity_at_steps(model, device, clean_obs, noise.clone()).float()
                    velocities = {}
                    for name, images in rendered.items():
                        value = T.velocity_at_steps(model, device, obs(k, images), noise.clone()).float()
                        delta = value - clean
                        velocities[name] = delta
                        report['rows'].append(dict(task=task, split=split, frame=k, episode=int(data['ep_id'][k]),
                            condition=name, noise_seed=seed,
                            training_noise=seed == 20260901, **breakdown(delta.cpu().numpy())))
                    native_delta = T.velocity_at_steps(model, device, native_obs, noise.clone()).float() - clean
                    a = velocities['calibration_random'].flatten();b = native_delta.flatten()
                    proxy_sse = float(a.square().sum());native_sse = float(b.square().sum())
                    report['proxy_native_rows'].append(dict(task=task, split=split, frame=k, noise_seed=seed,
                        pixels=pixel_rows, delta_v_cosine=float(torch.nn.functional.cosine_similarity(a[None], b[None])),
                        proxy=breakdown(a.cpu().numpy()), native=breakdown(b.cpu().numpy()),
                        relative_full_SSE_error=abs(proxy_sse-native_sse)/max(native_sse, 1e-12)))
            if args.gradients and split == 'train' and k == selected[0][1]:
                q = qs['trained'].detach().clone().requires_grad_(True)
                noise = torch.from_numpy(np.random.RandomState(20260901).standard_normal((1, 10, 32)).astype(np.float32)).to(device)
                with torch.no_grad():
                    clean = T.velocity_at_steps(model, device, obs(k), noise.clone()).float()
                delta = (T.velocity_at_steps(model, device, obs(k, {v: proxies[v](q, k) for v in proxies}), noise.clone()).float()-clean).reshape(10, 32)
                full = delta.square().sum();executed = delta[:5, :7].square().sum()
                gf, = torch.autograd.grad(full, q, retain_graph=True)
                ge, = torch.autograd.grad(executed, q)
                report['gradient_rows'].append(dict(task=task, frame=k, noise_seed=20260901,
                    full_gradient_norm=float(gf.norm()), executed5x7_gradient_norm=float(ge.norm()),
                    cosine=float(torch.nn.functional.cosine_similarity(gf.flatten()[None], ge.flatten()[None])),
                    sign_agreement=float((gf.sign()==ge.sign()).float().mean()),
                    finite=bool(torch.isfinite(gf).all() and torch.isfinite(ge).all()),
                    policy_parameter_gradients=sum(p.grad is not None for p in model.parameters())))
            dump(result_path, report)
            print('[diagnostic]', task, split, k, 'rows', len(report['rows']), flush=True)
        del proxies, qs, data
        torch.cuda.empty_cache()
    report['complete'] = True
    report['completed'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    dump(result_path, report)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tasks', default=','.join(TASKS))
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu-index', type=int, required=True)
    ap.add_argument('--gpu-uuid', required=True)
    ap.add_argument('--frames-per-split', type=int, default=1)
    ap.add_argument('--noise-draws', type=int, default=4)
    ap.add_argument('--gradients', action='store_true')
    args = ap.parse_args()
    assert args.frames_per_split > 0 and args.noise_draws >= 2
    run(args)
