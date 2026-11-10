"""E-MT step 2: one shared Q128 trained jointly on several LIBERO-goal tasks.

Identical to E26 dual_view_train --camera-mode both (gray init, 500x8, sign
step .01, fixed noise 20260901, first-step velocity SSE vs same-noise clean,
both camera proxies, frozen pi05_libero). Only changes: frames come from
several tasks, each frame uses its own task prompt, and every batch draws
batch/ntasks frames from each task (stratified) so no task dominates.
"""
import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path
import drift_patch_train as T
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks-root', required=True, type=Path)
    ap.add_argument('--tasks', required=True, help='comma list of task folder names under tasks-root')
    ap.add_argument('--fit-root', default=str(T.OUT/'dual_view/fit'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=.01)
    ap.add_argument('--seed', type=int, default=20261005)
    ap.add_argument('--save-every', type=int, default=25)
    ap.add_argument('--eot', action='store_true', help='texture-plane EOT of #51/#52 (one draw per sampled frame)')
    ap.add_argument('--eot-seed', type=int, default=2026102901)
    ap.add_argument('--proxy-rmse-max', type=float, default=None,
                    help='abort before texture updates if sampled proxy/render interior RGB RMSE exceeds this [0,1] limit')
    args = ap.parse_args()
    import torch
    import jax
    from PIL import Image
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.models import model as M
    from dual_view_proxy import CameraProxy
    import sys; sys.path.insert(0, str(T.PROBE.parent/'robotwinprobe'))
    import texture_eot as E
    eot_plan = E.plan(args.iters, args.batch, args.eot_seed) if args.eot else None
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    names = args.tasks.split(',')
    assert args.batch % len(names) == 0, 'stratified batch needs batch divisible by #tasks'
    per_task = args.batch//len(names)
    dest = Path(args.out); dest.mkdir(parents=True, exist_ok=True)
    assert not (dest/'config.json').exists(), 'refuse to overwrite a training run'
    fit = Path(args.fit_root)
    cfg = C.get_config('pi05_libero')
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = PC.create_trained_policy(cfg, T.TORCH_CKPT)
    model, device = policy._model, policy._pytorch_device
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    tasks = []
    for n in names:
        base_path, wrist_path = args.tasks_root/n/'base.npz', args.tasks_root/n/'wrist.npz'
        with np.load(base_path, allow_pickle=False) as z:
            d = {k: z[k] for k in ['train_mask', 'validation_mask', 'ep_id', 'clean_base', 'clean_wrist', 'state8', 'prompt', 'patch_m', 'cell_world']}
        assert np.array_equal(d['cell_world'], np.load(args.tasks_root/n/'wrist.npz')['cell_world'])
        assert float(d['patch_m']) == .13
        tasks.append(dict(name=n, d=d, base=CameraProxy(base_path, device, fit/'base.json'),
                          wrist=CameraProxy(wrist_path, device, fit/'wrist.json'),
                          train=np.flatnonzero(d['train_mask']), validation=np.flatnonzero(d['validation_mask']),
                          sha={'base': sha(base_path), 'wrist': sha(wrist_path)}))
    noise_np = np.random.RandomState(T.SEED_NOISE).standard_normal((T.AH, T.AD)).astype(np.float32)
    noise = torch.from_numpy(noise_np)[None].to(device)
    rng_batch = np.random.RandomState(args.seed)
    metadata = {'schema': 'wam-multitask-texture-train-v1', 'complete': False, 'mode': 'both',
        'tasks': names, 'prompts': [str(t['d']['prompt']) for t in tasks],
        'iters': args.iters, 'batch': args.batch, 'per_task_per_batch': per_task, 'lr': args.lr,
        'seed_batch': args.seed, 'noise_seed': T.SEED_NOISE, 'init': 'constant gray .5',
        'calibration_sha256': {t['name']: t['sha'] for t in tasks},
        'fit_sha256': {v: sha(fit/(v+'.json')) for v in ['base', 'wrist']},
        'train_frames': {t['name']: len(t['train']) for t in tasks},
        'validation_frames': {t['name']: len(t['validation']) for t in tasks},
        'source_sha256': {Path(p).name: sha(p) for p in [__file__, T.PROBE/'dual_view_proxy.py']},
        'checkpoint': str(T.TORCH_CKPT), 'patch_m': .13, 'cell_world': tasks[0]['d']['cell_world'].tolist(),
        'cell_world_by_task': {t['name']: t['d']['cell_world'].tolist() for t in tasks},
        'eot': None if not args.eot else {'ranges': E.RANGES, 'seed': args.eot_seed, 'source': 'robotwinprobe/texture_eot.py',
                                          'scope': 'one draw per sampled frame, shared by base+wrist proxies; diag/validation nominal'},
        'source_eot_sha256': sha(T.PROBE.parent/'robotwinprobe/texture_eot.py'),
        'notes': 'E26 both-camera recipe unchanged except multi-task stratified sampling and per-frame prompt.'
                 + (' Texture-plane EOT on.' if args.eot else ' No EOT.')}
    (dest/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')

    def obs(t, k):
        d = tasks[t]['d']
        x = policy._input_transform({'observation/image': d['clean_base'][k], 'observation/wrist_image': d['clean_wrist'][k],
                                     'observation/state': d['state8'][k], 'prompt': str(d['prompt'])})
        x = jax.tree.map(lambda a: torch.from_numpy(np.array(a)).to(device)[None], x)
        return M.Observation.from_dict(x)

    clean = {}
    with torch.no_grad():
        a = T.velocity_at_steps(model, device, obs(0, int(tasks[0]['train'][0])), noise.clone())
        b = T.velocity_at_steps(model, device, obs(0, int(tasks[0]['train'][0])), noise.clone())
        assert torch.equal(a, b), 'same-input forward must be deterministic'
        for t, task in enumerate(tasks):
            for k in np.concatenate([task['train'], task['validation']]):
                clean[(t, int(k))] = T.velocity_at_steps(model, device, obs(t, int(k)), noise.clone()).detach()
            print('[cache]', task['name'], len(task['train'])+len(task['validation']), flush=True)
    # proxy check at this position: random texture through proxy vs real render (native, inside visible mask)
    proxy_check = {}
    with torch.no_grad():
        for task in tasks:
            for view in ['base', 'wrist']:
                with np.load(args.tasks_root/task['name']/(view+'.npz'), allow_pickle=False) as z:
                    rnd, real, m = z['random__texture'], z['random256'], z['mask']
                q = torch.from_numpy(rnd.transpose(2, 0, 1)[None].astype(np.float32)/255).to(device)
                errs = []
                samples = np.unique(np.r_[np.linspace(0, len(real)-1, 12, dtype=int), int(np.argmax(m.sum(axis=(1, 2))))])
                for k in samples:
                    if m[k].sum() < 50: continue
                    img = task[view](q, int(k), native=True)[0].permute(1, 2, 0).cpu().numpy()
                    import cv2
                    vis = cv2.erode(cv2.resize(m[k].astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST),
                                    np.ones((5, 5), np.uint8)).astype(bool)
                    if vis.sum() < 20: continue
                    errs.append(float(np.sqrt(((img[vis]-real[k][vis]/255.)**2).mean())))
                proxy_check[task['name']+'/'+view] = {'frames': len(errs), 'mask_RMSE_max': max(errs) if errs else None}
    print('[proxy check]', proxy_check, flush=True)
    metadata['proxy_check'] = proxy_check
    metadata['proxy_rmse_max'] = args.proxy_rmse_max
    (dest/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if args.proxy_rmse_max is not None:
        assert args.proxy_rmse_max > 0
        for name, check in proxy_check.items():
            error = check['mask_RMSE_max']
            assert error is None or (np.isfinite(error) and error <= args.proxy_rmse_max), ('proxy/render mismatch', name, check)
            if name.endswith('/base'):
                assert check['frames'] > 0, ('base proxy check has no visible samples', name)
    Q = torch.full((1, 3, 128, 128), .5, device=device, requires_grad=True)

    def loss(t, k, draw=None):
        o = obs(t, k)
        q = Q if draw is None else E.transform(Q, draw)
        o.images[T.BASE_KEY] = tasks[t]['base'](q, k)*2-1
        assert 'left_wrist_0_rgb' in o.images
        o.images['left_wrist_0_rgb'] = tasks[t]['wrist'](q, k)*2-1
        return (T.velocity_at_steps(model, device, o, noise.clone())-clean[(t, k)]).square().sum()

    def evaluate(pairs):
        with torch.no_grad():
            return [float(loss(t, int(k))) for t, k in pairs]

    diag = [(t, int(k)) for t, task in enumerate(tasks) for k in task['train'][np.linspace(0, len(task['train'])-1, 4, dtype=int)]]
    validation = [(t, int(k)) for t, task in enumerate(tasks) for k in task['validation']]
    initial = evaluate(diag)
    print('[init]', np.mean(initial), flush=True)
    history, task_history, batches = [], [], []
    started = time.monotonic()

    def save(done):
        q = Q.detach().cpu().numpy()[0]
        Image.fromarray(np.clip(np.rint(q.transpose(1, 2, 0)*255), 0, 255).astype(np.uint8)).save(dest/f'texture_{done:04d}.png')
        np.savez_compressed(dest/f'checkpoint_{done:04d}.npz', Q=q, history=np.array(history), task_history=np.array(task_history),
                            batches=np.array(batches), noise=noise_np)
        (dest/'progress.json').write_text(json.dumps({'complete': done == args.iters, 'completed_iters': done,
            'elapsed_training_s': time.monotonic()-started, 'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30,
            'last_batch_L0': history[-1], 'last_batch_L0_by_task': dict(zip(names, task_history[-1]))}, indent=2)+'\n')

    for it in range(args.iters):
        batch = [(t, int(k)) for t, task in enumerate(tasks) for k in rng_batch.choice(task['train'], per_task, replace=False)]
        Q.grad = None
        values = []
        for slot, (t, k) in enumerate(batch):
            objective = loss(t, k, None if eot_plan is None else eot_plan[it][slot])
            (-objective).backward(); values.append(float(objective.detach()))
        assert Q.grad is not None and torch.isfinite(Q.grad).all() and float(Q.grad.norm()) > 0
        with torch.no_grad():
            Q.sub_(args.lr*Q.grad.sign()).clamp_(0, 1)
        history.append(float(np.mean(values)))
        task_history.append([float(np.mean(values[t*per_task:(t+1)*per_task])) for t in range(len(tasks))])
        batches.append([[t, k] for t, k in batch])
        if (it+1) % args.save_every == 0 or it+1 == args.iters: save(it+1)
        if it == 0 or (it+1) % 5 == 0:
            print('[train]', it+1, 'L0', history[-1], dict(zip(names, np.round(task_history[-1], 3))),
                  'min', round((time.monotonic()-started)/60, 2), flush=True)
    final_diag, final_val = evaluate(diag), evaluate(validation)
    by_task = lambda pairs, vals: {names[t]: float(np.mean([v for (s, _), v in zip(pairs, vals) if s == t])) for t in range(len(tasks))}
    result = {'complete': True, 'iterations': len(history), 'gradient_frame_forwards': len(history)*args.batch,
              'initial_train_diag_L0': float(np.mean(initial)), 'initial_train_diag_L0_by_task': by_task(diag, initial),
              'final_train_diag_L0': float(np.mean(final_diag)), 'final_train_diag_L0_by_task': by_task(diag, final_diag),
              'final_validation_proxy_L0': float(np.mean(final_val)), 'final_validation_L0_by_task': by_task(validation, final_val),
              'elapsed_training_s': time.monotonic()-started, 'texture_sha256': sha(dest/f'texture_{args.iters:04d}.png'),
              'notes': 'Proxy L0 only; real-render closed-loop evaluation separate.'}
    (dest/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    metadata['complete'] = True
    (dest/'config.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('[done]', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
