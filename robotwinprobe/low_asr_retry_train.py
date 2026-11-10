"""Fixed500x8 first-flow SSE retry using all native trajectory stages."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import bootstrap as B
from low_asr_retry_common import DEFAULT, SOURCE, TASK, CAMERAS, STAGES, TRAIN_SEEDS, VALIDATION_SEEDS, collected_records
from main_method_common import dump, sha


def specification(root):
    protocol, rows, episodes = collected_records(root)
    config = dict(schema='robotwin-full-trajectory-retry-train-v1',
                  protocol_sha256=sha(root/'protocol.json'), method=protocol['method'],
                  records=[r for r in rows if r['seed'] in TRAIN_SEEDS],
                  validation_records=[r for r in rows if r['seed'] in VALIDATION_SEEDS],
                  checkpoint=str(B.ROOT/'checkpoints/pi05_robotwin2_pytorch'),
                  checkpoint_sha256=sha(B.ROOT/'checkpoints/pi05_robotwin2_pytorch/model.safetensors'),
                  unfiltered_collection_episodes=[dict(seed=e['seed'], split='train' if e['seed'] in TRAIN_SEEDS else 'validation', clean_success=e['clean_success'],
                      steps=e['steps'], queries=len(e['records'])) for e in episodes])
    expected = json.loads((SOURCE/'protocol.json').read_text())['method']['checkpoint_sha256']
    assert config['checkpoint_sha256'] == expected, 'frozen policy checkpoint changed'
    return config, rows


def training_plan(rows):
    by_stage = {s: {} for s in STAGES}
    for i, row in enumerate(rows):
        if row['seed'] in TRAIN_SEEDS:
            by_stage[row['stage']].setdefault(row['seed'], []).append(i)
    assert all(by_stage.values()), 'missing training stage'
    rng = np.random.RandomState(20261005); plan = []
    for _ in range(500):
        ids = []
        for stage in STAGES:
            seeds = sorted(by_stage[stage])
            for _ in range(2):
                seed = int(rng.choice(seeds))
                ids.append(int(rng.choice(by_stage[stage][seed])))
        plan.append(ids)
    return plan


def compatible(folder, config):
    paths = [folder/n for n in ['config.json', 'results.json', 'texture_0500.png']]
    if not all(p.is_file() for p in paths):
        return False
    result = json.loads(paths[1].read_text())
    return (json.loads(paths[0].read_text()) == config and result['complete'] is True
            and result['iterations'] == 500 and result['gradient_frame_forwards'] == 4000
            and result['config_sha256'] == sha(paths[0]) and result['texture_sha256'] == sha(paths[2]))


def train(root):
    import torch
    from PIL import Image
    import texture_eot as E
    from crossarch_backend import Backend
    from crossarch_proxy import images
    from main_method_train import fidelity
    config, rows = specification(root)
    folder = root/'train'/TASK
    if compatible(folder, config):
        print('[train] compatible completed retry reused', flush=True)
        return
    assert not (folder/'config.json').exists(), 'preserve incompatible or interrupted retry; use a new output directory'
    folder.mkdir(parents=True, exist_ok=True)
    dump(folder/'config.json', config)
    backend = Backend('pi', folder); device = backend.policy._pytorch_device
    measured = {name: fidelity([r for r in rows if r['seed'] in seeds], device)
                for name, seeds in [('train', TRAIN_SEEDS), ('validation', VALIDATION_SEEDS)]}
    dump(folder/'proxy_fidelity.json', measured)
    meta = []
    for row in rows:
        with np.load(row['path'], allow_pickle=False) as z:
            meta.append((z['state14'].copy(), str(z['prompt'])))
    clean = []
    with torch.no_grad():
        for i, (state, prompt) in enumerate(meta):
            value, _ = backend.response(backend.prepare(state, prompt, images(None, rows[i]['path'])), 20260901)
            clean.append(value.detach())
        repeated, _ = backend.response(backend.prepare(*meta[0], images(None, rows[0]['path'])), 20260901)
        assert torch.equal(repeated, clean[0])
    def evaluate_by_stage():
        result = {}
        with torch.no_grad():
            for name, seeds in [('train', TRAIN_SEEDS), ('validation', VALIDATION_SEEDS)]:
                result[name] = {}
                for stage in STAGES:
                    ids = [i for i, r in enumerate(rows) if r['seed'] in seeds and r['stage'] == stage]
                    assert ids, ('missing split/stage', name, stage)
                    values = []
                    for i in ids:
                        state, prompt = meta[i]
                        value, _ = backend.response(backend.prepare(state, prompt, images(q, rows[i]['path'])), 20260901)
                        values.append(float((value.float()-clean[i].float()).square().sum()))
                    result[name][stage] = dict(n=len(values), mean_L0=float(np.mean(values)))
        return result
    q = torch.full((1, 3, 128, 128), .5, device=device, requires_grad=True)
    initial_by_stage = evaluate_by_stage()
    dump(folder/'initial_proxy_L0.json', initial_by_stage)
    eot = E.plan(500, 8, 2026102901); plan = training_plan(rows)
    history = []; batches = []; started = time.monotonic()
    for it in range(500):
        ids = plan[it]
        q.grad = None; values = []
        for slot, i in enumerate(ids):
            state, prompt = meta[i]
            value, _ = backend.response(backend.prepare(state, prompt, images(E.transform(q, eot[it][slot]), rows[i]['path'])), 20260901)
            loss = (value.float()-clean[i].float()).square().sum()
            (-loss/8).backward(); values.append(float(loss.detach()))
        assert q.grad is not None and torch.isfinite(q.grad).all() and float(q.grad.norm()) > 0
        assert all(p.grad is None for p in backend.model.parameters()), 'policy must remain frozen'
        with torch.no_grad():
            q.sub_(.01*q.grad.sign()).clamp_(0, 1)
        history.append(float(np.mean(values))); batches.append(ids)
        if it == 0 or (it+1) % 25 == 0:
            texture = q.detach().cpu().numpy()[0]
            Image.fromarray(np.rint(texture.transpose(1, 2, 0)*255).clip(0, 255).astype(np.uint8)).save(folder/f'texture_{it+1:04d}.png')
            np.savez_compressed(folder/f'checkpoint_{it+1:04d}.npz', Q=texture, history=history, batches=batches)
            dump(folder/'progress.json', dict(complete=it+1 == 500, iterations=it+1,
                 loss=history[-1], elapsed_training_s=time.monotonic()-started))
            print('[train]', it+1, history[-1], flush=True)
    dump(folder/'results.json', dict(complete=True, iterations=500, gradient_frame_forwards=4000,
         texture_sha256=sha(folder/'texture_0500.png'), config_sha256=sha(folder/'config.json'),
         elapsed_training_s=time.monotonic()-started, proxy_fidelity=measured,
         initial_proxy_L0_by_stage=initial_by_stage, final_proxy_L0_by_stage=evaluate_by_stage(),
         checkpoint_selection='fixed final500; no test-driven selection'))


def write_manifest(root):
    from PIL import Image
    from render_geometry import patterns
    train_dir = root/'train'/TASK
    config, _ = specification(root)
    assert compatible(train_dir, config)
    texture = root/'random.png'
    if not texture.exists():
        Image.fromarray(patterns()['random']).save(texture)
    value = dict(task=TASK, protocol_sha256=sha(root/'protocol.json'),
                 training_result_sha256=sha(train_dir/'results.json'),
                 textures=dict(clean=None, random=dict(path=str(texture), sha256=sha(texture)),
                               method=dict(path=str(train_dir/'texture_0500.png'), sha256=sha(train_dir/'texture_0500.png'))))
    path = root/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == value, 'preserve changed retry evaluation manifest'
    else:
        dump(path, value)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=DEFAULT)
    args = ap.parse_args(); root = args.root.resolve(); train(root); write_manifest(root)
