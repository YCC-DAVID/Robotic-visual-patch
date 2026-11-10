"""One stage, one frozen policy, one 500x8 texture; matched plans across models."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from crossarch_common import STAGES, request, dump
from crossarch_capture import records
from crossarch_stage_common import DEFAULT, load


def train(root, model, stage):
    p = load(root)
    review = json.loads((root / 'audit/proxy_review.json').read_text())
    assert review['accepted_for_stage_experiment']
    assert review['models'][model]['pixel_pass']
    rs = [r for seed in p['data']['train_seeds'] for r in records(root, seed) if r['stage'] == stage]
    assert len(rs) == 384 and all(Path(r['path']).exists() for r in rs)
    rng = np.random.default_rng(p['patch']['seed'] + STAGES.index(stage))
    plan = [[dict(record=rs[int(rng.integers(len(rs)))]['path'], seed=int(rng.integers(1, 2**30)))
             for _ in range(p['patch']['batch'])] for _ in range(p['patch']['steps'])]
    path = root / 'plans' / (stage + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            assert json.loads(path.read_text()) == plan
        else:
            dump(path, plan)
    dest = root / 'patches' / model / stage
    dest.mkdir(parents=True, exist_ok=True)
    checkpoint = dest / 'training_state.npz'
    q = np.full((1, 3, 128, 128), .5, np.float32)
    history, start = [], 0
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            q, start, history = z['texture'].copy(), int(z['iteration']), json.loads(str(z['history']))
    began = time.monotonic()
    for it in range(start, p['patch']['steps']):
        gradient = np.zeros_like(q)
        values = []
        for draw in plan[it]:
            result = request(model, dict(command='gradient', texture=q, **draw))
            gradient += result['gradient'] / p['patch']['batch']
            values.append(result['L0'])
        norm = float(np.linalg.norm(gradient))
        assert np.isfinite(gradient).all() and np.isfinite(values).all() and norm > 0
        q = np.clip(q + p['patch']['lr'] * np.sign(gradient), 0, 1)
        history.append(float(np.mean(values)))
        temp = checkpoint.with_suffix('.tmp.npz')
        np.savez_compressed(temp, iteration=it+1, texture=q, history=json.dumps(history))
        temp.replace(checkpoint)
        if it == 0 or (it+1) % 25 == 0:
            Image.fromarray(np.rint(q[0].transpose(1, 2, 0)*255).astype(np.uint8)).save(dest / f'texture_{it+1:04d}.png')
        progress = dict(model=model, specialist_stage=stage, iteration=it+1, total=p['patch']['steps'],
            complete=it+1 == p['patch']['steps'], train_L0=history[-1], gradient_norm=norm,
            elapsed_s_this_run=time.monotonic()-began)
        dump(dest / 'progress.json', progress)
        print('[stage train]', progress, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=DEFAULT)
    ap.add_argument('--model', choices=['pi', 'wam'], required=True)
    ap.add_argument('--stage', choices=STAGES, required=True)
    args = ap.parse_args()
    train(args.root, args.model, args.stage)
