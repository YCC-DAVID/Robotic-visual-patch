"""Frozen per-run physical evaluation manifest and frequency-matched random PNG."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--stage',choices=['development','confirmation'],default='development')
    args=ap.parse_args();run=Path(args.run).resolve();dest=Path(args.out).resolve();dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'manifest.json').exists()
    cfg=json.loads((run/'config.json').read_text());assert cfg['complete']
    n=cfg['parameter_shape'][-1];seed=20261008
    latent=np.random.RandomState(seed).randint(0,256,(1,3,n,n)).astype(np.float32)/255
    q=torch.from_numpy(latent)
    if n!=128:q=F.interpolate(q,(128,128),mode='bilinear',align_corners=False)
    img=np.rint(q.numpy()[0].transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
    Image.fromarray(img).save(dest/'random.png');np.savez_compressed(dest/'random.npz',latent=latent,Q=q.numpy(),seed=seed)
    optimized=run/'texture_0500.png';assert optimized.is_file()
    textures={'clean':None,'random':str(dest/'random.png'),'optimized':str(optimized)}
    info={'stage':args.stage,'geometry':{'patch_m':cfg['patch_m'],'cell_world':cfg['cell_world']},
        'frequency':cfg.get('frequency_parameterization','full (frozen E22 Q128 reference)'),'mode':cfg['mode'],'train_run':str(run),
        'train_config_sha256':hashlib.sha256((run/'config.json').read_bytes()).hexdigest(),'textures':textures,
        'texture_sha256':{c:None if p is None else hashlib.sha256(Path(p).read_bytes()).hexdigest() for c,p in textures.items()},
        'random_seed':seed,'random_parameter_shape':[1,3,n,n],
        'training_indices':list(range(10)),'validation_indices':[10,11],
        'development_indices':list(range(40,45)),'confirmation_indices':list(range(45,50)),
        'notes':'Same raw simulator trajectories, per-geometry calibration, 500x8 training. Both cameras see real render. Development selects sizes; confirmation must not be used to adapt a selected candidate.'}
    (dest/'manifest.json').write_text(json.dumps(info,indent=2)+'\n');print(dest/'manifest.json')


if __name__=='__main__':main()
