"""Frozen learnedQ and random32 nearest-upsampled control matching block structure."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    run=Path(args.run).resolve();dest=Path(args.out).resolve();dest.mkdir(parents=True,exist_ok=True);assert not (dest/'manifest.json').exists()
    cfg=json.loads((run/'config.json').read_text());assert cfg['complete']
    seed=20261008;latent=np.random.RandomState(seed).randint(0,256,(1,3,32,32)).astype(np.float32)/255
    q=np.repeat(np.repeat(latent,4,axis=2),4,axis=3);image=np.rint(q[0].transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
    Image.fromarray(image).save(dest/'random.png');np.savez_compressed(dest/'random.npz',latent=latent,Q=q,seed=seed)
    textures={'clean':None,'random':str(dest/'random.png'),'optimized':str(run/'texture_0500.png')}
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    d={'stage':'exploratory_retest','geometry':{'patch_m':cfg['patch_m'],'cell_world':cfg['cell_world']},'frequency':'32-grid constant4x4blocks; nearest expansion, NOT bilinear band32',
       'mode':cfg['mode'],'train_run':str(run),'train_config_sha256':sha(run/'config.json'),'textures':textures,
       'texture_sha256':{c:None if p is None else sha(p) for c,p in textures.items()},'random_seed':seed,'random_parameter_shape':[1,3,32,32],
       'notes':'Matched random control chosen after checking analytical block invariant, before final training checkpoint or any new eval outcome. Same geometry/noise/clean reference; both views real render.'}
    (dest/'manifest.json').write_text(json.dumps(d,indent=2)+'\n')

if __name__=='__main__':main()
