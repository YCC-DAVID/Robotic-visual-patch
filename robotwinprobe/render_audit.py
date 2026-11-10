"""Independent dataset split, visibility, and UV direction checks."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from render_proxy import grid_for
from render_geometry import CAMERAS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='robotwinprobe/out/render_transfer')
    args=ap.parse_args();root=Path(args.root)
    paths=sorted((root/'calibration').glob('seed*.npz'))
    assert [int(p.stem[4:]) for p in paths]==list(range(10100,10108))+[10110,10111]
    rows=[]
    for p in paths:
        with np.load(p) as d,np.load(root/'clean'/p.name) as clean:
            assert np.array_equal(d['state14'],clean['states14'])
            assert np.array_equal(d['query'],np.arange(len(clean['states14'])))
            assert float(d['size_m'])==.20 and np.allclose(d['center_m'],[0,0,.741])
            row={'seed':int(d['seed']),'frames':len(d['state14']),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'views':{}}
            for name in CAMERAS:
                assert np.array_equal(d[name+'_clean'],clean[name])
                correlations=[]
                for i in range(len(d['state14'])):
                    g=grid_for(d[name+'_intrinsic'][i],d[name+'_extrinsic'][i],d['center_m'],float(d['size_m']))
                    mask=d[name+'_mask'][i]&(abs(g).max(-1)<.8)
                    uv=(d[name+'_uv'][i].astype(float)-d[name+'_black'][i])/np.maximum(d[name+'_white'][i].astype(float)-d[name+'_black'][i],1)
                    corr=[]
                    for channel in (0,1):
                        if mask.sum()>100 and np.std(g[...,channel][mask])>.15:
                            value=float(np.corrcoef(g[...,channel][mask],uv[...,channel][mask])[0,1]);assert value>.85,(p,name,i,channel,value)
                            corr.append(value)
                    if corr:correlations.append({'query':i,'correlations':corr})
                row['views'][name]={'patch_pixels':d[name+'_mask'].sum((1,2)).tolist(),'uv_direction':correlations}
            rows.append(row)
    report={'complete':True,'episodes':rows,'train_frames':sum(r['frames'] for r in rows if r['seed']<10110),
            'validation_frames':sum(r['frames'] for r in rows if r['seed']>=10110),
            'checks':{'clean_RGB_bitwise_equal':True,'model_state14_bitwise_equal':True,'UV_axes_correct':True,
                      'seeds_disjoint':True,'physical_geometry_fixed':True}}
    (root/'calibration_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Audited',report['train_frames'],'train frames;',report['validation_frames'],'validation frames')


if __name__=='__main__':main()
