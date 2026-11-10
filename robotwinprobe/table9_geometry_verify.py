"""Verify selected sites with real segmentation, independent renders per site."""
import argparse
import json
from pathlib import Path

import numpy as np
from radius_plan import dump
from table9_plan import sha
from render_geometry import CAMERAS


# Fixed before reading the actual segmentation verification. Rasterization at
# ~25x25 model pixels permits a small boundary discrepancy from depth screening.
THRESHOLDS = dict(min_train_head_exposure=.85, mean_train_head_exposure=.95,
                  max_between_site_mean_head_pixel_ratio=1.30,
                  expected_training_frames=54, expected_validation_frames=9)


def projected_area(k, e, pose, size=.10):
    angle=np.deg2rad(pose[3]);c,s=np.cos(angle),np.sin(angle)
    xy=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])*size
    xy=xy@np.array([[c,s],[-s,c]])+pose[:2]
    world=np.c_[xy,np.full(4,pose[2])]
    cv=world@e[:,:3].T+e[:,3]
    if (cv[:,2]<=0).any():return None
    uv=cv@k.T;uv=uv[:,:2]/uv[:,2:]
    x,y=uv.T
    return float(abs((x*np.roll(y,-1)-y*np.roll(x,-1)).sum())/2)


def model_mask(mask):
    # Same indices as torch interpolate(mode='nearest') 240x320 ->168x224.
    h,w=mask.shape
    return mask[(np.arange(168)*h//168)[:,None],(np.arange(224)*w//224)[None,:]]


def measure(root):
    selection=json.loads((root/'geometry/selection.json').read_text())
    rows=[];files={}
    for path in sorted((root/'geometry/verified').glob('seed*.npz')):
        files[str(path.relative_to(root))]=sha(path)
        with np.load(path) as z, np.load(root/'geometry/clean'/path.name) as clean:
            seed=int(z['seed']);indices={int(q):i for i,q in enumerate(clean['query'])}
            for i,q in enumerate(z['query']):
                ci=indices[int(q)]
                assert np.array_equal(z['state14'][i],clean['state14'][ci])
                for site in selection['centers']:
                    sid=site['id']
                    for pi in [0,1]:
                        pose=z[f'{sid}_p{pi}_pose'][i]
                        row=dict(seed=seed,query=int(q),site=sid,pose=pose.tolist(),
                                 probe='nominal' if pi==0 else 'boundary',
                                 split='train' if seed<10110 else 'validation',cameras={})
                        for camera in CAMERAS:
                            mask=z[f'{sid}_p{pi}_{camera}_mask'][i]
                            area=projected_area(clean[camera+'_intrinsic'][ci],clean[camera+'_extrinsic'][ci],pose)
                            row['cameras'][camera]=dict(native_visible_pixels=int(mask.sum()),
                                model_visible_pixels=int(model_mask(mask).sum()),full_native_projected_area=area,
                                exposed_fraction_raw=float(mask.sum()/area) if area else None)
                        rows.append(row)
    return selection,rows,files


def verify(root,partial=False):
    target=root/'geometry/verification.json'
    if target.exists() and not partial:
        saved=json.loads(target.read_text())
        assert saved['selection_sha256']==sha(root/'geometry/selection.json')
        for p,h in saved['capture_sha256'].items():assert sha(root/p)==h
        assert saved['passed'],'Prior geometry verification failed; inspect before continuing'
        return saved
    selection,rows,files=measure(root)
    stats=[]
    for site in selection['centers']:
        sid=site['id'];summary=dict(site=sid,center_m=site['center_m'])
        for split in ['train','validation']:
            subset=[r for r in rows if r['site']==sid and r['split']==split]
            if not subset:continue
            values=np.array([r['cameras']['head_camera']['exposed_fraction_raw'] for r in subset])
            summary[split]=dict(frames=len(subset)//2,minimum_head_exposure=float(values.min()),
                mean_head_exposure=float(np.minimum(values,1).mean()),
                mean_head_model_pixels=float(np.mean([r['cameras']['head_camera']['model_visible_pixels'] for r in subset])),
                cameras={})
            for camera in CAMERAS:
                pixels=np.array([r['cameras'][camera]['model_visible_pixels'] for r in subset if r['probe']=='nominal'])
                summary[split]['cameras'][camera]=dict(mean_nominal_visible_pixels=float(pixels.mean()),
                    fraction_nominal_frames_visible=float((pixels>0).mean()))
        stats.append(summary)
    train=[s['train'] for s in stats if 'train' in s]
    ratio=max(s['mean_head_model_pixels'] for s in train)/min(s['mean_head_model_pixels'] for s in train) if train else None
    complete=(root/'geometry/verify_capture_complete.json').exists()
    counts_ok=all(s.get('train',{}).get('frames')==THRESHOLDS['expected_training_frames'] and
                  s.get('validation',{}).get('frames')==THRESHOLDS['expected_validation_frames'] for s in stats)
    passed=bool(complete and counts_ok and len(stats)==9 and ratio<=THRESHOLDS['max_between_site_mean_head_pixel_ratio'] and
        all(s['minimum_head_exposure']>=THRESHOLDS['min_train_head_exposure'] and
            s['mean_head_exposure']>=THRESHOLDS['mean_train_head_exposure'] for s in train))
    result=dict(passed=passed,capture_complete=complete,thresholds=THRESHOLDS,
        selection_sha256=sha(root/'geometry/selection.json'),capture_sha256=files,
        single_patch_per_scene=True,area_matching='head camera; wrist visibility measured, not matched',
        mean_head_model_pixel_ratio=ratio,sites=stats,rows=rows)
    dump(root/'geometry'/('verification_partial.json' if partial else 'verification.json'),result)
    plot(root,selection)
    report(root,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','capture_sha256']},indent=2),flush=True)
    if not partial:assert passed,'Real rendered exposure verification failed; training stays gated'
    return result


def plot(root,selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths=sorted((root/'geometry/verified').glob('seed*.npz'))
    if not paths:return
    with np.load(paths[0]) as z:
        fig,axs=plt.subplots(3,3,figsize=(12,9))
        for ax,site in zip(axs.flat,selection['centers']):
            ax.imshow(z[f"{site['id']}_p0_head_camera_rgb"][0]);ax.axis('off')
            ax.set_title(f"{site['id']} | ({site['center_m'][0]:+.2f}, {site['center_m'][1]:+.2f}) m")
        fig.suptitle('Nine separate real renders; one 10 cm patch in each scene',fontsize=14)
        fig.tight_layout();fig.savefig(root/'geometry/nine_separate_renders.png',dpi=160);plt.close(fig)
        fig,axs=plt.subplots(9,3,figsize=(12,23))
        for i,site in enumerate(selection['centers']):
            for j,camera in enumerate(CAMERAS):
                axs[i,j].imshow(z[f"{site['id']}_p0_{camera}_rgb"][0]);axs[i,j].axis('off')
                axs[i,j].set_title(f"{site['id']} | {camera}")
        fig.tight_layout();fig.savefig(root/'geometry/three_view_visibility.png',dpi=140);plt.close(fig)


def report(root,result):
    lines=['# 九个位置：真实渲染可见面积核验','',
        '每张图是独立场景，场景中始终只有一张 10×10 cm patch。棋盘纹理仅用于几何核验，尚非训练结果。',
        '',f"采集完成：{result['capture_complete']}；训练前几何检查通过：{result['passed']}。",
        '', '主相机匹配无遮挡比例和像素面积；腕部视野随位置变化，单独统计。54 个训练帧是九组共用的无遮挡子集，覆盖全部 8 个训练初态。验证集 9 帧全部保留，不因遮挡或攻击结果筛选。',
        '', '![九张独立渲染图](nine_separate_renders.png)',
        '', '|位置|训练帧数|最小主相机露出比例|平均主相机像素|左腕像素|右腕像素|',
        '|---|---:|---:|---:|---:|---:|']
    for s in result['sites']:
        t=s.get('train')
        if not t:continue
        lines.append(f"|{s['site']}|{t['frames']}|{t['minimum_head_exposure']:.1%}|{t['mean_head_model_pixels']:.1f}|{t['cameras']['left_camera']['mean_nominal_visible_pixels']:.1f}|{t['cameras']['right_camera']['mean_nominal_visible_pixels']:.1f}|")
    lines += ['', '主相机的最小露出和平均像素包含名义位置及边界探针；左右腕为名义位置平均值。露出比例分母是完整投影四边形面积，分子是实际 segmentation 像素数，栅格化可造成小幅超过 100%。',
        '', '![三路真实视野](three_view_visibility.png)', '', '完整逐帧数据和冻结阈值见 `verification.json`（完成前为 `verification_partial.json`）。']
    (root/'geometry/report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--partial',action='store_true')
    args=parser.parse_args();verify(Path(args.root).resolve(),args.partial)
