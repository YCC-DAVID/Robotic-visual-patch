"""Show final learned textures at their own original sites using saved true RT."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import bootstrap as B
from radius_plan import dump
from table9_plan import DEFAULT, records, sha
from render_geometry import CAMERAS


def quad(intrinsic,extrinsic,center,size):
    world=np.array([[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[-.5,.5,0]])*size+center
    camera=world@extrinsic[:,:3].T+extrinsic[:,3]
    assert (camera[:,2]>0).all()
    projected=camera@intrinsic.T
    return projected[:,:2]/projected[:,2:]


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    p=json.loads((root/'protocol.json').read_text())
    dest=root/'comparison/location_preview';dest.mkdir(parents=True,exist_ok=True)
    frames=[];manifest=[];common_state=None;common_clean=None
    for site in p['sites']:
        sid=site['id'];record=records(root,sid,'post')[0]
        assert record['yaw_deg']==0 and record['light']=='nominal'
        path=Path(record['path']);clean_path=Path(record['clean_path'])
        saved_hashes=json.loads((root/'sites'/sid/'post_texture_hashes.json').read_text())
        texture=root/'sites'/sid/'train/texture_0500.png'
        assert sha(texture)==saved_hashes[sid]
        with np.load(path) as z,np.load(clean_path) as c:
            assert np.array_equal(z['center_m'],np.array(site['center_m']))
            assert float(z['size_m'])==.10 and float(z['yaw_deg'])==0
            assert np.array_equal(z['state14'],c['state14'])
            if common_state is None:
                common_state=z['state14'].copy();common_clean={n:c[n].copy() for n in CAMERAS}
            assert np.array_equal(z['state14'],common_state)
            for n in CAMERAS:assert np.array_equal(c[n],common_clean[n])
            frame=dict(site=sid,center=site['center_m'],images={},quads={},masks={})
            for n in CAMERAS:
                rgb=z[n+'_'+sid].copy()
                frame['images'][n]=rgb
                frame['masks'][n]=z[n+'_mask'].copy()
                frame['quads'][n]=quad(z[n+'_intrinsic'],z[n+'_extrinsic'],np.array(site['center_m']),.10)
                Image.fromarray(rgb).save(dest/f'{sid}_{n}.png')
            frames.append(frame)
            manifest.append(dict(site=sid,center_m=site['center_m'],size_m=.10,
                seed=int(z['seed']),query=int(z['query']),frame=int(z['frame']),
                yaw_deg=0,light='nominal',texture=str(texture),texture_sha256=sha(texture),
                capture=str(path),capture_sha256=sha(path),clean_capture=str(clean_path),
                camera_visible_pixels={n:int(frame['masks'][n].sum()) for n in CAMERAS}))
    colors=plt.cm.tab10(np.arange(9))
    fig,axs=plt.subplots(3,3,figsize=(14,11))
    for ax,f,color in zip(axs.flat,frames,colors):
        ax.imshow(f['images']['head_camera'])
        ax.add_patch(Polygon(f['quads']['head_camera'],fill=False,edgecolor=color,linewidth=1.6))
        x,y,_=f['center'];ax.set_title(f"{f['site']} | X={x:+.2f} m, Y={y:+.2f} m",fontsize=12)
        ax.axis('off')
    fig.suptitle('RoboTwin | final trained 10 cm patches at their own sites\nSame scene state: seed 10110, query 0 | each panel is a separate true render',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(dest/'nine_trained_head.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,9))
    ax.imshow(common_clean['head_camera'])
    for f,color in zip(frames,colors):
        q=f['quads']['head_camera'];center=q.mean(0)
        ax.add_patch(Polygon(q,fill=False,edgecolor=color,linewidth=2))
        ax.text(*center,f['site'][4:],color='white',ha='center',va='center',fontsize=13,
                bbox=dict(boxstyle='round,pad=.12',facecolor=color,alpha=.9,edgecolor='none'))
    ax.set_title('Position map on clean head-camera image\nColored outlines and IDs only; no patch textures in this map',fontsize=14)
    ax.axis('off');fig.tight_layout();fig.savefig(dest/'positions_on_clean.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(9,3,figsize=(13,26))
    labels=['Head camera','Left wrist','Right wrist']
    for i,f in enumerate(frames):
        for j,(n,label) in enumerate(zip(CAMERAS,labels)):
            ax=axs[i,j];ax.imshow(f['images'][n]);ax.axis('off')
            q=f['quads'][n]
            ax.add_patch(Polygon(q,fill=False,edgecolor=colors[i],linewidth=1))
            h,w=f['images'][n].shape[:2];ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5)
            ax.set_title(f"{f['site']} | {label}",fontsize=11)
    fig.suptitle('Same world patch observed by all three cameras | one separate scene per row',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(dest/'nine_trained_three_views.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for f,color in zip(frames,colors):
        x,y,_=f['center']
        q=np.array([[-.05,-.05],[.05,-.05],[.05,.05],[-.05,.05]])+[x,y]
        ax.add_patch(Polygon(q,fill=False,edgecolor=color,linewidth=1.5,linestyle='--'))
        ax.scatter(x,y,color=color,s=35);ax.annotate(f['site'],(x,y),xytext=(0,8),textcoords='offset points',ha='center')
    ax.set(xlim=(-.40,.16),ylim=(.08,.32),xlabel='World X (m)',ylabel='World Y (m)',
        title='Original training centers | Z=0.741 m | dashed squares are 10 cm footprints')
    ax.set_aspect('equal');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(dest/'world_xy_positions.png',dpi=180);plt.close(fig)
    dump(dest/'manifest.json',dict(source='existing actual SAPIEN ray-tracing captures; no texture image-space compositing',
        shared_scene_state_and_clean_RGB_verified=True,single_patch_per_render=True,sites=manifest,
        panels='each final texture shown at its own nominal training center',script_sha256=sha(__file__)))
    print(dest,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(DEFAULT))
    main(Path(ap.parse_args().root).resolve())
