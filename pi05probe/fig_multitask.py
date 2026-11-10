"""Figures for #59/#60: patch textures and what the policy actually sees per task."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent/'out'
MT, ST = ROOT/'multitask_patch', ROOT/'singletask_patch'
ZH = {'put_the_bowl_on_the_plate': 'bowl→plate (train)', 'put_the_bowl_on_top_of_the_cabinet': 'bowl→cabinet top (train)',
      'put_the_wine_bottle_on_the_rack': 'wine→rack (train)', 'turn_on_the_stove': 'turn on stove (train)',
      'put_the_bowl_on_the_stove': 'bowl→stove (held out)', 'put_the_cream_cheese_in_the_bowl': 'cream cheese→bowl (held out)',
      'open_the_middle_drawer_of_the_cabinet': 'open middle drawer (held out)'}
TASKS = list(ZH)


def textures():
    items = [('random', ROOT/'dual_view/manifest_both/random.png'),
             ('single-task E26\n(bowl→plate)', ROOT/'dual_view/train_both/texture_0500.png'),
             ('multi-task #59\n(4 train tasks)', MT/'train/texture_0500.png')]
    for s, name in [('st_plate', 'bowl→plate'), ('st_cabinet', 'bowl→cabinet'), ('st_wine', 'wine→rack'), ('st_stove', 'stove')]:
        p = ST/'train'/s/'texture_0500.png'
        if p.exists(): items.append((f'single-task new\n({name})', p))
    fig, ax = plt.subplots(1, len(items), figsize=(2.3*len(items), 2.8))
    for a, (title, p) in zip(ax, items):
        a.imshow(Image.open(p)); a.set_title(title, fontsize=9); a.axis('off')
    fig.suptitle('13 cm patch textures (128×128, as uploaded to MuJoCo)', fontsize=11)
    fig.tight_layout(); fig.savefig(MT/'fig_textures.png', dpi=130); plt.close(fig)


def record(task, cond, init, base=None):
    for folder in ['rollout', 'rollout_retry49']:
        p = (base or MT)/'eval'/task/folder/f'init{init}_{cond}.npz'
        if p.exists(): return np.load(p)


def outcome(task, cond, init, base=None):
    for folder in ['rollout_retry49', 'rollout']:
        p = (base or MT)/'eval'/task/folder/'episodes.json'
        if p.exists():
            for r in json.loads(p.read_text())['per_episode'].get(cond, []):
                if r['init_index'] == init: return r['success']


def scenes(init=40, cond='multi_task'):
    cols = ['clean base t=0', f'{cond} base t=0', f'{cond} wrist t=0', f'{cond} wrist mid', 'clean base last', f'{cond} base last']
    fig, ax = plt.subplots(len(TASKS), 6, figsize=(13, 2.25*len(TASKS)))
    for i, task in enumerate(TASKS):
        c, m = record(task, 'clean', init), record(task, cond, init)
        vis = m['visible_pixels224']
        mid = int(np.argmax(vis[:, 1])) if vis[:, 1].max() > 0 else len(vis)//2
        imgs = [c['base224'][0], m['base224'][0], m['wrist224'][0], m['wrist224'][mid], c['base224'][-1], m['base224'][-1]]
        notes = ['', f'patch px {vis[0,0]}', f'patch px {vis[0,1]}', f'frame {mid}, px {vis[mid,1]}',
                 'success' if outcome(task, 'clean', init) else 'FAIL', 'success' if outcome(task, cond, init) else 'FAIL']
        for j, (img, note) in enumerate(zip(imgs, notes)):
            a = ax[i, j]; a.imshow(img); a.set_xticks([]); a.set_yticks([])
            if i == 0: a.set_title(cols[j], fontsize=9)
            a.set_xlabel(note, fontsize=8, color='red' if note == 'FAIL' else 'black')
        ax[i, 0].set_ylabel(ZH[task], fontsize=9)
    fig.suptitle(f'Actual policy inputs (real MuJoCo render), init {init}; wrist "mid" = frame with most visible patch', fontsize=11)
    fig.tight_layout(); fig.savefig(MT/f'fig_scenes_{cond}_init{init}.png', dpi=110); plt.close(fig)


OWN = {'put_the_bowl_on_the_plate': 'st_plate', 'put_the_bowl_on_top_of_the_cabinet': 'st_cabinet',
       'put_the_wine_bottle_on_the_rack': 'st_wine', 'turn_on_the_stove': 'st_stove'}


def own_scenes(init=40):
    fig, ax = plt.subplots(4, 5, figsize=(11, 9.4))
    for i, (task, cond) in enumerate(OWN.items()):
        c, m = record(task, 'clean', init, ST), record(task, cond, init, ST)
        vis = m['visible_pixels224']; mid = int(np.argmax(vis[:, 1]))
        imgs = [np.asarray(Image.open(ST/'train'/cond/'texture_0500.png')), m['base224'][0], m['wrist224'][0], m['wrist224'][mid], m['base224'][-1]]
        notes = [cond, f'patch px {vis[0,0]}', f'patch px {vis[0,1]}', f'frame {mid}, px {vis[mid,1]}',
                 'success' if outcome(task, cond, init, ST) else 'FAIL']
        for j, (img, note) in enumerate(zip(imgs, notes)):
            a = ax[i, j]; a.imshow(img); a.set_xticks([]); a.set_yticks([]); a.set_xlabel(note, fontsize=8)
            if i == 0: a.set_title(['own single-task texture', 'base t=0', 'wrist t=0', 'wrist, most patch', 'base last'][j], fontsize=9)
        ax[i, 0].set_ylabel(ZH[task], fontsize=9)
    fig.suptitle(f'Each single-task patch in its own task (init {init}, real render)', fontsize=11)
    fig.tight_layout(); fig.savefig(MT/f'fig_scenes_own_single_init{init}.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    textures(); scenes(40, 'multi_task'); scenes(40, 'single_task'); own_scenes(40)
    print('ok')
