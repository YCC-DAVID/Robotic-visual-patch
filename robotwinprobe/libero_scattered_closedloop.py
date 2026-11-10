"""Closed-loop ASR of scattered (non-rectangular) crops of the max-min+EOT joint texture (result/51).

Textures come from libero_masked_texture_prep.py: inside an IG(both-model) mask the joint texture,
outside it a fitted table-camouflage print (MuJoCo ignores texel alpha).  Same 13cm site101 footprint,
init 40-49, native per-model rollout protocol and frozen models as result/51; nothing is trained.
Conditions: clean, table_only (camouflage check, expect ASR 0), ig50, ig70, ig80 (scattered, 50/70/80%
area), blob70 (smooth random shape, 70%).  Also plain-proxy L0 of every PNG on the 32 validation frames.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

import libero_joint_maxmin as M
from crossarch_common import dump, sha

ROOT = M.ROOT
RUN = ROOT / 'robotwinprobe/out/libero13_joint_eot_20261029'
DEFAULT = ROOT / 'robotwinprobe/out/libero13_scattered_closedloop_20261104'
REPORT = ROOT / 'result/60_libero_scattered_closedloop.md'
CONDITIONS = ['clean', 'table_only', 'ig50', 'ig70', 'ig80', 'blob70']
TEXTURES = CONDITIONS[1:]
KINDS = M.KINDS
M.PORT = {'pi': 18741, 'fast': 18742}
REF51 = {'pi': '7/9', 'fast': '7/10'}  # full eot_nominal texture, result/51


def prepare(root):
    for name in ['logs', 'audit']:
        (root / name).mkdir(exist_ok=True)
    for t in TEXTURES:
        assert (root / 'textures' / f'{t}.png').exists(), t
    p = dict(experiment='libero13_scattered_closedloop', source_texture=str(RUN / 'textures/eot_nominal.png'),
             textures={t: sha(root / 'textures' / f'{t}.png') for t in TEXTURES},
             masks='IG of both models w.r.t. texture-space mask on 16 training frames (libero13_mask_crop_20261104), top-f after sigma=2 blur',
             outside_mask='fitted table camouflage texture (MuJoCo 3.2.3 ignores texel alpha)',
             evaluation='fresh clean (must equal result/51 clean per init) then conditions, init40-49, own clean denominator',
             models='frozen, no training')
    path = root / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == p
    else:
        dump(path, p)
    for kind in KINDS:
        r = root / 'models' / kind; r.mkdir(parents=True, exist_ok=True)
        cfg = json.loads((RUN / 'models' / kind / 'protocol.json').read_text())
        cfg.update(experiment=p['experiment'], control_port=M.PORT[kind], parent_root=str(root))
        cfg['evaluation'].update(conditions=CONDITIONS, baseline_reuse=False,
                                 texture_paths={t: str(root / 'textures' / f'{t}.png') for t in TEXTURES})
        dest = r / 'protocol.json'
        if dest.exists():
            assert json.loads(dest.read_text()) == cfg
        else:
            dump(dest, cfg)
        if kind == 'fast':
            for name in ['prompts.npz', 'checkpoints/libero_uncond_2cam224.pt', 'checkpoints/libero_uncond_2cam224_dataset_stats.json']:
                dest = r / name; dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    os.link(RUN / 'models/fast' / name, dest)
    cfgdir = root / 'libero_config'; cfgdir.mkdir(exist_ok=True)
    (cfgdir / 'config.yaml').write_text((RUN / 'libero_config/config.yaml').read_text())


def report(root):
    s = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {'stage': 'prepared'}
    meta = json.loads((root / 'textures/meta.json').read_text())
    lines = ['# LIBERO 13cm：零散非矩形裁剪的闭环ASR', '', f'状态：`{s["stage"]}`；节点：`{s.get("host", "unknown")}`。', '',
             '源纹理：第51号最大最小+EOT联训纹理（整块闭环 π 7/9、FastWAM 7/10）。不重训，只裁剪。',
             '掩码：在掩码裁剪代理测试（libero13_mask_crop_20261104）的16张训练帧上，按两模型对掩码的积分梯度之和取前f面积。',
             'MuJoCo不支持纹理透明，掩码外印拟合的桌面伪装纹理；验证帧上贴纸区域RMSE '
             f'{meta["table_fit_rmse_validation"]["table_fit"]:.3f}（灰色贴纸{meta["table_fit_rmse_validation"]["gray"]:.3f}）。'
             'table_only即整块印桌面，用来检验伪装（应为0）。', '']
    f = root / 'proxy_l0.json'
    if f.exists():
        a = json.loads(f.read_text())
        lines += ['验证帧（32帧）代理L0，括号内为相对整块纹理的比例：', '', '| 纹理 | 面积 | π | FastWAM |', '|---|---:|---:|---:|']
        for t in ['full', 'random'] + TEXTURES:
            area = meta['masks'].get(t, {}).get('area', {'full': 1., 'random': 1., 'table_only': 0.}.get(t))
            lines.append(f'| {t} | {area:.2f} | ' + ' | '.join(f'{a[k][t]:.4g}（{a[k][t] / a[k]["full"]:.2f}）' for k in KINDS) + ' |')
        lines.append('')
    lines += ['| 目标 | 条件 | 完成 | 配对失败/clean成功 |', '|---|---|---:|---:|']
    for kind in KINDS:
        rows = M.outcomes(root, kind); clean = {i for i, v in rows.get('clean', {}).items() if v}
        for c in CONDITIONS:
            r = rows.get(c, {}); pair = clean & r.keys()
            score = '—' if c == 'clean' or not pair else f'{sum(not r[i] for i in pair)}/{len(pair)}'
            lines.append(f'| {kind} | {c} | {len(r)}/10 | {score} |')
        lines.append(f'| {kind} | 整块（第51号） | 10/10 | {REF51[kind]} |')
    if s.get('error'):
        lines += ['', s['error']]
    lines += ['', '单任务10初态，探索性。', f'原始目录：`{root}`。']
    REPORT.write_text('\n'.join(lines) + '\n')


def status(root, stage, **kw):
    dump(root / 'status.json', dict(stage=stage, pid=os.getpid(), host=socket.gethostname(), updated=time.time(), **kw)); report(root)


def proxy_l0(root):
    if (root / 'proxy_l0.json').exists():
        return
    idx = M.parallel_rpc('validation_indices')['pi']['indices']
    paths = {'full': RUN / 'textures/eot_nominal.png', 'random': M.SOURCE / 'textures/random.png'}
    paths.update({t: root / 'textures' / f'{t}.png' for t in TEXTURES})
    out = {k: {} for k in KINDS}
    for t, path in paths.items():
        res = M.parallel_rpc('batch', texture=M.load_png(path), indices=idx, seeds=[20260901] * len(idx), gradient=False)
        for k in KINDS:
            out[k][t] = float(np.mean(res[k]['losses']))
    out['indices'] = idx
    dump(root / 'proxy_l0.json', out)


def pipeline(root, gpus):
    """gpus = {kind: (uuid, physical_index)}; backend and rollouts of a model share its card."""
    M.SERIAL_RPC = len({g[1] for g in gpus.values()}) == 1
    root = root.absolute()
    lock = (root / 'pipeline.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepare(root); services = {}; rollouts = []
    snap = root / 'source_snapshots' / str(os.getpid()); snap.mkdir(parents=True)
    for name in ['libero_scattered_closedloop.py', 'libero_masked_texture_prep.py', 'libero_mask_crop_probe.py',
                 'libero_joint_maxmin.py', 'libero_pi_adapter.py', 'fastwam_adapter.py', 'fastwam_libero_trial.py']:
        shutil.copy2(ROOT / 'robotwinprobe' / name, snap / name)
    dump(snap / 'manifest.json', {f.name: sha(f) for f in snap.glob('*.py')})
    envs = {k: dict(os.environ, WAMATTACK_PROJECT_ROOT=str(ROOT), WAMATTACK_LIBERO_CONFIG_PATH=str(root / 'libero_config'),
                    CUDA_VISIBLE_DEVICES=gpus[k][0], MUJOCO_EGL_DEVICE_ID=str(gpus[k][1]), PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4',
                    HF_HUB_OFFLINE='1', JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false') for k in KINDS}
    try:
        for kind in KINDS:
            status(root, 'start_' + kind)
            ready = root / 'models' / kind / 'ready.json'
            if ready.exists():
                ready.unlink()
            with (root / 'logs' / (kind + '_backend.log')).open('a') as log:
                services[kind] = subprocess.Popen([M.PY[kind], '-u', str(Path(__file__).absolute()), 'server', '--root', str(root), '--kind', kind],
                                                  cwd=ROOT / 'robotwinprobe', env=envs[kind], stdout=log, stderr=subprocess.STDOUT)
        t0 = time.monotonic()
        while not all((root / 'models' / k / 'ready.json').exists() for k in services):
            if any(p.poll() is not None for p in services.values()) or time.monotonic() - t0 > 900:
                raise RuntimeError('backend initialization failed; see logs')
            time.sleep(2)
        dump(root / 'execution.json', dict(host=socket.gethostname(), pipeline_pid=os.getpid(), gpus=gpus,
                                           backends={k: v.pid for k, v in services.items()}, started_at=time.time()))
        status(root, 'native_audit'); M.parallel_rpc('audit')
        status(root, 'proxy_l0'); proxy_l0(root); M.parallel_rpc('trim')
        for c in CONDITIONS:
            for index in range(40, 50):
                status(root, f'evaluate_{c}_{index}')
                jobs = []
                for kind in KINDS:  # pi and fast rollouts run concurrently (separate backends)
                    path = root / 'models' / kind / 'episodes' / c / f'init{index}.json'
                    if path.exists() and 'success' in json.loads(path.read_text()):
                        continue
                    log = (root / 'logs' / f'{kind}_{c}_{index}.log').open('a')
                    proc = subprocess.Popen([M.SIM, '-u', str(ROOT / 'robotwinprobe/fastwam_libero_trial.py'), 'rollout', '--root',
                                             str(root / 'models' / kind), '--condition', c, '--index', str(index)],
                                            cwd=ROOT, env=dict(envs[kind], CUDA_VISIBLE_DEVICES=str(gpus[kind][1])), stdout=log, stderr=subprocess.STDOUT)
                    rollouts.append(proc); jobs.append((kind, proc, log))
                for kind, proc, log in jobs:
                    rc = proc.wait(); log.close(); rollouts.remove(proc)
                    if rc:
                        raise RuntimeError(f'{kind} {c} init{index} rollout failed')
            if c == 'clean':
                for kind in KINDS:
                    assert M.outcomes(root, kind)['clean'] == M.outcomes(RUN, kind)['clean'], (kind, 'clean baseline changed vs result/51')
        status(root, 'complete')
    except Exception as e:
        status(root, 'runtime_error', error=repr(e), traceback=traceback.format_exc()); raise
    finally:
        for proc in rollouts:
            if proc.poll() is None:
                proc.terminate(); proc.wait(timeout=30)
        for kind, proc in services.items():
            if proc.poll() is None:
                try:
                    from multiprocessing.connection import Client
                    with Client(('127.0.0.1', M.PORT[kind]), authkey=M.AUTH) as c:
                        c.send({'command': 'shutdown'}); c.recv()
                except Exception:
                    proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['pipeline', 'server', 'prepare', 'report'])
    ap.add_argument('--root', type=Path, default=DEFAULT); ap.add_argument('--kind', choices=KINDS)
    ap.add_argument('--gpu', action='append', default=[], help='kind=UUID:index'); a = ap.parse_args()
    if a.command == 'pipeline':
        g = {}
        for x in a.gpu:
            kind, spec = x.split('=')
            uuid, index = spec.rsplit(':', 1)
            g[kind] = (uuid, int(index))
        assert set(g) == set(KINDS), g
        pipeline(a.root, g)
    elif a.command == 'server':
        M.server(a.root, a.kind)
    elif a.command == 'prepare':
        prepare(a.root); status(a.root, 'prepared')
    else:
        report(a.root)
