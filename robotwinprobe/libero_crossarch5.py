"""LIBERO-Goal, 5 tasks: does a π0.5+FastWAM joint (max-min + EOT) patch transfer to ImageWAM?

Per task (same recipe as #50/#51 on put_the_bowl_on_the_plate, only the task changes):
  position  : 13cm patch at the per-task visible position (rule fixed before results: among 470 legal
              table positions with base>=300px on >=90% training frames, max mean wrist px).
  tasks     : the 5 of 7 pooled LIBERO-Goal tasks with the largest selected-position mean wrist px.
  training  : π-only and FastWAM-only nominal specialists (sign-PGD, gray init, 500x8, lr .01) give the
              max-min reference curves; then the joint max-min + texture-plane EOT texture (#51 recipe).
  evaluation: real MuJoCo closed loop, inits 30-49 (20), each model's native protocol;
              conditions clean / random / joint; ASR = clean success & attacked failure / clean successes.
              ImageWAM never contributes gradients or selection: zero-shot third-architecture target.
Subcommands: run (GPU-waiting scheduler), job (one scheduled job), server, report.
"""
import argparse
import concurrent.futures
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from multiprocessing.connection import Client, Listener

import numpy as np
from PIL import Image

ROOT = Path('/shared/user65/workspace/chence/WAMattack')
os.environ.setdefault('WAMATTACK_PROJECT_ROOT', str(ROOT))
sys.path.insert(0, str(ROOT/'robotwinprobe'))
from crossarch_common import AUTH, dump, sha  # noqa: E402
from texture_eot import IDENTITY, plan as eot_plan  # noqa: E402

OUT = ROOT/'robotwinprobe/out/libero_crossarch5_20261105'
REPORT = ROOT/'result/61_libero_crossarch5.md'
PIPE = ROOT/'pi05probe'
VIS = PIPE/'out/visible_pos'
ENVROOT = Path('/shared/user65/miniconda3/envs')
PY = dict(pi=str(ENVROOT/'openpi-server/bin/python'), fast=str(ENVROOT/'fastwam/bin/python'),
          imagewam='/shared/user64/workspace/yuhao/imagewam/ImageWAM/.venv/bin/python')
SIM = str(ENVROOT/'openpi-libero/bin/python')
PREFLIGHT = ROOT/'skills/gpu-preflight/scripts/check_gpus.py'
CANDIDATES = ['put_the_bowl_on_the_plate', 'put_the_bowl_on_top_of_the_cabinet', 'put_the_wine_bottle_on_the_rack',
              'turn_on_the_stove', 'put_the_bowl_on_the_stove', 'put_the_cream_cheese_in_the_bowl',
              'open_the_middle_drawer_of_the_cabinet']
N_TASKS = 5
INDICES = list(range(30, 50))
CONDITIONS = ['clean', 'random', 'joint']
KINDS = ['pi', 'fast']
MODELS = ['pi', 'fast', 'imagewam']
STEPS, BATCH, LR = 500, 8, .01
BASE_MIN, BASE_FRAC = 300, .9
SCAN_SHA = sha(PIPE/'visibility_scan.py')
TEMPLATES = dict(pi=ROOT/'robotwinprobe/out/libero13_joint_eot_20261029/models/pi/protocol.json',
                 fast=ROOT/'robotwinprobe/out/libero13_joint_eot_20261029/models/fast/protocol.json',
                 imagewam=ROOT/'robotwinprobe/out/imagewam_libero_eot_transfer_20261030/models/imagewam/protocol.json')
FAST_CKPT = ROOT/'robotwinprobe/out/libero13_joint_eot_20261029/models/fast/checkpoints'
E26_FIT = PIPE/'out/dual_view/fit'
EXCLUDED = {('nnmc63', 0)}  # historical ECC fault
ALLOWED_HOSTS = {'nnmc61', 'nnmc62', 'nnmc63', 'nnmc64', 'nnmc65'}  # 2026-11-07 user instruction
# nnmc71 (H100 NVL): only GPU3 (user-approved), only for pipeline jobs. Its EGL segmentation read_pixels aborts
# (core dump) and RGB re-renders differ in ~10 px/frame, so scan/calibrate (segmentation + exact checks) stay on L40S.
H100_ALLOWED = set()  # 2026-11-07 user: no H100 for now, only nnmc61-65
MAX_RUNNING = 6


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def log(*a):
    print(now(), *a, flush=True)


def fixed(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, ('immutable file changed', str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True); dump(path, value)


# ----------------------------------------------------------------------------- per-task inputs
def pool_path(task):
    old = PIPE/'out/multitask_patch/tasks'/task/'base.npz'
    return old if old.exists() else VIS/'pools'/task/'pool.npz'


def scan_dir(task):
    """Reuse a validated full scan from #60 (same scanner source); otherwise our own."""
    for d in [VIS/'scan'/task, OUT/'scan'/task]:
        try:
            m = json.loads((d/'visibility.json').read_text())
            if m.get('validated') and not m.get('smoke') and m.get('source_sha256') == SCAN_SHA and (d/'visibility.npz').exists():
                return d
        except (OSError, ValueError):
            pass
    return None


def selected_position(task):
    z = np.load(scan_dir(task)/'visibility.npz')
    base, wrist, cand = z['base_px'], z['wrist_px'], z['candidates']
    ok = (base >= BASE_MIN).mean(0) >= BASE_FRAC
    assert ok.any(), task
    j = int(np.argmax(np.where(ok, wrist.mean(0), -1)))
    return dict(task=task, candidate=j, world=[float(v) for v in cand[j]], wrist_mean=float(wrist[:, j].mean()),
                base_mean=float(base[:, j].mean()), eligible=int(ok.sum()), scan=str(scan_dir(task)))


def selection():
    path = OUT/'selection.json'
    if path.exists():
        return json.loads(path.read_text())
    if not all(scan_dir(t) for t in CANDIDATES):
        return None
    rows = {t: selected_position(t) for t in CANDIDATES}
    chosen = sorted(CANDIDATES, key=lambda t: -rows[t]['wrist_mean'])[:N_TASKS]
    value = dict(rule=f'top {N_TASKS} of {len(CANDIDATES)} by mean wrist px at the selected position', positions=rows,
                 tasks=[t for t in CANDIDATES if t in chosen], decided=now())
    dump(path, value)
    return value


def task_root(task):
    return OUT/'tasks'/task


def calibration(task):
    return task_root(task)/'calib'


def ports(task):
    base = 18800+10*CANDIDATES.index(task)
    return dict(pi=base, fast=base+1, imagewam=base+2)


def prepare(task):
    root = task_root(task); cal = calibration(task)
    for name in ['logs', 'audit', 'textures', 'training']:
        (root/name).mkdir(parents=True, exist_ok=True)
    with np.load(cal/'base.npz') as z:
        prompt = str(z['prompt']); world = [float(v) for v in z['cell_world']]
        train = [int(k) for k in np.flatnonzero(z['train_mask'])]
    rng = np.random.default_rng(2026110501)
    fixed(root/'training_plan.json', [dict(indices=[int(v) for v in rng.choice(train, BATCH, replace=False)],
                                          seeds=[int(v) for v in rng.integers(1, 2**30, size=BATCH)]) for _ in range(STEPS)])
    fixed(root/'eot_plan.json', eot_plan())
    fixed(root/'protocol.json', dict(experiment='libero_crossarch5', task=task, prompt=prompt, cell_world=world,
        calibration=str(cal/'base.npz'), calibration_sha256=sha(cal/'base.npz'), indices=INDICES, conditions=CONDITIONS,
        training=dict(steps=STEPS, batch=BATCH, lr=LR, init=.5, plan_seed=2026110501, eot_seed=2026102901,
                      specialists='pi-only and fast-only nominal sign-PGD on the same plan',
                      joint='max-min: step follows sign(grad) of argmin_m L_m/L_m^specialist(same step), + #51 texture-plane EOT'),
        transfer_target='imagewam (no gradients, no selection)'))
    shutil.copy2(cal/'random.png', root/'textures/random.png')
    for kind in MODELS:
        r = root/'models'/kind; (r/'audit').mkdir(parents=True, exist_ok=True)
        cfg = json.loads(TEMPLATES[kind].read_text())
        cfg.update(experiment='libero_crossarch5', task=task, prompt_tasks=[prompt], control_port=ports(task)[kind],
                   calibration=str(cal/'base.npz'), calibration_sha256=sha(cal/'base.npz'), parent_root=str(root))
        cfg['patch'] = dict(size_m=.13, resolution=128, position='per-task visible position', cell_world=world)
        cfg['evaluation'].update(indices=INDICES, conditions=CONDITIONS, baseline_reuse=False,
                                 texture_paths={c: str(root/'textures'/(c+'.png')) for c in CONDITIONS if c != 'clean'},
                                 texture_paths_by_init={})
        for key in ['expected_new_episodes', 'expected_episodes', 'reference']:
            cfg['evaluation'].pop(key, None)
        if kind == 'imagewam':
            cfg['training'] = dict(enabled=False, target_only=True)
        fixed(r/'protocol.json', cfg)
        if kind == 'fast':
            (r/'checkpoints').mkdir(exist_ok=True)
            for f in FAST_CKPT.iterdir():
                if not (r/'checkpoints'/f.name).exists():
                    os.link(f, r/'checkpoints'/f.name)
    config = root/'libero_config'; config.mkdir(exist_ok=True)
    (config/'config.yaml').write_text((ROOT/'robotwinprobe/out/libero13_joint_eot_20261029/libero_config/config.yaml').read_text())


def outcomes(task, kind):
    rows = {}
    for c in (task_root(task)/'models'/kind/'episodes').glob('*'):
        rows[c.name] = {v['init_index']: v['success'] for f in c.glob('init*.json')
                        for v in [json.loads(f.read_text())] if 'success' in v}
    return rows


# ----------------------------------------------------------------------------- model servers
def server(task, kind):
    root = task_root(task); r = root/'models'/kind
    if kind == 'imagewam':
        from imagewam_libero_adapter import Backend
        b = Backend(root); handlers = dict(infer=b.infer, audit=lambda req: b.audit(r))
    else:
        import torch
        torch.set_num_threads(4); torch.cuda.init()
        if kind == 'pi':
            from libero_pi_adapter import Backend
        else:
            from fastwam_adapter import Backend
        from texture_eot import transform
        sys.path.insert(0, str(PIPE))
        from dual_view_proxy import CameraProxy
        b = Backend(r); cal = calibration(task)
        with np.load(cal/'base.npz') as z:
            d = {n: z[n].copy() for n in ['state8', 'clean_base', 'clean_wrist', 'train_mask', 'validation_mask']}
            prompt = str(z['prompt'])
        proxy = [CameraProxy(cal/'base.npz', device='cuda', fit=E26_FIT/'base.json'),
                 CameraProxy(cal/'wrist.npz', device='cuda', fit=E26_FIT/'wrist.json')]
        cache = {}

        def clean_views(k):
            return [torch.tensor(d[n][k].copy(), device='cuda').permute(2, 0, 1)[None].float()/255 for n in ['clean_base', 'clean_wrist']]

        def response(k, seed, images):
            v, _ = b.response(b.prepare(d['state8'][k], prompt, images), seed, full=False)
            return v.float() if kind == 'pi' else v[..., :7].float()

        def batch(req):
            losses, total = [], None
            draws = req.get('draws') or [IDENTITY]*len(req['indices'])
            for k, seed, draw in zip(req['indices'], req['seeds'], draws):
                k, seed = int(k), int(seed)
                if (k, seed) not in cache:
                    with torch.no_grad():
                        cache[(k, seed)] = response(k, seed, clean_views(k)).cpu()
                with torch.set_grad_enabled(req['gradient']):
                    q = torch.tensor(req['texture'], device='cuda', requires_grad=req['gradient'])
                    t = transform(q, draw)
                    delta = response(k, seed, [proxy[0](t, k), proxy[1](t, k)])-cache[(k, seed)].cuda()
                    loss = delta.square().sum() if kind == 'pi' else delta.square().mean()
                    assert torch.isfinite(loss)
                    losses.append(float(loss.detach()))
                    if req['gradient']:
                        g, = torch.autograd.grad(loss, q)
                        assert torch.isfinite(g).all() and float(g.norm()) > 0
                        assert all(p.grad is None for p in b.model.parameters())
                        total = g.cpu().numpy() if total is None else total+g.cpu().numpy()
            out = dict(losses=losses)
            if total is not None:
                out['gradient'] = total/len(losses)
            return out

        def audit(req):
            rows = []
            with torch.no_grad():
                for k in np.flatnonzero(d['validation_mask'])[:2]:
                    prepared = b.prepare(d['state8'][k], prompt, clean_views(k)); _, a = b.response(prepared, 20261201, full=True)
                    if kind == 'pi':
                        native = b.native_full(prepared, 20261201)
                    else:
                        image, proprio, (context, mask) = prepared
                        native = b.model.infer_action(prompt=None, input_image=image, proprio=proprio, context=context, context_mask=mask,
                                                      action_horizon=32, num_inference_steps=10, seed=20261201, rand_device='cpu')['action'][None].float().cpu()
                    err = float((a.float().cpu()-native).abs().max()); assert err == 0, ('native parity', kind, err)
                    rows.append(dict(index=int(k), max_abs=err))
            dump(r/'native_audit.json', dict(passed=True, rows=rows)); return dict(passed=True)

        def trim(req):
            torch.cuda.empty_cache(); return dict(trimmed=True)
        handlers = dict(batch=batch, audit=audit, infer=b.infer, trim=trim,
                        validation_indices=lambda req: dict(indices=[int(k) for k in np.flatnonzero(d['validation_mask'])]))
    with Listener(('127.0.0.1', ports(task)[kind]), authkey=AUTH) as listener:
        dump(r/'ready.json', dict(pid=os.getpid(), host=socket.gethostname()))
        running = True
        while running:
            with listener.accept() as c:
                try:
                    req = c.recv(); cmd = req['command']
                    if cmd == 'shutdown':
                        running = False; out = dict(stopped=True)
                    elif cmd == 'infer':
                        out = handlers['infer'](req)
                    else:
                        out = handlers[cmd](req)
                    c.send(out)
                except Exception as e:
                    traceback.print_exc(); c.send(dict(error=repr(e), traceback=traceback.format_exc()))


def rpc(task, kind, command, **kw):
    with Client(('127.0.0.1', ports(task)[kind]), authkey=AUTH) as c:
        c.send(dict(command=command, **kw)); out = c.recv()
    if 'error' in out:
        raise RuntimeError(kind+': '+out['error']+'\n'+out.get('traceback', ''))
    return out


def both(task, command, **kw):
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        futures = {k: ex.submit(rpc, task, k, command, **kw) for k in KINDS}
        return {k: f.result() for k, f in futures.items()}


def save_png(q, path):
    Image.fromarray(np.rint(q[0].transpose(1, 2, 0)*255).astype(np.uint8)).save(path)


def load_png(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32).transpose(2, 0, 1)[None]/255


# ----------------------------------------------------------------------------- training
def train_specialist(task, kind):
    root = task_root(task); plan = json.loads((root/'training_plan.json').read_text())
    dest = root/'training'/kind; dest.mkdir(parents=True, exist_ok=True); ck = dest/'state.npz'
    q = np.full((1, 3, 128, 128), .5, np.float32); hist = []
    if ck.exists():
        with np.load(ck) as z:
            q = z['texture'].copy(); hist = json.loads(str(z['history']))
    for it in range(len(hist), STEPS):
        out = rpc(task, kind, 'batch', texture=q, indices=plan[it]['indices'], seeds=plan[it]['seeds'], gradient=True)
        hist.append(float(np.mean(out['losses'])))
        q = np.clip(q+LR*np.sign(out['gradient']), 0, 1).astype(np.float32)
        tmp = ck.with_suffix('.tmp.npz'); np.savez_compressed(tmp, texture=q, history=json.dumps(hist)); tmp.replace(ck)
        if (it+1) % 50 == 0:
            log('[specialist]', task, kind, it+1, hist[-1])
    if not (root/'textures'/(kind+'.png')).exists():
        save_png(q, root/'textures'/(kind+'.png'))
    return hist


def train_joint(task, ref):
    root = task_root(task); plan = json.loads((root/'training_plan.json').read_text())
    draws = json.loads((root/'eot_plan.json').read_text())
    dest = root/'training'/'joint'; dest.mkdir(parents=True, exist_ok=True); ck = dest/'state.npz'
    q = np.full((1, 3, 128, 128), .5, np.float32); hist = []
    if ck.exists():
        with np.load(ck) as z:
            q = z['texture'].copy(); hist = json.loads(str(z['history']))
    for it in range(len(hist), STEPS):
        out = both(task, 'batch', texture=q, indices=plan[it]['indices'], seeds=plan[it]['seeds'], draws=draws[it], gradient=True)
        raw = {k: float(np.mean(out[k]['losses'])) for k in KINDS}
        ratio = {k: raw[k]/ref[k][it] for k in KINDS}
        chosen = 'pi' if ratio['pi'] <= ratio['fast'] else 'fast'
        g = {k: out[k]['gradient'] for k in KINDS}
        cos = float(np.sum(g['pi']*g['fast'])/(np.linalg.norm(g['pi'])*np.linalg.norm(g['fast'])))
        q = np.clip(q+LR*np.sign(g[chosen]), 0, 1).astype(np.float32)
        hist.append(dict(iteration=it+1, raw=raw, ratio=ratio, chosen=chosen, cosine=cos))
        tmp = ck.with_suffix('.tmp.npz'); np.savez_compressed(tmp, texture=q, history=json.dumps(hist)); tmp.replace(ck)
        if (it+1) % 50 == 0:
            tail = hist[-50:]
            dump(dest/'progress.json', dict(iteration=it+1, tail_ratio={k: float(np.mean([h['ratio'][k] for h in tail])) for k in KINDS},
                                            chosen={k: sum(h['chosen'] == k for h in hist) for k in KINDS}))
            log('[joint]', task, it+1, {k: round(v, 3) for k, v in ratio.items()})
    if not (root/'textures/joint.png').exists():
        save_png(q, root/'textures/joint.png')


def heldout(task):
    root = task_root(task); path = root/'heldout.json'
    if path.exists():
        return
    idx = rpc(task, 'pi', 'validation_indices')['indices'][:8]
    out = dict(indices=idx, noise_seed=20260901)
    for t in ['random', 'pi', 'fast', 'joint']:
        res = both(task, 'batch', texture=load_png(root/'textures'/(t+'.png')), indices=idx, seeds=[20260901]*len(idx), gradient=False)
        out[t] = {k: float(np.mean(res[k]['losses'])) for k in KINDS}
    dump(path, out)


# ----------------------------------------------------------------------------- GPU allocation
def inventory():
    p = subprocess.run([sys.executable, str(PREFLIGHT), '--json'], capture_output=True, text=True, timeout=120)
    return json.loads(p.stdout)


def card_ok(host, index, uuid, min_free):
    """Immediate re-check on the executing host, right before use."""
    p = subprocess.run([sys.executable, str(PREFLIGHT), '--hosts', host, '--json'], capture_output=True, text=True, timeout=60)
    node = json.loads(p.stdout)['nodes'][0]
    card = next((g for g in node['gpus'] if g['index'] == index), None)
    return bool(node['status'] == 'ok' and card and card['uuid'] == uuid and card['eligible'] and card['free_mib'] >= min_free
                and (host, index) not in EXCLUDED), card


class Job:
    def __init__(self, job):
        self.name = job; self.dir = OUT/'jobs'/job.replace(':', '__'); self.dir.mkdir(parents=True, exist_ok=True)

    def state(self):
        try:
            return json.loads((self.dir/'state.json').read_text())
        except (OSError, ValueError):
            return {}

    def set(self, **kw):
        s = self.state(); s.update(kw, updated=now()); dump(self.dir/'state.json', s)


def env_for(gpu, extra=None):
    e = dict(os.environ, WAMATTACK_PROJECT_ROOT=str(ROOT), PYTHONNOUSERSITE='1', OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1',
             HF_HUB_DISABLE_TELEMETRY='1', JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false',
             CUDA_VISIBLE_DEVICES=gpu['uuid'], MUJOCO_EGL_DEVICE_ID=str(gpu['index']), NUMBA_CACHE_DIR='/tmp/wam-x5-numba')
    # nnmc71 has its own /home with a foreign ~/.libero; always pin the shared LIBERO config.
    shared = OUT/'libero_config'
    if not (shared/'config.yaml').exists():
        shared.mkdir(parents=True, exist_ok=True)
        (shared/'config.yaml').write_text((ROOT/'robotwinprobe/out/libero13_joint_eot_20261029/libero_config/config.yaml').read_text())
    e.update(LIBERO_CONFIG_PATH=str(shared), WAMATTACK_LIBERO_CONFIG_PATH=str(shared))
    e.update(extra or {})
    if 'WAMATTACK_LIBERO_CONFIG_PATH' in (extra or {}):
        e['LIBERO_CONFIG_PATH'] = extra['WAMATTACK_LIBERO_CONFIG_PATH']
    return e


def sim_env(gpu, task=None):
    # Native EGL order != nvidia-smi order (nnmc60-65/71), so resolve the EGL device of the assigned UUID.
    # robosuite asserts MUJOCO_EGL_DEVICE_ID appears in CUDA_VISIBLE_DEVICES; scan/capture/rollout do no CUDA compute.
    probe = json.loads(subprocess.check_output([SIM, str(ROOT/'robotwinprobe/egl_enum.py')], text=True,
                                               env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'])))
    hits = [d['egl'] for d in probe['devices'] if d['uuid'] == gpu['uuid']]
    assert len(hits) == 1, ('EGL device for assigned UUID not unique', gpu, probe)
    extra = dict(CUDA_VISIBLE_DEVICES=str(hits[0]), MUJOCO_EGL_DEVICE_ID=str(hits[0]), OMP_NUM_THREADS='2')
    if task:
        extra['WAMATTACK_LIBERO_CONFIG_PATH'] = str(task_root(task)/'libero_config')
    return env_for(gpu, extra)


def run(cmd, logfile, env, cwd=ROOT, timeout=None):
    with open(logfile, 'a') as f:
        subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def start_servers(task, assign):
    procs = {}
    for kind, gpu in assign.items():
        r = task_root(task)/'models'/kind
        (r/'ready.json').unlink(missing_ok=True)
        f = open(task_root(task)/'logs'/(kind+'_server.log'), 'a')
        procs[kind] = subprocess.Popen([PY[kind], '-u', str(ROOT/'robotwinprobe/libero_crossarch5.py'), 'server', '--task', task, '--kind', kind],
                                       cwd=ROOT/'robotwinprobe', env=env_for(gpu), stdout=f, stderr=subprocess.STDOUT)
    t0 = time.monotonic()
    while not all((task_root(task)/'models'/k/'ready.json').exists() for k in procs):
        if any(p.poll() is not None for p in procs.values()) or time.monotonic()-t0 > 900:
            raise RuntimeError('server start failed: '+task)
        time.sleep(3)
    return procs


def stop_servers(task, procs):
    for kind, p in procs.items():
        if p.poll() is None:
            try:
                rpc(task, kind, 'shutdown')
            except Exception:
                p.terminate()
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()


def rollouts(task, kind, gpu):
    root = task_root(task)/'models'/kind
    for c in CONDITIONS:
        for i in INDICES:
            out = root/'episodes'/c/f'init{i}.json'
            if out.exists() and 'success' in json.loads(out.read_text()):
                continue
            out.unlink(missing_ok=True)
            run([SIM, '-u', str(ROOT/'robotwinprobe/fastwam_libero_trial.py'), 'rollout', '--root', str(root), '--condition', c, '--index', str(i)],
                task_root(task)/'logs'/f'{kind}_{c}_{i}.log', sim_env(gpu, task), timeout=3600)


# ----------------------------------------------------------------------------- jobs
def job_scan(task, gpu):
    out = OUT/'scan'/task
    if out.exists() and not scan_dir(task):
        shutil.move(str(out), str(OUT/'scan'/('invalid_'+task+'_'+str(int(time.time())))))
    run([SIM, '-u', 'visibility_scan.py', '--task', task, '--pool', str(pool_path(task)), '--out', str(out)],
        OUT/'logs'/f'scan_{task}.log', sim_env(gpu), cwd=PIPE, timeout=6*3600)
    assert scan_dir(task), 'scan did not validate: '+task


def job_calibrate(task, gpu):
    sel = selection()['positions'][task]; cal = calibration(task)
    if (cal/'capture.json').exists():
        meta = json.loads((cal/'capture.json').read_text())
        assert meta.get('complete') and np.allclose(meta['cell_world'][:2], sel['world'], atol=1e-7)
        return
    if cal.exists():
        shutil.move(str(cal), str(cal.with_name('calib_interrupted_'+str(int(time.time())))))
    run([SIM, '-u', 'multitask_capture.py', '--task', task, '--port', '0', '--out', str(cal), '--pool-from', str(pool_path(task)),
         '--cell-world=%.4f,%.4f' % tuple(sel['world'])], OUT/'logs'/f'calib_{task}.log', sim_env(gpu), cwd=PIPE, timeout=4*3600)


def job_pipeline(task, gpus):
    """gpus: [card_a] (H100: everything) or [card_a, card_b]: π+ImageWAM on a, FastWAM on b."""
    prepare(task); root = task_root(task)
    a, b = gpus[0], gpus[-1]
    if not (root/'models/fast/prompts.npz').exists():
        run([PY['fast'], '-u', str(ROOT/'robotwinprobe/fastwam_libero_trial.py'), 'prompts', '--root', str(root/'models/fast')],
            root/'logs/prompts_fast.log', env_for(b), cwd=ROOT/'robotwinprobe')
    if not (root/'contexts.pt').exists():
        run([PY['imagewam'], '-u', str(ROOT/'robotwinprobe/imagewam_libero_adapter.py'), 'encode-context', '--root', str(root),
             '--task', json.loads((root/'protocol.json').read_text())['prompt']], root/'logs/prompts_imagewam.log', env_for(a), cwd=ROOT/'robotwinprobe')
    procs = start_servers(task, dict(pi=a, fast=b))
    try:
        both(task, 'audit')
        if not (root/'textures/joint.png').exists():
            with concurrent.futures.ThreadPoolExecutor(2) as ex:
                ref = {k: f.result() for k, f in {k: ex.submit(train_specialist, task, k) for k in KINDS}.items()}
            dump(root/'reference.json', ref)
            train_joint(task, json.loads((root/'reference.json').read_text()))
        heldout(task); both(task, 'trim')
        procs.update(start_servers(task, dict(imagewam=a)))
        rpc(task, 'imagewam', 'audit')
        with concurrent.futures.ThreadPoolExecutor(3) as ex:
            for f in [ex.submit(rollouts, task, k, g) for k, g in [('pi', a), ('fast', b), ('imagewam', a)]]:
                f.result()
    finally:
        stop_servers(task, procs)
    dump(root/'complete.json', dict(complete=True, at=now()))


def job_main(name, host, gpus):
    """Executed on `host`; re-check every assigned card immediately before use."""
    job = Job(name); kind, task = name.split(':')
    need = dict(scan=4000, calibrate=4000, pipeline=30000)[kind]
    for g in gpus:
        ok, card = card_ok(host, g['index'], g['uuid'], need)
        if not ok:
            job.set(stage='blocked', card=card); sys.exit(75)
    job.set(stage='running', host=host, gpus=gpus, pid=os.getpid(), started=now())
    try:
        dict(scan=lambda: job_scan(task, gpus[0]), calibrate=lambda: job_calibrate(task, gpus[0]),
             pipeline=lambda: job_pipeline(task, gpus))[kind]()
        job.set(stage='done', finished=now())
    except BaseException as e:
        job.set(stage='failed', error=repr(e), traceback=traceback.format_exc()); raise


def pending_jobs():
    """Jobs whose prerequisites are met and which are not done/running/failed."""
    out = []
    for t in CANDIDATES:
        if not scan_dir(t):
            out.append(('scan:'+t, 1))
    sel = selection()
    if sel:
        for t in sel['tasks']:
            if not (calibration(t)/'capture.json').exists():
                out.append(('calibrate:'+t, 1))
            elif not (task_root(t)/'complete.json').exists():
                out.append(('pipeline:'+t, 2))
    return out


def alive(host, pid):
    try:
        p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, f'kill -0 {pid}'], timeout=30, capture_output=True)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return True  # unknown: assume alive, never double-launch


def launch(name, host, gpus):
    job = Job(name)
    job.set(stage='launching', host=host, gpus=gpus, launched=now())
    cmd = (f'cd {ROOT}/robotwinprobe && nohup setsid {PY["pi"]} -u {ROOT}/robotwinprobe/libero_crossarch5.py job --name {name} --host {host} '
           f"--gpus '{json.dumps(gpus)}' >> {job.dir}/job.log 2>&1 < /dev/null & echo $!")
    try:
        p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, cmd], capture_output=True, text=True, timeout=60)
        job.set(launcher_pid=int(p.stdout.strip().split()[-1]))
    except Exception as e:
        job.set(launch_note=repr(e))  # the job itself records its pid once running
    log('[launch]', name, host, [g['index'] for g in gpus])


def scheduler():
    lock = open(OUT/'scheduler.lock', 'w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dump(OUT/'scheduler.json', dict(pid=os.getpid(), host=socket.gethostname(), started=now()))
    while True:
        jobs = pending_jobs()
        report()
        if not jobs and selection():
            log('[scheduler] all done'); return
        running, waiting = [], []
        for name, ncard in jobs:
            s = Job(name).state(); st = s.get('stage')
            if st == 'failed':
                continue
            if st in ('launching', 'running'):
                pid = s.get('pid') or s.get('launcher_pid')
                launched = s.get('launched', now())
                age = (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(launched)).total_seconds()
                if pid and not alive(s['host'], pid) and age > 120:
                    Job(name).set(stage='failed', error='process vanished')
                    continue
                if st == 'launching' and not pid and age > 600:
                    Job(name).set(stage='failed', error='never started')
                    continue
                running.append(s); continue
            waiting.append((name, ncard))
        busy = {(s['host'], g['index']) for s in running for g in s.get('gpus', [])}
        if waiting and len(running) < MAX_RUNNING:
            try:
                inv = inventory()
            except Exception as e:
                log('[scheduler] preflight failed', repr(e)); inv = dict(nodes=[])
            free = {}
            for node in inv['nodes']:
                if node['status'] != 'ok':
                    continue
                for g in node['gpus']:
                    if g['eligible'] and node['host'] in ALLOWED_HOSTS and (node['host'] != 'nnmc71' or (node['host'], g['index']) in H100_ALLOWED) and (node['host'], g['index']) not in EXCLUDED and (node['host'], g['index']) not in busy:
                        free.setdefault(node['host'], []).append(dict(index=g['index'], uuid=g['uuid'], name=g['name'], free_mib=g['free_mib']))
            dump(OUT/'last_inventory.json', dict(at=inv.get('checked_at'), free=free, running=len(running)))
            for name, ncard in waiting:
                if len(running) >= MAX_RUNNING:
                    break
                pick = None
                for host, cards in sorted(free.items(), key=lambda kv: -len(kv[1])):
                    if not name.startswith('pipeline:'):
                        cards = [c for c in cards if 'H100' not in c['name']]
                    h100 = [c for c in cards if 'H100' in c['name']]
                    if ncard == 2 and h100:
                        pick = (host, [h100[0]])
                    elif len(cards) >= ncard:
                        pick = (host, cards[:ncard])
                    if pick:
                        break
                if not pick:
                    continue
                host, gpus = pick
                for g in gpus:
                    free[host].remove(g)
                launch(name, host, gpus); running.append(dict(host=host, gpus=gpus))
        time.sleep(180)


# ----------------------------------------------------------------------------- report
def report():
    sel = selection() if (OUT/'selection.json').exists() else None
    L = ['# LIBERO 5 任务：π0.5+FastWAM 联训（最大最小+EOT）贴纸能否迁移到 ImageWAM', '',
         f'更新：{now()}。', '',
         '每个任务的做法与第 51 号（碗放盘子）相同，只换任务和贴纸位置：13cm 贴纸，位置按预先固定的规则挑选（主相机 90% 训练帧可见 ≥300 像素的位置中，取腕部相机平均可见像素最多的）；'
         '先各自单独训 π0.5 和 FastWAM 的贴纸得到参照曲线，再按最大最小规则加 EOT 联训一张贴纸。'
         'ImageWAM 不参与训练和挑选，是零样本迁移目标。闭环用初始状态 30–49（20 个），各模型用各自原生协议；'
         'ASR = 干净场景成功且贴纸后失败的次数 / 该模型自己干净场景成功次数。', '']
    if not sel:
        done = [t for t in CANDIDATES if scan_dir(t)]
        L += [f'阶段：可见位置扫描 {len(done)}/{len(CANDIDATES)} 个任务完成（{", ".join(done)}）。', '']
    else:
        L += ['## 任务与贴纸位置', '', '| 任务 | 入选 | 位置 (x, y) | 腕部平均可见像素 | 主相机平均可见像素 |', '|---|---|---|---:|---:|']
        for t in CANDIDATES:
            p = sel['positions'][t]
            L.append(f'| {t} | {"是" if t in sel["tasks"] else "否"} | ({p["world"][0]:.2f}, {p["world"][1]:.2f}) | {p["wrist_mean"]:.0f} | {p["base_mean"]:.0f} |')
        L += ['', '## 闭环结果', '', '| 任务 | 模型 | 干净成功 | 随机纹理 ASR | 联训贴纸 ASR |', '|---|---|---:|---:|---:|']
        for t in sel['tasks']:
            for kind in MODELS:
                rows = outcomes(t, kind) if task_root(t).exists() else {}
                clean = rows.get('clean', {}); succ = {i for i, v in clean.items() if v}

                def asr(c):
                    d = rows.get(c, {}); pair = succ & d.keys()
                    return '—' if not pair else f'{sum(not d[i] for i in pair)}/{len(pair)}' + ('' if len(d) == len(INDICES) else f'（进行中 {len(d)}/20）')
                L.append(f'| {t} | {dict(pi="π0.5", fast="FastWAM", imagewam="ImageWAM")[kind]} | '
                         f'{sum(clean.values()) if clean else "—"}/{len(clean) if clean else 20} | {asr("random")} | {asr("joint")} |')
    L += ['', '## 作业状态', '', '| 作业 | 状态 | 节点 / 卡 |', '|---|---|---|']
    for d in sorted((OUT/'jobs').glob('*')) if (OUT/'jobs').exists() else []:
        s = Job(d.name.replace('__', ':')).state()
        L.append(f'| {d.name.replace("__", ":")} | {s.get("stage")} | {s.get("host", "")} {[g["index"] for g in s.get("gpus", [])]} |')
    L += ['', f'原始目录：`{OUT}`。']
    REPORT.write_text('\n'.join(L)+'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['run', 'job', 'server', 'report', 'selection'])
    ap.add_argument('--task'); ap.add_argument('--kind'); ap.add_argument('--name'); ap.add_argument('--host'); ap.add_argument('--gpus')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); (OUT/'logs').mkdir(exist_ok=True)
    if a.command == 'run':
        scheduler()
    elif a.command == 'job':
        job_main(a.name, a.host, json.loads(a.gpus))
    elif a.command == 'server':
        server(a.task, a.kind)
    elif a.command == 'selection':
        print(json.dumps(selection(), indent=1))
    else:
        report()
