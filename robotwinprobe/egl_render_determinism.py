"""Render-determinism test on one assigned GPU (correct native EGL index).
Same LIBERO state rendered N times (RGB agentview/wrist + segmentation); also compares against the reset observation.
Env must set CUDA_VISIBLE_DEVICES=MUJOCO_EGL_DEVICE_ID=<native EGL index of the assigned card>."""
import os, sys, json, socket, pathlib, time
import numpy as np
sys.path.insert(0, '/shared/user65/workspace/chence/WAMattack/pi05probe'); sys.path.insert(0, '/shared/user65/workspace/chence/WAMattack/third_party/openpi/third_party/libero')
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite import macros
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
from OpenGL import GL

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
N = 20
suite = benchmark.get_benchmark_dict()['libero_goal']()
task = next(suite.get_task(i) for i in range(suite.n_tasks) if pathlib.Path(suite.get_task(i).bddl_file).stem == 'put_the_bowl_on_the_plate')
bddl = pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
env.seed(0); obs = env.reset()
sim = env.env.sim; ctx = sim._render_context_offscreen; ctx.gl_ctx.make_current()
res = dict(host=socket.gethostname(), egl=os.environ.get('MUJOCO_EGL_DEVICE_ID'),
           renderer=GL.glGetString(GL.GL_RENDERER).decode(), version=GL.glGetString(GL.GL_VERSION).decode(), cams={})
flip = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
for cam in ['agentview', 'robot0_eye_in_hand']:
    cid = sim.model.camera_name2id(cam); outs = []
    for _ in range(N):
        ctx.render(256, 256, camera_id=cid); outs.append(ctx.read_pixels(256, 256)[::flip].copy())
    o = obs[cam + '_image']
    diffs = [int((np.abs(r.astype(int) - o.astype(int)).sum(-1) > 0).sum()) for r in outs]
    pair = [int((np.abs(outs[i].astype(int) - outs[0].astype(int)).sum(-1) > 0).sum()) for i in range(1, N)]
    res['cams'][cam] = dict(px_diff_vs_obs=diffs, px_diff_vs_first=pair)
    np.save(OUT / f'{res["host"]}_egl{res["egl"]}_{cam}_render0.npy', outs[0])
(OUT / f'{res["host"]}_egl{res["egl"]}.json').write_text(json.dumps(dict(res, segmentation='pending (process may abort)'), indent=1))
try:
    seg = []
    for _ in range(5):
        ctx.render(256, 256, camera_id=sim.model.camera_name2id('agentview'), segmentation=True)
        seg.append(ctx.read_pixels(256, 256, segmentation=True)[::flip].copy())
    res['segmentation_px_diff_vs_first'] = [int((np.abs(s.astype(int) - seg[0].astype(int)).sum(-1) > 0).sum()) for s in seg[1:]]
except Exception as e:
    res['segmentation_error'] = repr(e)
res['time'] = time.time()
(OUT / f'{res["host"]}_egl{res["egl"]}.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res))
os._exit(0)
