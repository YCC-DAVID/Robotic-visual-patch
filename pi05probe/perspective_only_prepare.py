"""Audit camera geometry across the existing 13cm training/validation pool."""
import json
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-perspective-numba')
import drift_eval_digital as D
import numpy as np
from perspective_geometry import box_top_projection


def main():
    import yaml
    import scene_patch as sp
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    dest = D.OUT/'perspective_only'
    dest.mkdir(exist_ok=True)
    source = D.OUT/'render_proxy_scale/cal_13_101/calibration.npz'
    z = np.load(source, allow_pickle=False)
    cfg = yaml.safe_load((D.PROBE/'config/scene.yaml').read_text())
    cfg['patch']['size_wh'] = [float(z['patch_m'])]*2
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if D.pathlib.Path(suite.get_task(i).bddl_file).stem == D.TASK)
    env = OffScreenRenderEnv(bddl_file_name=str(D.pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                             camera_heights=256, camera_widths=256)
    env.env.set_xml_processor(sp.make_xml_processor(cfg, z['cell_world'],
        str(D.OUT/'render_proxy_scale/cal_13_101/random_texture.png')))
    Hs = []
    previous = None
    try:
        for k, state in enumerate(z['sim_states']):
            seed = int(z['env_seed'][k])
            if seed != previous:
                env.seed(seed); env.reset(); previous = seed
            env.regenerate_obs_from_state(state)
            H, geometry = box_top_projection(env.env.sim, cfg['patch']['name'], size=224,
                                             permutation=(3, 2, 1, 0))
            Hs.append(H/H[2, 2])
        difference = float(np.max(np.abs(np.array(Hs)-Hs[0])))
        assert difference < 1e-6, difference
    finally:
        env.close()
    report = {'complete': True, 'all_camera_projections_constant': True, 'frames_checked': len(Hs),
              'max_H_element_change': difference, 'Q128_to_image224_H': Hs[0].tolist(),
              'geometry': geometry, 'uv_to_world_corner_permutation': [3, 2, 1, 0],
              'notes': 'Homography derived from camera fovy/extrinsics and actual box top z=.902. '
                       'UV flip convention identified by independent existing ramp; no continuous H fit. '
                       'All 187 train/validation camera projections checked. No optical fit.'}
    (dest/'geometry_13_101.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
