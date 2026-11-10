"""Shared physical rectangle and camera rays for RoboTwin render transfer."""
import numpy as np
import bootstrap as B

CAMERAS = ('head_camera', 'left_camera', 'right_camera')
WIRE = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
MODEL = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')


def camera_objects(task):
    c = task.cameras
    return dict(zip(c.static_camera_name, c.static_camera_list)) | {
        'left_camera': c.left_camera, 'right_camera': c.right_camera}


class Patch:
    def __init__(self, task, dest, center=(0., 0., .741), size=.20):
        import sapien
        from scene_smoke import add_patch
        self.actor = add_patch(task, dest, center, size)
        self.body = self.actor.find_component_by_type(sapien.render.RenderBodyComponent)
        self.material = self.body.render_shapes[0].material
        self.center, self.size = np.array(center), size
        self.task, self.dest = task, dest
        self.textures = []

    def texture(self, rgb):
        import sapien
        from PIL import Image
        # SAPIEN RT caches bound material textures; rebind through a fresh
        # visual actor, rather than silently retaining the first texture.
        self.task.scene.remove_actor(self.actor)
        rgba = np.concatenate([np.asarray(rgb, np.uint8), np.full((*rgb.shape[:2], 1), 255, np.uint8)], -1)
        tex = sapien.render.RenderTexture2D(rgba, 'R8G8B8A8Unorm', mipmap_levels=1,
                                          filter_mode='linear', address_mode='repeat', srgb=True)
        self.material = sapien.render.RenderMaterial(base_color=[1,1,1,1], roughness=1, metallic=0)
        self.material.set_base_color_texture(tex)
        builder = self.task.scene.create_actor_builder()
        builder.add_visual_from_file(str(self.dest/'patch.obj'), material=self.material)
        builder.set_initial_pose(sapien.Pose(self.center))
        self.actor = builder.build_static(name='wam_visual_patch')
        self.body = self.actor.find_component_by_type(sapien.render.RenderBodyComponent)
        self.textures = [tex]

    def visible(self, value):
        self.body.visibility = float(value)

    def geometry(self, task):
        result = {}
        for name, camera in camera_objects(task).items():
            if name not in CAMERAS: continue
            k, e = camera.get_intrinsic_matrix(), camera.get_extrinsic_matrix()
            h, w = 240, 320
            yy, xx = np.indices((h, w))
            # CV intrinsics map integer pixel centres, as used by SAPIEN.
            rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(k).T
            e4 = np.eye(4); e4[:3] = e[:3]
            pose = np.linalg.inv(e4)
            direction = rays @ pose[:3, :3].T
            origin = pose[:3, 3]
            t = (self.center[2]-origin[2]) / np.where(abs(direction[..., 2])>1e-8, direction[..., 2], 1e-8)
            world = origin + t[..., None]*direction
            grid = np.stack([2*(world[..., 0]-self.center[0])/self.size,
                             -2*(world[..., 1]-self.center[1])/self.size], -1)
            seg = camera.get_picture('Segmentation')
            mask = seg[..., 1] == self.actor.per_scene_id
            result[name] = dict(grid=grid.astype(np.float32), mask=mask, intrinsic=k, extrinsic=e,
                                valid=(t>0) & (abs(grid).max(-1)<=1))
        return result


def patterns():
    yy, xx = np.indices((128, 128))
    colors = np.array([[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255]], np.uint8)
    return {'black': np.zeros((128,128,3), np.uint8), 'white': np.full((128,128,3),255,np.uint8),
            'gray': np.full((128,128,3),128,np.uint8),
            'uv': np.stack([xx*2, yy*2, np.full_like(xx,128)], -1).astype(np.uint8),
            'checker': colors[(xx//16+2*(yy//16))%6],
            'random': np.random.RandomState(20261007).randint(0,256,(128,128,3), dtype=np.uint8)}


def main():
    import argparse, json, random
    B.simulator_paths()
    from scene_smoke import make_task
    from PIL import Image
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    args = ap.parse_args(); dest = B.Path(args.out).resolve(); dest.mkdir(parents=True, exist_ok=True)
    random.seed(10000)
    task, _, _ = make_task('place_container_plate', 10000, dest)
    try:
        clean = task.get_obs()
        patch = Patch(task, dest)
        data = {}; report = {}
        for label, rgb in patterns().items():
            patch.texture(rgb); obs = task.get_obs()
            for name in CAMERAS:
                image = obs['observation'][name]['rgb']
                data[name+'_'+label] = image
                Image.fromarray(image).save(dest/(name+'_'+label+'.png'))
        geo = patch.geometry(task)
        patch.texture(patterns()['black']); repeat = task.get_obs()
        patch.visible(False); hidden = task.get_obs()
        for name in CAMERAS:
            data[name+'_clean'] = clean['observation'][name]['rgb']
            for key, value in geo[name].items(): data[name+'_'+key] = value
            report[name] = {'mask_pixels': int(geo[name]['mask'].sum()),
                'mask_outside_quad': int((geo[name]['mask'] & ~geo[name]['valid']).sum()),
                'black_repeat_mae': float(abs(repeat['observation'][name]['rgb'].astype(float)-data[name+'_black']).mean()),
                'hidden_clean_mae': float(abs(hidden['observation'][name]['rgb'].astype(float)-data[name+'_clean']).mean())}
        np.savez_compressed(dest/'capture.npz', **data)
        (dest/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        print(report, flush=True)
    finally: task.close_env()


if __name__ == '__main__': main()
