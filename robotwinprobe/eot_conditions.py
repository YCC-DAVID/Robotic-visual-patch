"""Finite physical EOT distribution; shared world placement and scene lighting."""
import numpy as np

POSITIONS = {
    'origin': [0., 0.], 'xm': [-.02, 0.], 'xp': [.02, 0.],
    'ym': [0., -.02], 'yp': [0., .02],
}
LIGHTS = {
    'nominal': [1., 1., 1.],
    'dim_warm': [.756, .7, .644],
    'bright_cool': [1.196, 1.3, 1.404],
}


def conditions():
    return {p+'__'+l: {'position': p, 'light': l, 'center': [*xy, .741],
                       'light_gain': gain, 'size_m': .20}
            for p, xy in POSITIONS.items() for l, gain in LIGHTS.items()}


def allowed(mode):
    return [k for k, c in conditions().items()
            if (mode in ('position', 'joint') or c['position'] == 'origin')
            and (mode in ('light', 'joint') or c['light'] == 'nominal')]


class Lighting:
    """Change actual SAPIEN light colors, relative to a captured nominal scene."""
    def __init__(self, task):
        self.scene = task.scene
        self.ambient = np.array(task.scene.ambient_light).copy()
        self.lights = list(task.direction_light_lst) + list(task.point_light_lst)
        self.colors = [np.array(light.color).copy() for light in self.lights]

    def apply(self, gain):
        gain = np.asarray(gain, np.float32)
        assert gain.shape == (3,) and np.all(gain > 0)
        self.scene.set_ambient_light(self.ambient * gain)
        for light, color in zip(self.lights, self.colors):
            light.set_color(color * gain)


def evaluation_conditions():
    # All nonnominal positions and gains below are absent from the training bank.
    return {
        'nominal': {'center': [0., 0., .741], 'light_gain': [1., 1., 1.]},
        'position_near': {'center': [.01, .01, .741], 'light_gain': [1., 1., 1.]},
        'position_far': {'center': [-.03, -.03, .741], 'light_gain': [1., 1., 1.]},
        'light_dim': {'center': [0., 0., .741], 'light_gain': [.85, .85, .85]},
        'light_bright': {'center': [0., 0., .741], 'light_gain': [1.15, 1.15, 1.15]},
        'joint_near': {'center': [.01, -.01, .741], 'light_gain': [.85, .85, .85]},
        'joint_far': {'center': [-.03, .03, .741], 'light_gain': [1.15, 1.15, 1.15]},
    }
