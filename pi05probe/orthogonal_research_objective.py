"""E30: directly target v_patch perpendicular to v_clean, never delta-v."""
import torch
from direction_objective import objective as original_objective


def make_target(initial_velocity, clean):
    # Fix one perpendicular direction per state before optimization. Use the
    # warm-start's transverse direction; preserve the clean vector's length.
    a, c = initial_velocity.detach().double(), clean.detach().double()
    transverse = a - (a*c).sum()/c.square().sum()*c
    assert float(transverse.norm()/c.norm()) > 1e-6
    return (transverse/transverse.norm()*c.norm()).to(clean.dtype).detach()


def objective(velocity, clean, mode, target=None):
    if mode in ['distance', 'orthogonal']:
        return original_objective(velocity, clean, mode)
    assert mode == 'target' and target is not None
    _, metrics = original_objective(velocity, clean, 'orthogonal')
    error = (velocity-target).square().sum()/clean.square().sum()
    metrics['target_normalized_squared_error'] = float(error.detach())
    return -error, metrics


def check():
    c = torch.tensor([1., 0., .1], dtype=torch.float64)
    a = torch.tensor([1., .2, .1], dtype=torch.float64, requires_grad=True)
    target = make_target(a, c)
    assert abs(float((target*c).sum())) < 1e-12
    assert torch.allclose(target.norm(), c.norm())
    assert torch.autograd.gradcheck(lambda x: objective(x,c,'target',target)[0], (a,))
    objective(a,c,'target',target)[0].backward()
    assert a.grad[0] < 0 and a.grad[1] > 0
    assert abs(float(objective(target,c,'target',target)[0])) < 1e-12
    print('Orthogonal target construction, norm and gradients verified')


if __name__ == '__main__':
    check()
