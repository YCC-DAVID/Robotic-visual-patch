"""Early flow direction objectives, maximized by the shared-Q trainer.

The vector is v_adv itself versus v_clean, flattened over 10 x 32 channels.
It is NOT delta-v versus v_clean, nor a Cartesian robot velocity.
"""
import torch


def objective(velocity, clean, mode):
    distance = (velocity - clean).square().sum()
    a, c = velocity.reshape(-1), clean.reshape(-1)
    na, nc = a.norm(), c.norm()
    # Stop instead of crediting a collapsed vector as a successful right angle.
    assert float(nc.detach()) > 1e-8, 'clean direction undefined'
    assert float(na.detach() / nc.detach()) > 1e-3, 'adversarial direction collapsed'
    cosine = (a * c).sum() / (na * nc)
    if mode == 'distance':
        value = distance
    elif mode == 'orthogonal':
        value = 1 - cosine.square()
    elif mode == 'reverse':
        value = -cosine
    else:
        raise ValueError(mode)
    metrics = {'distance': float(distance.detach()), 'cosine': float(cosine.detach()),
               'norm_ratio': float((na / nc).detach())}
    return value, metrics


def check():
    c = torch.tensor([1., 0.])
    for mode, expected in [('distance', [0., 2., 4.]),
                           ('orthogonal', [0., 1., 0.]),
                           ('reverse', [-1., 0., 1.])]:
        actual = [float(objective(torch.tensor(a), c, mode)[0])
                  for a in [[1., 0.], [0., 1.], [-1., 0.]]]
        assert actual == expected, (mode, actual)
    for mode in ['orthogonal', 'reverse']:
        a = torch.tensor([1., .3], requires_grad=True)
        objective(a, c, mode)[0].backward()
        assert a.grad[0] < 0 and a.grad[1] > 0
    try:
        objective(torch.zeros(2), c, 'orthogonal')
    except AssertionError:
        pass
    else:
        raise AssertionError('zero vector was accepted')
    print('direction objective checks passed')


if __name__ == '__main__':
    check()
