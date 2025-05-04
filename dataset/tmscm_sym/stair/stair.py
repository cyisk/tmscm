import torch as th
from torch.distributions import Normal

from dataset.tmscm import TMSCM


def fx(ux: th.Tensor):
    x = th.zeros_like(ux)
    x = 0.75 * ux
    return x


def fy(x: th.Tensor, uy: th.Tensor):
    y = th.zeros_like(uy)
    y[..., 0] = 10 * x[..., 0] + uy[..., 0]
    y[..., 1] = 0.25 * x[..., 0] - 0.5 * uy[..., 1] + 2.
    return y


def fz(y: th.Tensor, uz: th.Tensor):
    z = th.zeros_like(uz)
    z[..., 0] = -0.5 * y[..., 0] + uz[..., 0] - 4.
    z[..., 1] = 5 * y[..., 1] - 1.5 * uz[..., 1]
    z[..., 2] = y[..., 1] + 2 * uz[..., 2] - 0.5
    return z


stair_tmscm = TMSCM(
    dependencies={
        'x': [],
        'y': ['x'],
        'z': ['y'],
    },
    mechanisms={
        'x': fx,
        'y': fy,
        'z': fz,
    },
    exogenous_distributions={
        'ux': Normal(th.zeros((1, )), th.ones((1, ))),
        'uy': Normal(th.zeros((2, )), th.ones((2, ))),
        'uz': Normal(th.zeros((3, )), th.ones((3, ))),
    },
    endo_exo_index_pairs={
        ('x', 'ux'),
        ('y', 'uy'),
        ('z', 'uz'),
    },
)
