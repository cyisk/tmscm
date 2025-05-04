import torch as th
from torch.distributions import Normal

from dataset.tmscm import TMSCM


def fx(ux: th.Tensor):
    x = th.zeros_like(ux)
    x[..., 0] = ux[..., 0]
    x[..., 1] = ux[..., 0] + 0.25 * ux[..., 1]
    return x


def fy(uy: th.Tensor):
    y = th.zeros_like(uy)
    y[..., 0] = uy[..., 0]
    y[..., 1] = -0.5 * uy[..., 0] + uy[..., 1]
    return y


def fz(x: th.Tensor, y: th.Tensor, uz: th.Tensor):
    z = th.zeros_like(uz)
    denom0 = (1 + th.exp(-x[..., 0] - y[..., 1]))
    denom1 = (1 + th.exp(-y[..., 1] - x[..., 0]))
    z[..., 0] = 1 / denom0 - y[..., 0]**2 + 0.5 * uz[..., 0]
    z[..., 1] = 1 / denom1 - x[..., 1]**2 + 0.5 * (uz[..., 0] + uz[..., 1])
    return z


def fw(z: th.Tensor, uw: th.Tensor):
    w = th.zeros_like(uw)
    denom0 = (1 + th.exp(0.5 * z[..., 0]**2 - z[..., 1]))
    denom1 = (1 + th.exp(0.5 * z[..., 1]**2 - z[..., 0]))
    w[..., 0] = 20 / denom0 + uw[..., 0]
    w[..., 1] = 20 / denom1 + 0.25 * uw[..., 0] - uw[..., 1]
    return w


fork_tmscm = TMSCM(
    dependencies={
        'x': [],
        'y': [],
        'z': ['x', 'y'],
        'w': ['z'],
    },
    mechanisms={
        'x': fx,
        'y': fy,
        'z': fz,
        'w': fw,
    },
    exogenous_distributions={
        'ux': Normal(th.zeros((2, )), th.ones((2, ))),
        'uy': Normal(th.zeros((2, )), th.ones((2, ))),
        'uz': Normal(th.zeros((2, )), th.ones((2, ))),
        'uw': Normal(th.zeros((2, )), th.ones((2, ))),
    },
    endo_exo_index_pairs={
        ('x', 'ux'),
        ('y', 'uy'),
        ('z', 'uz'),
        ('w', 'uw'),
    },
)
