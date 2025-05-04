import math
import torch as th
from torch.distributions import Uniform

from dataset.tmscm import TMSCM

dim = 8


def s(x):
    if isinstance(x, th.Tensor):
        return th.log(1 + th.exp(x))
    else:
        return math.log(1 + math.exp(x))


def L(x, y):
    return s(x + 1) + s(0.5 + y) - 3


def cdf_1(mu, beta, x):
    return mu - beta * th.sign(x - 0.5) * math.sqrt(2) * th.erfinv(
        2 * th.abs(x - 0.5))


def fx(ux: th.Tensor):
    x = th.zeros_like(ux)
    for i in range(0, dim):
        if i == 0:
            x[..., i] = s(1.8 * ux[..., i]) - 1
        if 0 < i <= dim // 2:
            s1 = th.sum(x[..., :i], dim=-1)
            x[..., i] = L(s1, ux[..., i])
        if dim // 2 < i <= dim - dim // 4:
            s1 = th.sum(x[..., :i], dim=-1)
            x[..., i] = 0.3 * ux[..., i] + s(s1 + 1) - 1
        if dim - dim // 4 < i:
            s1 = th.sum(x[..., :i - 1], dim=-1)
            s2 = th.sum(x[..., :i - 2], dim=-1)
            x[..., i] = cdf_1(-s((s2 * 1.3 + s1) / 3 + 1) + 2, 0.6, ux[..., i])
    return x


def fy(x: th.Tensor, uy: th.Tensor):
    y = th.zeros_like(uy)
    for i in range(0, dim):
        if i == 0:
            y[..., i] = L(s(1.8 * uy[..., i]) - 1, x[..., i])
        if 0 < i <= dim // 2:
            s1 = th.sum(y[..., :i], dim=-1)
            y[..., i] = L(L(s1, uy[..., i]), x[..., i])
        if dim // 2 < i <= dim - dim // 4:
            s1 = th.sum(y[..., :i], dim=-1)
            s2 = L(s1, x[..., i])
            y[..., i] = cdf_1(-s((s2 * 1.3 + s1) / 3 + 1) + 2, 0.4, uy[..., i])
        if dim - dim // 4 < i:
            s1 = th.sum(y[..., :i - 1], dim=-1)
            s2 = th.sum(y[..., :i - 2], dim=-1)
            y[..., i] = 0.3 * uy[..., i] - 0.5 * s1 + s(s2 + 1) - 1

    return y


barbell_tmscm = TMSCM(
    dependencies={
        'x': [],
        'y': ['x'],
    },
    mechanisms={
        'x': fx,
        'y': fy,
    },
    exogenous_distributions={
        'ux': Uniform(th.zeros((dim, )), th.ones((dim, ))),
        'uy': Uniform(th.zeros((dim, )), th.ones((dim, ))),
    },
    endo_exo_index_pairs={
        ('x', 'ux'),
        ('y', 'uy'),
    },
)
