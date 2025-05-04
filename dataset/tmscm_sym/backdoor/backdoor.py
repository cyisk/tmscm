import torch as th
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from dataset.tmscm import TMSCM


def fx(ux: th.Tensor):
    x = th.zeros_like(ux)
    x[..., 0] = F.tanh(ux[..., 0])
    x[..., 1] = F.sigmoid(-ux[..., 1])
    x[..., 2] = x[..., 1] - F.tanh(ux[..., 2])
    x[..., 3] = 1 + F.sigmoid(ux[..., 3])
    return x


def fy(x: th.Tensor, uy: th.Tensor):
    y = th.zeros_like(uy)
    y[..., 0] = F.tanh(x[..., 0] + uy[..., 0])
    y[..., 1] = F.sigmoid(x[..., 1] - uy[..., 1])
    y[..., 2] = y[..., 1] - F.tanh(uy[..., 2])
    alpha = F.softmax(y[..., :3], dim=-1)
    y[..., 3] = (alpha * x[..., :3]).sum(dim=-1) + F.sigmoid(uy[..., 3])
    return y


def fz(y: th.Tensor, uz: th.Tensor):
    z = th.zeros_like(uz)
    z[..., 0] = F.elu(y[..., 0] - uz[..., 0])
    z[..., 1] = F.leaky_relu(y[..., 1] + uz[..., 1])
    z[..., 2] = z[..., 1] + F.elu(uz[..., 2])
    alpha = F.softmin(z[..., :3], dim=-1)
    z[..., 3] = (alpha * y[..., :3]).sum(dim=-1) + F.leaky_relu(uz[..., 3])
    return z


def fw(x: th.Tensor, y: th.Tensor, z: th.Tensor, uw: th.Tensor):
    w = th.zeros_like(uw)
    beta1 = th.cat([x[..., :1], y[..., :1], z[..., :1]], dim=-1)
    alpha1 = F.softmax(beta1, dim=-1)
    w[..., 0] = (alpha1 * beta1).sum(dim=-1)
    w[..., 0] += F.elu(z[..., 0] - uw[..., 0])

    beta2 = th.cat([w[..., :1], uw[..., :1]], dim=-1)
    alpha2 = F.softmin(beta2, dim=-1)
    w[..., 1] = (alpha2 * beta2).sum(dim=-1)
    w[..., 1] += F.leaky_relu(y[..., 1] + uw[..., 1])

    beta3 = th.cat([x[..., :3], y[..., :3], z[..., :3]], dim=-1)
    alpha3 = F.softmin(beta3, dim=-1)
    w[..., 2] = (alpha3 * beta3).sum(dim=-1)
    w[..., 2] += x[..., 1] + F.elu(uw[..., 2])

    beta4 = th.cat([w[..., :3], uw[..., :3]], dim=-1)
    alpha4 = F.softmax(beta4, dim=-1)
    w[..., 3] = (alpha4 * beta4).sum(dim=-1)
    w[..., 3] += F.leaky_relu(uw[..., 3])
    return w


Cov = th.tensor([
    [1, 0.8, 0.3, 0.2],
    [0.8, 1, 0.4, 0.1],
    [0.3, 0.4, 1, 0.6],
    [0.2, 0.1, 0.6, 1],
])

backdoor_tmscm = TMSCM(
    dependencies={
        'x': [],
        'y': ['x'],
        'z': ['y'],
        'w': ['x', 'y', 'z'],
    },
    mechanisms={
        'x': fx,
        'y': fy,
        'z': fz,
        'w': fw,
    },
    exogenous_distributions={
        'ux': MultivariateNormal(th.zeros((4, )), Cov),
        'uy': MultivariateNormal(th.zeros((4, )), Cov),
        'uz': MultivariateNormal(th.zeros((4, )), Cov),
        'uw': MultivariateNormal(th.zeros((4, )), Cov),
    },
    endo_exo_index_pairs={
        ('x', 'ux'),
        ('y', 'uy'),
        ('z', 'uz'),
        ('w', 'uw'),
    },
)
