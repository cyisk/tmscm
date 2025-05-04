import torch as th
from typing import Callable


# Get safely masked samples with 1-operands
def safe_masked_1(x: th.Tensor, dim=None):
    mask_x = ~(th.isnan(x) | th.isinf(x) | (x.abs() > 1e+7)).any(dim=dim)
    mask = mask_x
    return x[mask]


def safe_mean(x: th.Tensor, dim=None):
    """Return the safe mean of a Tensor, preventing potential gradient explosion and nan.

    Returns:
        Tensor: Safe mean.
    """
    x = safe_masked_1(x, dim=dim)
    if x.nelement() == 0:
        return th.zeros((1, ))[0].to(x.device).requires_grad_(True)
    else:
        return x.mean()


# Get safely masked samples with 2-operands
def safe_masked_2(x: th.Tensor, y: th.Tensor, dim=None):
    mask_x = ~(th.isnan(x) | th.isinf(x) | (x.abs() > 1e+7)).any(dim=dim)
    mask_y = ~(th.isnan(y) | th.isinf(y) | (y.abs() > 1e+7)).any(dim=dim)
    mask = mask_x & mask_y
    return x[mask], y[mask_y]


def safe_metric(metric: Callable, x: th.Tensor, y: th.Tensor, dim=None):
    """Return the safe metric value between 2 Tensors, preventing potential gradient explosion and nan.

    Returns:
        Tensor: Safe metric value.
    """
    x, y = safe_masked_2(x, y, dim)
    if x.nelement() == 0:
        return th.inf
    else:
        return metric(x, y)
