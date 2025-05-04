import torch as th


def masked_root_mean_squared_error(
    x: th.Tensor,
    y: th.Tensor,
    mask: th.BoolTensor = None,
):
    """Compute the Root Mean Squared Error (RMSE) between two sets of samples x and y.
    If mask is given, then only samples with `mask=True` is considered.

    Args:
        x (Tensor): A PyTorch tensor of shape (n, d), where n is the number of samples in x and d is the dimension.
        y (Tensor): A PyTorch tensor of shape (n, d), where n is the number of samples in y and d is the dimension.
        mask (BoolTensor, optional): A PyTorch tensor of shape (n, d), where n is the number of samples in x and y, and d is the dimension.

    Returns:
        float: The RMSE value between x and y.
    """
    if mask is None:
        mask = th.ones_like(x).bool()
    error = (x - y)**2
    mask = mask.float()
    denom = th.sum(mask)
    masked_mean = th.sum(error * mask) / denom
    return th.sqrt(masked_mean)
