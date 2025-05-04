from torch.nn import Module
from torch.optim import Adam
from typing import Dict, Any


def optimizer(
    model: Module,
    optimizer_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Instantiate optimizer configuration from `model`, `optimizer_kwargs`.

    Args:
        `model` (Module): Model to provide parameters.
        `optimizer_kwargs` (Dict[str, Any]): Key-value pairs used to instantiate optimizer.

    Returns:
        Dict (Dict[str, Any]): Optimizer configuration.
    """
    optimizer = Adam(  # Adam is enough
        model.parameters(),
        **optimizer_kwargs,
    )
    return {
        'optimizer': optimizer,
    }
