import torch as th
from torch.utils.data import Dataset
from typing import TypeVar, List, Dict, Any

DatasetClass = TypeVar('DatasetClass', bound=Dataset)


def vectorize(data: Dict[str, Any], rule: List[str] | Dict[str, Any] = None):
    """Vectorize a dictionary of tensors to one tensor by some rules.

    Args:
        `data` (Dict[str, Any]): A dictionary of tensors.
        `rule` (List[str] | Dict[str, Any]): Vectorization rule.
    """
    if type(rule) is list:
        assert type(
            data) is dict, "Requirement: Only dict of data can be vectorized."
        assert set(data.keys()) == set(
            rule), "Requirement: rule needs to cover all indexes."
        assert all([
            type(data[i]) is th.Tensor
            for i in rule
        ]), "Requirement: Only tensor data is supported."
        return th.concat([th.atleast_1d(data[i]) for i in rule], dim=-1)
    elif rule is None:
        return data
    else:
        assert type(
            rule
        ) is dict, "Requirement: Only rule dict or rule list is supported."
        newdata = {}
        for k, v in rule.items():
            newdata[k] = vectorize(data[k], v)
        return newdata


def vectorized(cls: DatasetClass):
    """A wrapper making Dataset automatically vectorized.

    Args:
        `cls` (DatasetClass): Dataset class. Only dataset with Dict of data can be vectorized.
    """

    class DatasetVectorizedWrapper(Dataset):

        def __init__(
            self,
            vectorize_rule: List[str] | Dict[str, Any],
            *args,
            **kwargs,
        ):
            """
            Args:
                `rule` (List[str] | Dict[str, Any]): Vectorization rule.

            Nested Dict is allowed, and `rule` should follow the structure of the data.\n

            Tuple and List are not supported, but one can ignore them by setting `rule` to `None`. For example:\n
            ```
            data = { 'variables': { 'x': tensor(1, 2, 3), 'y': tensor(4, 5, 6) }, 'tags': ['i', 'o'] }
            rule = { 'variables': ['y', 'x'], 'tags': None }
            ```
            Outputs:\n
            ```
            vectorized_data = { 'variables': tensor(4, 5, 6, 1, 2, 3), 'tags': ['i', 'o'] }
            ```
            """
            self.vectorize_rule = vectorize_rule
            self._wrapped_dataset = cls(*args, **kwargs)

        def __getitem__(self, index):
            data = self._wrapped_dataset[index]
            assert type(data) is dict, \
                "Requirement: Vectorization can only be applied to Dict of tensor data."
            return vectorize(data, self.vectorize_rule)

        def __len__(self):
            return len(self._wrapped_dataset)

    return DatasetVectorizedWrapper
