import torch as th
from torch.utils.data import TensorDataset
from typing import TypeVar, Dict, Any

DatasetClass = TypeVar('DatasetClass', bound=TensorDataset)


# Standardize according to rule
def standardize(
    mean: Dict[str, Any],
    std: Dict[str, Any],
    data: Dict[str, Any],
    rule: Dict[str, Any] = None,
):
    newdata = {}
    for k in data:
        if k in rule and rule[k] is None:
            newdata[k] = data[k]
            continue
        elif k not in rule:
            mean_k, std_k = mean[k], std[k]
        else:
            v = rule[k]
            mean_k = mean[v['mean']] if 'mean' in v else mean[k]
            std_k = std[v['std']] if 'std' in v else std[k]
        newdata[k] = (data[k] - mean_k) / std_k
    return newdata


def standardized(cls: DatasetClass):
    """A wrapper making Dataset automatically standardized.

    Args:
        `cls` (DatasetClass): Dataset class. Only dataset with Dict of tensor data can be standardized.
    """

    class DatasetStandardizedWrapper(TensorDataset):

        def __init__(
            self,
            standardize_rule: Dict[str, Dict[str, str]],
            *args,
            **kwargs,
        ):
            """
            Args:
                `standardize_rule` (List[str] | Dict[str, Any]): Standardization rule.

            By default, one column will be standardized by its statistics (`mean` and `std`),
            but it can be standardized with other column's statistics specified by `rule`.
            Nested Dict is not allowed.

            If one want to ignore part of data, set `rule` to `None`. For example:\n
            ```
            data = { 'x': tensor(1, 2, 3), 'y': tensor(-2, 6), 'masks': tensor(0, 1, 1)}
            rule = { 'y': { 'mean': 'x' }, 'tags': None }
            ```
            If `x.mean` is :math:`2`, `x.std` is :math:`1`, and `y.std` is :math:`2`, then outputs:\n
            ```
            standardized_data = { 'x': tensor(-1, 0, 1), 'y': tensor(-1, 2), 'masks': tensor(0, 1, 1)}
            ```
            """
            self.standardize_rule = standardize_rule
            self._wrapped_dataset = cls(*args, **kwargs)
            data_keys = list(self._wrapped_dataset[0].keys())
            data = {
                k:
                th.stack(
                    tensors=[
                        self._wrapped_dataset[i][k]
                        for i in range(len(self._wrapped_dataset))
                    ],
                    dim=0,
                )
                for k in data_keys
            }
            self._mean = {
                k: th.mean(data[k], dim=0)
                for k in data_keys
                if k not in standardize_rule or standardize_rule[k] is not None
            }
            self._std = {
                k: th.std(data[k], dim=0)
                for k in data_keys
                if k not in standardize_rule or standardize_rule[k] is not None
            }

        def __getitem__(self, index):
            data = self._wrapped_dataset[index]
            return standardize(
                mean=self._mean,
                std=self._std,
                data=data,
                rule=self.standardize_rule,
            )

        def __len__(self):
            return len(self._wrapped_dataset)

    return DatasetStandardizedWrapper
