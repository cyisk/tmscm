import os
import torch as th
from torch.utils.data import Dataset
from typing import TypeVar

DatasetClass = TypeVar('DatasetClass', bound=Dataset)


def saved(cls: DatasetClass):
    """A wrapper making Dataset automatically saved and loaded.

    Args:
        `cls` (DatasetClass): Dataset class. Requrie data to be saved in attribute `data`.
    """

    class DatasetSavedWrapper(Dataset):

        def __init__(
            self,
            filepath: str,
            *args,
            **kwargs,
        ):
            """
            Args:
                `filepath` (str): Path to save or load data.
            """
            self.filepath = filepath
            if os.path.exists(self.filepath):
                self.data = th.load(self.filepath, weights_only=True)
            else:
                _wrapped_dataset = cls(*args, **kwargs)
                self.data = [
                    _wrapped_dataset[i] for i in range(len(_wrapped_dataset))
                ]
                th.save(
                    obj=self.data,
                    f=self.filepath,
                )

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    return DatasetSavedWrapper
