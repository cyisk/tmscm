from torch import Tensor
from torch.utils.data import Dataset


class ColumnDataset(Dataset):
    """A special TensorDataset that only accepts one tensor as input."""

    def __init__(self, column: Tensor) -> None:
        self.column = column

    def __getitem__(self, index):
        return self.column[index]

    def __len__(self):
        return self.column.size(0)
