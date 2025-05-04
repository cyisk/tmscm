from itertools import accumulate
from torch import arange, cat, Tensor, device
from typing import List, Dict


class Iota:

    def __init__(
        self,
        dimensions: Dict[str, int],
        order: List[str],
    ):
        """Rules for vectorization by causal order.

        Args:
            `dimensions` (Dict[str, int]): The dimension mapping from indexes to number of dimensions.
            `order` (List[str]): The causal order for vectorization.
        """
        assert set(list(dimensions.keys())) == set(order), \
            "Requirement: Order needs to cover all indexes."
        self.indexes: List[int] = list(dimensions.keys())
        self.dimensions: Dict[str, int] = dimensions
        self.order: List[str] = order
        ordered_dimensions = [dimensions[i] for i in order]
        cumsum_dimensions = list(accumulate(ordered_dimensions))
        self.lower_dimensions: Dict[str, int] = {
            i: cumsum_dimensions[o - 1] if o > 0 else 0
            for o, i in enumerate(order)
        }
        self.cardinality: int = cumsum_dimensions[-1]

    def __repr__(self):
        return self.__class__.__name__ + "(\n" + "    Dimensions: " + str(
            self.dimensions) + "\n    Orders: " + str(self.order) + "\n)"

    def ij(
        self,
        i: str,
        j: int,
    ) -> int:
        """Get the index of vectorized :math:`(index, dimension)`.

        Args:
            `i` (str): The index of variable.
            `j` (int): The dimension.

        Returns:
            int: The index of vectorized (index, dimension).
        """
        return self.lower_dimensions[i] + j

    def i(
        self,
        i: str,
        prefix: bool = False,
        device: device = None,
    ) -> Tensor:
        """Get indexes of vectorized :math:`(index, *)`.

        Args:
            `i` (str): The index of variable.
            `prefix` (bool, optional): Whether get all prefix of :math:`(index, *)` by causal order. Defaults to False.
            `device` (device, optional): The device of indexes tensor. Defaults to None.

        Returns:
            Tensor: Indexes of vectorized :math:`(index, *)`.
        """
        if prefix:
            return cat([self.i(i, device=device) for i in range(0, i + 1)])
        low = self.lower_dimensions[i]
        high = low + self.dimensions[i]
        return arange(low, high).int().to(device)

    def prefix(
        self,
        i: str,
        device: device = None,
    ) -> Tensor:
        """Get all prefixed indexes of vectorized :math:`(index, *)`, with `i` excluded.

        Args:
            `i` (str): The index of variable.
            `device` (device, optional): The device of indexes tensor. Defaults to None.

        Returns:
            Tensor: All prefixed indexes of vectorized :math:`(index, *)`.
        """
        low = self.lower_dimensions[i]
        return arange(0, low).int().to(device)

    def suffix(
        self,
        i: str,
        device: device = None,
    ) -> Tensor:
        """Get all suffixed indexes of vectorized :math:`(index, *)`, with `i` excluded.

        Args:
            `i` (str): The index of variable.
            `device` (device, optional): The device of indexes tensor. Defaults to None.

        Returns:
            Tensor: All suffixed indexes of vectorized :math:`(index, *)`.
        """
        low = self.lower_dimensions[i]
        high = low + self.dimensions[i]
        return arange(high, self.cardinality).int().to(device)
