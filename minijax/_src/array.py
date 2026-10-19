import numpy as np


class ArrayImpl:  # concrete array, ConcreteArray(ShapedArray)
    """
    Base Array class for mini JAX
    """

    __slots__ = ["_data", "_weak_type"]

    def __init__(self, data: np.ndarray, weak_type=False) -> None:
        self._data = data
        self._weak_type = weak_type

    @property
    def data(self) -> np.array:
        return self._data

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def shape(self) -> np.shape:
        return self._data.shape

    @property
    def ndim(self) -> np.ndim:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def weak_type(self):
        return self._weak_type
