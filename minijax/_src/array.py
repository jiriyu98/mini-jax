import numpy as np

# -------------------- Abstract Value --------------------


class AbstractValue:
    """
    Base class for abstract values (the compile-time description of a value).
    """

    __slots__ = []


class ShapedArray(AbstractValue):
    __slots__ = ["dtype", "shape", "weak_type"]

    def __init__(self, dtype, shape, weak_type=False):
        self.dtype = dtype
        self.shape = shape
        self.weak_type = weak_type


# -------------------- Concrete Value --------------------


class Array:
    """
    Base Array class for mini-jax.
    Array is for concrete array, corresponding to the concept of shaped array.
    """

    __slots__ = ["_data", "_weak_type"]

    @property
    def data(self) -> np.array:
        raise NotImplementedError

    @property
    def dtype(self) -> np.dtype:
        raise NotImplementedError

    @property
    def shape(self) -> np.shape:
        raise NotImplementedError

    @property
    def ndim(self) -> np.ndim:
        raise NotImplementedError

    @property
    def size(self) -> int:
        raise NotImplementedError

    @property
    def weak_type(self) -> bool:
        raise NotImplementedError

    def __getitem__(self, *args):
        raise NotImplementedError

    def __setitem__(self, *args):
        raise NotImplementedError


class ArrayImpl(Array):
    """
    Base Array Impl class for mini JAX
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
    def weak_type(self) -> bool:
        return self._weak_type

    def __getitem__(self, *args):
        return self._data.__getitem__(*args)

    def __setitem__(self, *args):
        return self._data.__setitem__(*args)

    def __str__(self):
        return self._data.__str__()
