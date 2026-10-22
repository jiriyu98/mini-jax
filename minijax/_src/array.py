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
