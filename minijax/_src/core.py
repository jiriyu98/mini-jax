import numpy as np


class ShapedArray:
    __slots__ = ["dtype", "shape", "weak_type"]

    def __init__(self, dtype, shape, weak_type=False):
        self.dtype = dtype
        self.shape = shape
        self.weak_type = weak_type


impl_rules = {
    "add": lambda x, y: np.add(x, y),
    "mul": lambda x, y: np.multiply(x, y),
    "neg": lambda x: np.negative(x),
    "sin": lambda x: np.sin(x),
    "cos": lambda x: np.cos(x),
}


class Primitive:
    __slots__ = ["impl_rule", "name"]

    def __init__(self, name: str):
        self.name = name
        self.impl_rule = impl_rules[name]

    def bind(self, *args, **params):
        # https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L631
        from minijax._src.array import ArrayImpl

        unwrapped = [a._data if isinstance(a, ArrayImpl) else a for a in args]
        return ArrayImpl(self.impl_rule(*unwrapped, **params))

    def abstract_eval():
        pass


class Jaxpr:
    pass
