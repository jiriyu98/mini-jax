from minijax._src.array import ArrayImpl
from minijax._src.core import Primitive

add_p = Primitive("add")
mul_p = Primitive("mul")
neg_p = Primitive("neg")
sin_p = Primitive("sin")
cos_p = Primitive("cos")


def add(x, y):
    return add_p.bind(x, y)


def mul(x, y):
    return mul_p.bind(x, y)


def neg(x):
    return neg_p.bind(x)


def sin(x):
    return sin_p.bind(x)


def cos(x):
    return cos_p.bind(x)


ArrayImpl.__add__ = add
ArrayImpl.__radd__ = add
ArrayImpl.__mul__ = mul
ArrayImpl.__neg__ = neg
