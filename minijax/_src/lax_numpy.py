import numpy as np

from minijax._src.core import EvalTrace, JVPTrace, Primitive

add_p = Primitive("add", np.add)
mul_p = Primitive("mul", np.multiply)
neg_p = Primitive("neg", np.negative)
sin_p = Primitive("sin", np.sin)
cos_p = Primitive("cos", np.cos)

EvalTrace.impl_rules = {
    add_p: np.add,
    mul_p: np.multiply,
    neg_p: np.negative,
    sin_p: np.sin,
    cos_p: np.cos,
}


def add_jvp(primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return x + y, x_dot + y_dot


def mul_jvp(primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return x * y, y * x_dot + x * y_dot


def sin_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return np.sin(x), np.cos(x) * x_dot


def cos_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return np.cos(x), -np.sin(x) * x_dot


def neg_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return -x, -x_dot


JVPTrace.impl_rules = {
    add_p: add_jvp,
    mul_p: mul_jvp,
    neg_p: neg_jvp,
    sin_p: sin_jvp,
    cos_p: cos_jvp,
}


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
