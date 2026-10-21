import numpy as np
from absl.testing import absltest, parameterized

from minijax import ArrayImpl
from minijax.numpy import add, cos, mul, neg, sin


def _arr(a):
    return ArrayImpl(np.asarray(a))


def _rand(shape, dtype):
    if np.issubdtype(dtype, np.integer):
        a = np.random.randint(-5, 5, size=shape)
    else:
        a = np.random.randn(*shape)
    return np.asarray(a).astype(dtype)


class LaxNumpyTest(parameterized.TestCase):
    """M0 parity: minijax ops vs numpy, across a shape x dtype matrix."""

    @parameterized.product(
        op=[("add", add, np.add), ("mul", mul, np.multiply)],
        dtype=[np.float32, np.float64, np.int32],
        shape=[(), (3,), (2, 3)],
    )
    def test_binary(self, op, dtype, shape):
        _, mj_fn, np_fn = op
        a = _rand(shape, dtype)
        b = _rand(shape, dtype)
        np.testing.assert_allclose(
            mj_fn(_arr(a), _arr(b)).data, np_fn(a, b), rtol=1e-6, atol=1e-6
        )

    @parameterized.product(
        op=[("neg", neg, np.negative), ("sin", sin, np.sin), ("cos", cos, np.cos)],
        dtype=[np.float32, np.float64],
        shape=[(), (3,), (2, 3)],
    )
    def test_unary(self, op, dtype, shape):
        _, mj_fn, np_fn = op
        a = _rand(shape, dtype)
        np.testing.assert_allclose(mj_fn(_arr(a)).data, np_fn(a), rtol=1e-6, atol=1e-6)

    def test_operators(self):
        x = _arr([1.0, 2.0, 3.0])
        y = _arr([4.0, 5.0, 6.0])
        np.testing.assert_allclose((x + x).data, [2.0, 4.0, 6.0])
        np.testing.assert_allclose((x * y).data, [4.0, 10.0, 18.0])
        np.testing.assert_allclose((-x).data, [-1.0, -2.0, -3.0])


if __name__ == "__main__":
    absltest.main()
