import numpy as np
from absl.testing import absltest

from minijax._src.core import JVPTrace, JVPTracer, new_main
from minijax.numpy import add, mul


def _jvp(fun, primals, tangents):
    """Run `fun` under a JVPTrace and return (primal_out, tangent_out)."""
    trace = JVPTrace()
    with new_main(trace):
        out = fun(*[JVPTracer(trace, p, t) for p, t in zip(primals, tangents)])
    return out.primal, out.tangent


class JVPTest(absltest.TestCase):
    """M3: JVP trace returns primal + directional derivative."""

    def test_jvp_add(self):
        primal, tangent = _jvp(
            add,
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        )
        np.testing.assert_allclose(primal, [4.0, 6.0])
        np.testing.assert_allclose(tangent, [1.0, 1.0])

    def test_jvp_chain_rule(self):
        def f(x, y):
            return add(mul(x, y), x)

        primal, tangent = _jvp(
            f,
            [np.array(2.0), np.array(3.0)],
            [np.array(1.0), np.array(0.0)],
        )
        # f(x, y) = x*y + x  ->  primal = 2*3 + 2 = 8
        # d/dx with (dx, dy) = (1, 0)  ->  y*dx + x*dy + dx = 3 + 0 + 1 = 4
        np.testing.assert_allclose(primal, 8.0)
        np.testing.assert_allclose(tangent, 4.0)


if __name__ == "__main__":
    absltest.main()
