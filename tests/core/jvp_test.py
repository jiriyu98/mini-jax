import numpy as np
from absl.testing import absltest

from minijax._src.core import JVPTrace, JVPTracer, full_raise, new_main
from minijax._src.lax_numpy import add, mul


def _jvp(fun, primals, tangents):
    """Run `fun` under a JVPTrace and return (primal_out, tangent_out)."""
    trace = JVPTrace()
    with new_main(trace):
        out = fun(JVPTracer(trace, primals, tangents))
        out = full_raise(out, trace)
    return out.primal, out.tangent


def derivative(f, x):
    _, tangent = _jvp(f, x, 1.0)
    return tangent


class JVPTest(absltest.TestCase):
    def test_jvp_high_order(self):
        def f(x):

            def g(y):
                return mul(x, y)

            should_x = derivative(g, 3.0)
            return add(mul(x, should_x), x)

        np.testing.assert_allclose(derivative(f, 5.0), 11.0)
        np.testing.assert_allclose(derivative(f, 10.0), 21.0)


if __name__ == "__main__":
    absltest.main()
