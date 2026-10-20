import numpy as np
from absl.testing import absltest

from minijax import ArrayImpl, EvalTrace, Tracer, add, mul, new_main
from minijax._src.lax_numpy import add_p


def _arr(a):
    return ArrayImpl(np.asarray(a))


def f(x, y):
    return add(mul(x, y), x)


class TraceTest(absltest.TestCase):
    """M1a: bind routes through an active trace; pass-through gives eager results."""

    def test_eval_trace_processes_primitive(self):
        trace = EvalTrace()
        out = trace.process_primitive(
            add_p,
            [Tracer(trace, np.array([1.0])), Tracer(trace, np.array([2.0]))],
            {},
        )
        np.testing.assert_allclose(out.val, [3.0])

    def test_bind_routes_through_active_trace(self):
        class CountingTrace(EvalTrace):
            def __init__(self):
                self.calls = 0

            def process_primitive(self, primitive, tracers, params):
                self.calls += 1
                return super().process_primitive(primitive, tracers, params)

        trace = CountingTrace()
        with new_main(trace):
            out = add(_arr([1.0]), _arr([2.0]))
        self.assertEqual(trace.calls, 1)
        np.testing.assert_allclose(out.data, [3.0])

    def test_trace_equals_eager(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        with new_main(EvalTrace()):
            traced = f(_arr(x), _arr(y))
        expected = np.add(np.multiply(x, y), x)
        np.testing.assert_allclose(traced.data, expected)


if __name__ == "__main__":
    absltest.main()
