from minijax._src.array import ArrayImpl


def _unwrap(x):
    """ArrayImpl -> its underlying ndarray; anything else passes through."""
    return x._data if isinstance(x, ArrayImpl) else x


# -------------------- Tracing --------------------

# Let's think about why we need to "raise" "lower" "find top trace":
# For short: we have different traces with corresponding tracer,
# which is because tracer (interpreters have different transformation rules!),
# we must make sure current val is as the same trace type.
# EvalTrace and JVPTrace can be a good comparison for understanding this.


class Tracer[TraceType: Trace]:
    """A value during tracing: a concrete value plus the trace it belongs to."""

    __slots__ = ["_trace"]

    _trace: TraceType

    def full_lower(self):
        return self


class JVPTracer(Tracer):
    __slots__ = ["_trace", "primal", "tangent"]

    def __init__(self, trace, primal, tangent):
        self._trace = trace
        self.primal = primal
        self.tangent = tangent

    def full_lower(self):
        return self


class Trace:
    """Base class. A trace decides what `bind` does with a primitive."""

    __slots__ = ["impl_rules"]

    def pure(self, val):
        """
        transform a concrete value to a tracer
        """
        raise NotImplementedError

    def lift(self, val):
        """
        lift a tracer in outer trace to inner trace

        e.g.             bottom   top
            trace stack:[trace1, trace2]
            Transform a tracer type in trace1 to trace2 is what jax calls "lift".
        """
        raise NotImplementedError

    def process_primitive(self, primitive, tracers, params):
        raise NotImplementedError


class EvalTrace(Trace):
    """
    The bottom trace: run `impl` and hand the result back (pass-through).
    ref:
    https://github.com/jax-ml/jax/blob/1937fa70027e81886ab9878e57a58482b94abee6/jax/_src/core.py#L1249
    """

    def pure(self, val):
        return val

    def lift(self, val):
        return val

    def process_primitive(self, primitive, tracers, params):
        return self.impl_rules[primitive](*[_unwrap(t) for t in tracers], **params)


class JVPTrace(Trace):
    """
    The JVP AD Trace.
    ref:
    https://github.com/jax-ml/jax/blob/1937fa70027e81886ab9878e57a58482b94abee6/jax/_src/interpreters/ad.py#L529
    """

    def pure(self, val):
        return JVPTracer(self, val, 0.0)

    def lift(self, val):
        return JVPTracer(self, val.primal, 0.0)

    def process_primitive(self, primitive, tracers, params):
        primals = [tracer.primal for tracer in tracers]
        tangents = [tracer.tangent for tracer in tracers]
        primal_out, tangent_out = self.impl_rules[primitive](primals, tangents)
        return JVPTracer(self, primal_out, tangent_out)


def full_lower(val):
    if isinstance(val, Tracer):
        return val.full_lower()
    return val


def full_raise(val, trace: Trace):
    """
    handle all trace transformation
    """
    if isinstance(val, Tracer):
        if val._trace is trace:
            return val
        return trace.lift(val)
    return trace.pure(val)


# -------------------- dynamic scope --------------------
# currently I don't think we need "__array_priority__"
_trace_stack: list[Trace] = [EvalTrace()]


class new_main:
    """Enter/exit a trace: push on enter, pop on exit (dynamic scope)."""

    def __init__(self, trace):
        self.trace = trace

    def __enter__(self):
        _trace_stack.append(self.trace)
        return self.trace

    def __exit__(self, *exc):
        _trace_stack.pop()


class pop_trace_stack:
    """Pop trace stack to make a transient enviornment"""

    def __init__(self):
        self.prev = _trace_stack[-1]

    def __enter__(self):
        return _trace_stack.pop()

    def __exit__(self, *exc):
        _trace_stack.append(self.prev)


# -------------------- Primitive --------------------


class Primitive:
    """
    Operators like add, neg, mul or customized ones.
    For more information, refer to: https://docs.jax.dev/en/latest/601/jax-primitives.html
    """

    __slots__ = ["impl", "name"]

    def __init__(self, name, impl):
        self.name = name
        self.impl = impl

    def bind(self, *args, **params):
        with pop_trace_stack() as trace:
            tracers = [
                full_raise(a, trace) if isinstance(a, Tracer) else trace.pure(a)
                for a in args
            ]
            outs = trace.process_primitive(self, tracers, params)

            return full_lower(outs)
