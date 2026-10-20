from minijax._src.array import ArrayImpl


def _unwrap(x):
    """ArrayImpl -> its underlying ndarray; anything else passes through."""
    return x._data if isinstance(x, ArrayImpl) else x


# -------------------- Tracing --------------------


class Tracer:
    """A value during tracing: a concrete value plus the trace it belongs to."""

    __slots__ = ["trace", "val"]

    def __init__(self, trace, val):
        self.trace = trace
        self.val = val


class Trace:
    """Base class. A trace decides what `bind` does with a primitive."""

    def process_primitive(self, primitive, tracers, params):
        raise NotImplementedError


class EvalTrace(Trace):
    """The bottom trace: run `impl` and hand the result back (pass-through)."""

    def process_primitive(self, primitive, tracers, params):
        return Tracer(self, primitive.impl(*[t.val for t in tracers], **params))


# -------------------- dynamic scope --------------------
# currently I don't think we need "__array_priority__"
_trace_stack = [EvalTrace()]


class new_main:
    """Enter/exit a trace: push on enter, pop on exit (dynamic scope)."""

    def __init__(self, trace):
        self.trace = trace

    def __enter__(self):
        _trace_stack.append(self.trace)
        return self.trace

    def __exit__(self, *exc):
        _trace_stack.pop()


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
        trace = _trace_stack[-1]
        tracers = [
            a if isinstance(a, Tracer) else Tracer(trace, _unwrap(a)) for a in args
        ]
        return ArrayImpl(trace.process_primitive(self, tracers, params).val)
