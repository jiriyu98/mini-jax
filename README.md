```
What you know about computing, other people will learn. -- Alan J. Perlis
```

# mini-jax

A learning-oriented project for people like me who are interested in Jax and its internals.

# try it

```bash
uv sync          # or: pip install -e .
uv run python
```

```python
import minijax.numpy as jnp
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 5.0, 6.0])

jnp.add(x, y)  # -> array([5., 7., 9.])
jnp.sin(x)     # -> [0.841... 0.909... 0.141...]
```

# layers (build order)

1. **numpy wrapper** — `minijax.numpy` exposes `add`, `mul`, `neg`, `sin`, `cos` as a thin layer over
   numpy (plain scalars/ndarrays in, ndarray out).
2. **trace + tracer** — a [`Trace`](https://docs.jax.dev/en/latest/autodidax2_part1.html#main-idea-context-sensitive-interpretation)
   is an interpreter; a `Tracer` is a value plus the trace it belongs to.
3. **bind/impl** — a [`Primitive`](https://docs.jax.dev/en/latest/601/jax-primitives.html) (name + numpy
   `impl`); `bind` routes through the active trace.

# reference
- [Autodidax1](https://docs.jax.dev/en/latest/autodidax.html), [Autodidax2](https://docs.jax.dev/en/latest/autodidax2_part1.html)