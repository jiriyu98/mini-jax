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

jnp.add(x, y)  # -> ArrayImpl([5. 7. 9.])
jnp.sin(x)     # -> [0.841... 0.909... 0.141...]
```

# component

- [Primitive](https://docs.jax.dev/en/latest/601/jax-primitives.html): basica computational unit in jaxx
- [Trace](https://docs.jax.dev/en/latest/autodidax2_part1.html#main-idea-context-sensitive-interpretation): interpreter, while trace is a jax jargon

# reference
- [Autodidax1](https://docs.jax.dev/en/latest/autodidax.html), [Autodidax2](https://docs.jax.dev/en/latest/autodidax2_part1.html)