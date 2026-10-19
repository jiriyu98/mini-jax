```
What you know about computing, other people will learn. -- Alan J. Perlis
```

This is a learning project for people like me who are interested in Jax and its internals.

add(x, y) # simple python layer add
    ---> bind(add, x, y) # add - primtiive here
         ---> process_primitive(add, tracers) # add - primitive here, tracers - tracer wraps x & y