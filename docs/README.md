# mini-jax

A from-scratch reimplementation of the whole JAX stack — tracing, IR, StableHLO, compiler, runtime,
sharding, Pallas — written to be **read**, not fast. NumPy only.

## Locked decisions

- **backend**: NumPy interpreter. No LLVM / Triton / MLIR.
- **name**: repo + package `mini-jax`, import `minijax`, namespace mirrors `jax`.
- **language**: pure Python ≥ 3.12 (this machine has 3.9.6 → use `uv`).

Don't re-open these.

## Ground rules

1. **Bottom-up honesty** — every layer runs standalone, no stubs.
2. **No upward imports** — enforced by `import-linter`, not hope.
3. **`minijax/` never imports `jax`** — real JAX appears only in `tests/diff/`.
4. **Readable over fast** — write down *why* it's slow, don't optimize.
5. **Diff, don't copy** — read JAX at the *compare* step, not the design step. Read it first and you'll
   transcribe production complexity instead of understanding the layer.
6. **Write the note before the code** — if you can't explain the layer, you can't build it.
7. **`main` stays runnable from M0 up** — one runnable demo per milestone. Blown past ~2× estimate, cut
   scope, not quality.

## Build order

`numpy wrapper → trace/tracer → bind/impl`. Once `bind` routes through a trace, the seam is
open: every later feature — AD, `vmap`, `jit` — is a *new `Trace` subclass*, not new machinery. Then
`{export → compiler → minijaxlib}` and `{sharding → pallas}` in parallel.

## Milestones

### M0 · numpy wrapper
- [x] `minijax.numpy` elementwise ops over plain scalars/ndarrays: `add`, `mul`, `neg`, `sin`, `cos`
- [x] parity vs numpy across a shape × dtype matrix

### M1 · trace + tracer
- [x] `Tracer` (a value + the trace it belongs to) and `Trace` (base interpreter)
- [x] `EvalTrace` — bottom trace: run `impl`, hand back a value (pass-through)
- [x] dynamic scope: `_trace_stack` + `new_main` push/pop

### M2 · bind/impl — the seam
- [x] `Primitive(name, impl)` — op registry; `impl` is the eager numpy function
- [x] `Primitive.bind` routes through the active trace → `impl`
- [x] tracer dispatch (`__add__`, `__mul__`, `__neg__`, `__radd__`)

M2 is the pivot: `EvalTrace` is only the first trace. Everything after this — `JVP`/`grad`,
`vmap`, `jit` — is a new `Trace` subclass plus its per-primitive rules over the *same* primitive
table, not a new interpreter.

### M3 · AD traces
- [ ] `JVPTrace.process_primitive` (stub exists) — forward-mode linearization
- [ ] transpose → `vjp`, `grad`, `value_and_grad`, `jacrev`/`jacfwd`
- [ ] finite-difference checks; transpose identity; `grad ∘ vmap`

### M4 · `vmap`
- [ ] `BatchTrace` / `BatchTracer` carrying `batch_dim`
- [ ] batch rules per primitive + generic moveaxis fallback
- [ ] nested vmap (depth 3); vs manual Python loop

### M5 · `jit`
- [ ] `abstract_eval` per primitive → avals (reuse the `ShapedArray` stub in `array.py`)
- [ ] `JaxprTrace` capturing `Var` / `Eqn` / `Jaxpr` + pretty printer
- [ ] `eval_jaxpr` with an environment dict
- [ ] trace cache keyed on avals + static args; `partial_eval`; `static_argnums`
- [ ] tracer-leak detection (a tracer escaping its trace → error)

### M6 · control flow
- [ ] `cond`, `scan`, `while_loop` — each with batch / jvp / transpose rules
- [ ] actionable error when a Python `if` depends on a tracer

### M7 · StableHLO
- [ ] opset table; type + attribute printer
- [ ] text emitter and parser (round-trip)
- [ ] verifier; compare emitted ops to real JAX for common ops

### M8 · compiler
- [ ] module IR + shape inference + verifier
- [ ] passes: DCE, CSE, constant folding, algebraic simplification
- [ ] fusion (elementwise chains), layout assignment, memory plan
- [ ] `Plan`: frozen dataclass that round-trips through `bytes`

### M9 · runtime
- [ ] runtime Protocol in **C-ABI-shaped types** — no `ndarray` crosses the seam
- [ ] device, `Buffer`, `device_put`, donation semantics
- [ ] executable + compilation cache
- [ ] async dispatch + `block_until_ready`

### M10 · sharding
- [ ] `Mesh`, `PartitionSpec`, `NamedSharding`
- [ ] per-op sharding propagation + collectives (`all_reduce`, `all_gather`, `reduce_scatter`)
- [ ] `shard_map`; 2-device result equals 1-device result

### M11 · Pallas
- [ ] block / grid / `Ref` DSL + memory spaces
- [ ] tile interpreter (deterministic and debuggable)
- [ ] lower to M8 IR; tiled matmul + fused softmax

### M12 · tinyGPT — the acceptance test
- [ ] attention, MLP, LayerNorm, embedding in `minijax.numpy`
- [ ] `jit(value_and_grad(loss))` + Adam
- [ ] sharded params; one Pallas kernel in the hot path
- [ ] overfit a single batch: loss strictly decreases

### M13 · polish
- [ ] "mini-jax vs JAX" mapping doc; benchmark table with a written explanation per slowdown
- [ ] notebooks: trace → jit → vmap → grad → shard → pallas

If M12 trains, the whole stack is right. If it doesn't, whichever layer broke is the one that's wrong.

## Done

A milestone is done when the gate passes and you've tagged:

```bash
uv run pytest -q                  # unit + golden + laws
uv run pytest tests/diff -q       # differential vs real jax
uv run lint-imports               # architecture legal (rules 2 + 3)
uv run python examples/mN_*.py    # demo runs
```

- Append 3 lines to §Notes: *what real JAX does / what we simplified / what we skipped*.
- Tag `mN-xxx`, version `0.N`. The tag chain is your learning log.

Tests use `absl.testing` (`absltest` + `parameterized`), run with pytest (or a single file directly:
`uv run python tests/lax_numpy/lax_numpy_test.py`). Toolchain: `uv` · `ruff` · `import-linter`.
`pyproject.toml`: runtime dep `numpy` only, `jax` behind a `[diff]` extra.

## Notes

<!--
### M0 · what real JAX does / what we simplified / what we skipped
-->
