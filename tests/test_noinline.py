import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .helpers import tree_allclose


def test_simple():
    @eqx.internal.noinline
    def addone(x):
        return x + 1

    assert tree_allclose(addone(1), 2)
    assert tree_allclose(addone(jnp.array(1)), jnp.array(2))
    assert tree_allclose(eqx.filter_jit(addone)(jnp.array(1)), jnp.array(2))


def test_mlp(getkey):
    mlp = eqx.nn.MLP(2, 2, 512, 2, key=getkey())
    mlp_noinline = eqx.internal.noinline(mlp)
    mlp_jit = eqx.filter_jit(mlp, donate="none")
    mlp_jit_noinline = eqx.filter_jit(mlp_noinline, donate="none")
    x = jr.normal(getkey(), (2,))
    o1 = mlp(x)
    o2 = mlp_jit(x)
    o3 = mlp_noinline(x)
    o4 = mlp_jit_noinline(x)
    assert tree_allclose(o1, o2)
    assert tree_allclose(o1, o3)
    assert tree_allclose(o1, o4)


def test_vmap(getkey):
    mlp = eqx.nn.MLP(2, 2, 512, 2, key=getkey())
    mlp_noinline = eqx.internal.noinline(mlp)
    mlp_vmap = eqx.filter_vmap(mlp)
    mlp_jit_vmap = eqx.filter_jit(mlp_vmap, donate="none")
    mlp_vmap_noinline = eqx.filter_vmap(mlp_noinline)
    mlp_jit_vmap_noinline = eqx.filter_jit(mlp_vmap_noinline, donate="none")
    x = jr.normal(getkey(), (5, 2))
    o1 = mlp_vmap(x)
    o2 = mlp_jit_vmap(x)
    o3 = mlp_vmap_noinline(x)
    o4 = mlp_jit_vmap_noinline(x)
    assert tree_allclose(o1, o2, atol=1e-5)
    assert tree_allclose(o1, o3, atol=1e-5)
    assert tree_allclose(o1, o4, atol=1e-5)


def test_jvp(getkey):
    mlp = eqx.nn.MLP(2, 2, 512, 2, key=getkey())
    mlp_noinline = eqx.internal.noinline(mlp)
    mlp_jvp = lambda p, t: jax.jvp(mlp, (p,), (t,))
    mlp_jit_jvp = eqx.filter_jit(mlp_jvp, donate="none")
    mlp_jvp_noinline = lambda p, t: jax.jvp(mlp_noinline, (p,), (t,))
    mlp_jit_jvp_noinline = eqx.filter_jit(mlp_jvp_noinline, donate="none")
    x = jr.normal(getkey(), (2,))
    y = jr.normal(getkey(), (2,))
    o1 = mlp_jvp(x, y)
    o2 = mlp_jit_jvp(x, y)
    o3 = mlp_jvp_noinline(x, y)
    o4 = mlp_jit_jvp_noinline(x, y)
    assert tree_allclose(o1, o2)
    assert tree_allclose(o1, o3)
    assert tree_allclose(o1, o4)


def test_grad(getkey):
    mlp = eqx.nn.MLP(2, 2, 512, 2, key=getkey())
    mlp_noinline = eqx.internal.noinline(mlp)
    mlp_grad = jax.grad(lambda x: jnp.sum(mlp(x)))
    mlp_jit_grad = eqx.filter_jit(mlp_grad, donate="none")
    mlp_grad_noinline = jax.grad(lambda x: jnp.sum(mlp_noinline(x)))
    mlp_jit_grad_noinline = eqx.filter_jit(mlp_grad_noinline, donate="none")
    x = jr.normal(getkey(), (2,))
    o1 = mlp_grad(x)
    o2 = mlp_jit_grad(x)
    o3 = mlp_grad_noinline(x)
    o4 = mlp_jit_grad_noinline(x)
    assert tree_allclose(o1, o2)
    assert tree_allclose(o1, o3)
    assert tree_allclose(o1, o4)


def test_num_traces():
    num_traces = 0

    @eqx.internal.noinline
    def fn(x):
        nonlocal num_traces
        num_traces += 1
        return x * 2

    @jax.jit
    def g(x):
        return fn(x) + fn(x) + fn(x) + fn(x)

    assert tree_allclose(g(1), jnp.array(8))
    assert num_traces == 2


def test_pytree_in():
    @eqx.filter_jit
    @eqx.internal.noinline
    def fn(f, x):
        return f(x[0][0])

    o1 = fn(lambda x: x + 1, [(1,)])
    o2 = fn(lambda x: x + 1, ([jnp.array(1)],))
    assert tree_allclose(o1, 2)
    assert tree_allclose(o2, jnp.array(2))


def test_abstract():
    f_num_traces = 0
    g_num_traces = 0
    call_num_traces = 0

    def abstract(_, x, y):
        return jnp.broadcast_arrays(x, y)[0]

    def f(x, y):
        nonlocal f_num_traces
        f_num_traces += 1
        return x + y

    def g(x, y):
        nonlocal g_num_traces
        g_num_traces += 1
        return x * y

    f = eqx.internal.noinline(f, abstract)
    g = eqx.internal.noinline(g, abstract)

    @jax.jit
    def call(fn, x, y):
        nonlocal call_num_traces
        call_num_traces += 1
        return fn(x, y)

    assert tree_allclose(call(f, 2, 3), jnp.array(5))
    assert tree_allclose(call(g, 2, 3), jnp.array(6))
    assert f_num_traces == 1
    assert g_num_traces == 1
    assert call_num_traces == 1


def test_complicated(getkey):
    num_lowerings = 0

    def increment(stack):
        nonlocal num_lowerings
        if stack.endswith("mlir"):
            num_lowerings += 1

    def call(f, x):
        return f(x + 1)

    @eqx.internal.noinline
    def call_noinline(f, x):
        print("hi")
        x = eqx.internal.announce_transform(x, announce=increment)
        return call(f, x)

    mlp = eqx.nn.MLP(1, 1, 16, 2, key=getkey())

    @jax.jit
    @jax.vmap
    @jax.grad
    def run(x):
        x = x[None]
        y = call(mlp, x)
        z = call(mlp, y)
        w = call(mlp, z)
        (out,) = w * x
        return out

    @jax.jit
    @jax.vmap
    @jax.grad
    def run_noinline(x):
        x = x[None]
        y = call_noinline(mlp, x)
        z = call_noinline(mlp, y)
        w = call_noinline(mlp, z)
        (out,) = w * x
        return out

    # Note that the w*x in the above is actually meaningful.
    # This promotes the cotangent on the loss (=1.0) to something
    # with a batch dimension.
    # Otherwise the `w = call_noinline(mlp, z)` receives a cotangent
    # without a batch dimension, whilst all the others do get a batch
    # dimension, and that triggers an additional compilation.

    xs = jnp.array([1.0, 2.0, 3.0])

    assert tree_allclose(run(xs), run_noinline(xs))
    assert num_lowerings == 3
    # Why three lowerings?
    # 1. Primal computation
    # 2. Cotangent computation
    # 3. Residuals for cotangent, i.e. another primal computation, just for the
    #    cotangents. This is because `noinline` currently doesn't pass residuals
    #    between forward and backward passes, and instead uses checkpointing.
    # Anyway, it's better than the six we'd have without `noinline`!
