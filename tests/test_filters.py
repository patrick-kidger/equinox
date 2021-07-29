import equinox as eqx
import jax
import jax.numpy as jnp


def test_is_inexact_array():
    objs = [1, 2., [2.], True, object(), jnp.array([1])]
    results = [False, False, False, False, False, True]
    for o, r in zip(objs, results):
        assert eqx.is_inexact_array(o) == r


def test_is_array_like():
    objs = [1, 2., [2.], True, object(), jnp.array([1])]
    results = [True, True, True, True, False, True]
    for o, r in zip(objs, results):
        assert eqx.is_array_like(o) == r


def test_has_annotation():
    objs = [1, 2., [2.], True, object(), [jnp.array([1])]]
    eqx.set_annotation(objs[0], "my_annotation")
    eqx.set_annotation(objs[-1][0], "another_annotation")

    def f(args):
        flat, _ = jax.tree_flatten(args[-1])
        for arg in flat:
            assert not isinstance(arg, jax.core.Tracer)
        assert isinstance(args[-1], jax.core.Tracer)
        return 1.  # A scalar

    def filter_fn(obj):
        try:
            annotation = eqx.get_annotation(obj)
        except KeyError:
            return False
        else:
            return annotation == "another_annotation"

    eqx.jitf(f, filter_fn=filter_fn)(objs)
    eqx.gradf(f, argnums=(0, 1, 2, 3, 4, 5), filter_fn=filter_fn)(objs)
