import jax
import jax.numpy as jnp

import equinox.internal as eqxi

from .helpers import shaped_allclose


def _safe_zip(*args):
    length = len(args[0])
    assert all(len(a) == length for a in args[1:])
    return zip(*args)


def _assert_vars_equal(obj1, obj2, varnames):
    for varname in varnames:
        vars1 = getattr(obj1, varname)
        vars2 = getattr(obj2, varname)
        for a, b in _safe_zip(vars1, vars2):
            assert a.aval.strip_weak_type() == b.aval.strip_weak_type()


def _assert_jaxpr_equal(jaxpr1: jax.core.ClosedJaxpr, jaxpr2: jax.core.ClosedJaxpr):
    assert jaxpr1.consts == jaxpr2.consts
    jaxpr1 = jaxpr1.jaxpr
    jaxpr2 = jaxpr2.jaxpr
    _assert_vars_equal(jaxpr1, jaxpr2, ("invars", "outvars", "constvars"))
    for eqn1, eqn2 in _safe_zip(jaxpr1.eqns, jaxpr2.eqns):
        assert eqn1.primitive == eqn2.primitive
        assert eqn1.effects == eqn2.effects
        assert eqn1.params == eqn2.params
        _assert_vars_equal(eqn1, eqn2, ("invars", "outvars"))


def test_jaxpr2jaxpr_nocustom_idempotent():
    def fn(x):
        x = x + 1
        x + x * 2
        return x

    jaxpr = jax.make_jaxpr(fn)(1)
    jaxpr2 = eqxi.finalise_jaxpr(jaxpr)
    _assert_jaxpr_equal(jaxpr, jaxpr2)


def test_jaxpr2jaxpr_custom_idempotent():
    def fn(x):
        x = jnp.invert(x)
        x = eqxi.unvmap_any(x)
        x = jnp.invert(x)
        return x

    jaxpr = jax.make_jaxpr(fn)(True)
    jaxpr2 = eqxi.finalise_jaxpr(jaxpr, wrapper_prims=[eqxi.unvmap_any_p])
    jaxpr3 = eqxi.finalise_jaxpr(jaxpr2, wrapper_prims=[eqxi.unvmap_any_p])
    _assert_jaxpr_equal(jaxpr2, jaxpr3)

    jaxpr = jax.make_jaxpr(jax.vmap(fn))(jnp.array([True, False]))
    jaxpr2 = eqxi.finalise_jaxpr(jaxpr, wrapper_prims=[eqxi.unvmap_any_p])
    jaxpr3 = eqxi.finalise_jaxpr(jaxpr2, wrapper_prims=[eqxi.unvmap_any_p])
    _assert_jaxpr_equal(jaxpr2, jaxpr3)


def test_fn2fn_nocustom_idempotent():
    def fn(x):
        x = x + 1
        x + x * 2
        return x

    finalised_fn = eqxi.finalise_fn(fn)
    assert shaped_allclose(fn(1), finalised_fn(1), match_weak=True)
    assert shaped_allclose(fn(5), finalised_fn(5), match_weak=True)
    assert shaped_allclose(fn(-1), finalised_fn(-1), match_weak=True)

    jaxpr = jax.make_jaxpr(fn)(1)
    finalised_jaxpr = jax.make_jaxpr(finalised_fn)(1)
    _assert_jaxpr_equal(finalised_jaxpr, jaxpr)


def test_fn2fn_custom_idempotent():
    def fn(x):
        x = jnp.invert(x)
        x = eqxi.unvmap_any(x)
        x = jnp.invert(x)
        return x

    finalised_fn = eqxi.finalise_fn(fn, wrapper_prims=[eqxi.unvmap_any_p])
    assert shaped_allclose(fn(False), finalised_fn(False))
    assert shaped_allclose(fn(True), finalised_fn(True))

    finalised_jaxpr = jax.make_jaxpr(finalised_fn)(True)
    finalised_finalised_jaxpr = jax.make_jaxpr(eqxi.finalise_fn(finalised_fn))(True)
    _assert_jaxpr_equal(finalised_jaxpr, finalised_finalised_jaxpr)
    for eqn in finalised_jaxpr.eqns:
        assert eqn.primitive != eqxi.unvmap_any_p

    vmap_fn = jax.vmap(fn)
    finalised_vmap_fn = eqxi.finalise_fn(vmap_fn, wrapper_prims=[eqxi.unvmap_any_p])
    for arg in (
        jnp.array([False, False]),
        jnp.array([False, True]),
        jnp.array([True, False]),
        jnp.array([True, True]),
    ):
        assert shaped_allclose(vmap_fn(arg), finalised_vmap_fn(arg))

    finalised_vmap_jaxpr = jax.make_jaxpr(finalised_vmap_fn)(jnp.array([False, False]))
    finalised_finalised_vmap_jaxpr = jax.make_jaxpr(
        eqxi.finalise_fn(finalised_vmap_fn)
    )(jnp.array([False, False]))
    for eqn in finalised_vmap_jaxpr.eqns:
        assert eqn.primitive != eqxi.unvmap_any_p
    _assert_jaxpr_equal(finalised_vmap_jaxpr, finalised_finalised_vmap_jaxpr)
