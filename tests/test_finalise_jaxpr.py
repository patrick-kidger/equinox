from typing import cast

import equinox.internal as eqxi
import jax
import jax.core
import jax.extend.core
import jax.lax as lax
import jax.numpy as jnp

from .helpers import tree_allclose


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


def _assert_jaxpr_equal(
    jaxpr1: jax.extend.core.ClosedJaxpr, jaxpr2: jax.extend.core.ClosedJaxpr
):
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
        x = x * 2
        return x

    jaxpr = cast(jax.extend.core.ClosedJaxpr, jax.make_jaxpr(fn)(1))
    jaxpr2 = eqxi.finalise_jaxpr(jaxpr)
    _assert_jaxpr_equal(jaxpr, jaxpr2)


def test_jaxpr2jaxpr_custom_idempotent():
    def fn(x):
        x = jnp.invert(x)
        x = eqxi.unvmap_any(x)
        x = jnp.invert(x)
        return x

    jaxpr = cast(jax.extend.core.ClosedJaxpr, jax.make_jaxpr(fn)(True))
    jaxpr2 = eqxi.finalise_jaxpr(jaxpr)
    jaxpr3 = eqxi.finalise_jaxpr(jaxpr2)
    _assert_jaxpr_equal(jaxpr2, jaxpr3)

    jaxpr = jax.make_jaxpr(jax.vmap(fn))(jnp.array([True, False]))
    jaxpr = cast(jax.extend.core.ClosedJaxpr, jaxpr)
    jaxpr2 = eqxi.finalise_jaxpr(jaxpr)
    jaxpr3 = eqxi.finalise_jaxpr(jaxpr2)
    _assert_jaxpr_equal(jaxpr2, jaxpr3)


def test_fn2fn_nocustom_idempotent():
    def fn(x):
        x = jnp.asarray(x)
        x = x + 1
        x = x * 2
        return x

    finalised_fn = eqxi.finalise_fn(fn)
    assert tree_allclose(fn(1), finalised_fn(1))
    assert tree_allclose(fn(5), finalised_fn(5))
    assert tree_allclose(fn(-1), finalised_fn(-1))

    jaxpr = jax.make_jaxpr(fn)(1)
    jaxpr = cast(jax.extend.core.ClosedJaxpr, jaxpr)
    finalised_jaxpr = jax.make_jaxpr(finalised_fn)(1)
    finalised_jaxpr = cast(jax.extend.core.ClosedJaxpr, finalised_jaxpr)
    _assert_jaxpr_equal(finalised_jaxpr, jaxpr)


def test_fn2fn_custom_idempotent():
    def fn(x):
        x = jnp.invert(x)
        x = eqxi.unvmap_any(x)
        x = jnp.invert(x)
        return x

    finalised_fn = eqxi.finalise_fn(fn)
    assert tree_allclose(fn(False), finalised_fn(False))
    assert tree_allclose(fn(True), finalised_fn(True))

    finalised_jaxpr = jax.make_jaxpr(finalised_fn)(True)
    finalised_jaxpr = cast(jax.extend.core.ClosedJaxpr, finalised_jaxpr)
    finalised_finalised_jaxpr = jax.make_jaxpr(eqxi.finalise_fn(finalised_fn))(True)
    finalised_finalised_jaxpr = cast(
        jax.extend.core.ClosedJaxpr, finalised_finalised_jaxpr
    )
    _assert_jaxpr_equal(finalised_jaxpr, finalised_finalised_jaxpr)
    for eqn in finalised_jaxpr.eqns:
        assert eqn.primitive != eqxi.unvmap_any_p

    vmap_fn = jax.vmap(fn)
    finalised_vmap_fn = eqxi.finalise_fn(vmap_fn)
    for arg in (
        jnp.array([False, False]),
        jnp.array([False, True]),
        jnp.array([True, False]),
        jnp.array([True, True]),
    ):
        assert tree_allclose(vmap_fn(arg), finalised_vmap_fn(arg))

    finalised_vmap_jaxpr = jax.make_jaxpr(finalised_vmap_fn)(jnp.array([False, False]))
    finalised_vmap_jaxpr = cast(jax.extend.core.ClosedJaxpr, finalised_vmap_jaxpr)
    finalised_finalised_vmap_jaxpr = jax.make_jaxpr(
        eqxi.finalise_fn(finalised_vmap_fn)
    )(jnp.array([False, False]))
    finalised_finalised_vmap_jaxpr = cast(
        jax.extend.core.ClosedJaxpr, finalised_finalised_vmap_jaxpr
    )
    for eqn in finalised_vmap_jaxpr.eqns:
        assert eqn.primitive != eqxi.unvmap_any_p
    _assert_jaxpr_equal(finalised_vmap_jaxpr, finalised_finalised_vmap_jaxpr)


def _assert_no_unvmap(jaxpr: jax.extend.core.Jaxpr):
    for eqn in jaxpr.eqns:
        assert eqn.primitive not in (eqxi.unvmap_any_p, eqxi.unvmap_all_p)
    for subjaxpr in jax.core.subjaxprs(jaxpr):
        _assert_no_unvmap(subjaxpr)


def test_cond():
    def f(pred, x):
        return lax.cond(pred, eqxi.unvmap_all, eqxi.unvmap_any, x)

    single_jaxpr = eqxi.finalise_make_jaxpr(f)(True, True)
    batch_jaxpr = eqxi.finalise_make_jaxpr(f)(False, jnp.array([True, False]))

    for jaxpr in (single_jaxpr, batch_jaxpr):
        _assert_no_unvmap(jaxpr.jaxpr)


def test_custom_jvp():
    @jax.custom_jvp
    def f(pred, x):
        pred = eqxi.unvmap_any(pred)
        return lax.cond(pred, lambda y: y, lambda y: y + 1, x)

    @f.defjvp
    def f_jvp(x, tx):
        # Note that we don't need to remove any wrapper primitives in a JVP rule, since
        # finalisation happens after autodiff.
        assert False

    jaxpr = eqxi.finalise_make_jaxpr(f)(True, 1.0)
    _assert_no_unvmap(jaxpr.jaxpr)


def test_custom_vjp():
    @jax.custom_vjp
    def f(pred, x):
        pred = eqxi.unvmap_any(pred)
        return lax.cond(pred, lambda y: y, lambda y: y + 1, x)

    def f_fwd(pred, x):
        assert False

    def f_bwd(aux, pred, x):
        assert False

    f.defvjp(f_fwd, f_bwd)

    jaxpr = eqxi.finalise_make_jaxpr(f)(True, 1.0)
    _assert_no_unvmap(jaxpr.jaxpr)


def test_checkpoint():
    @jax.checkpoint  # pyright: ignore
    def f(pred, x):
        pred = eqxi.unvmap_any(pred)
        return lax.cond(pred, lambda y: y, lambda y: y + 1, x)

    jaxpr = eqxi.finalise_make_jaxpr(f)(True, 1.0)
    _assert_no_unvmap(jaxpr.jaxpr)
