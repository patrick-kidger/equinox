import functools as ft

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import equinox as eqx


def test_basic():
    a = [jnp.array(3), jnp.array(2)]
    b = [jnp.array(4), jnp.array(5)]
    index = eqx.experimental.StateIndex()
    eqx.experimental.set_state(index, a)
    assert eqx.experimental.get_state(index, b) == a


def test_jit():
    index = eqx.experimental.StateIndex()

    @eqx.filter_jit
    def set_state(x):
        eqx.experimental.set_state(index, x)

    @eqx.filter_jit
    def get_state(x):
        return eqx.experimental.get_state(index, x)

    a = [jnp.array(3), jnp.array(2)]
    b = [jnp.array(4), jnp.array(5)]
    set_state(a)
    assert get_state(b) == a


def test_no_nonjaxarray():
    c = 0
    index = eqx.experimental.StateIndex()
    with pytest.raises(TypeError):
        eqx.experimental.set_state(index, c)

    d = object()
    index = eqx.experimental.StateIndex()
    with pytest.raises(TypeError):
        eqx.experimental.set_state(index, d)

    e = [jnp.array(2), 0]
    index = eqx.experimental.StateIndex()
    with pytest.raises(TypeError):
        eqx.experimental.set_state(index, e)


def test_no_set():
    index = eqx.experimental.StateIndex()
    a = jnp.array(2)
    with pytest.raises(RuntimeError):
        eqx.experimental.get_state(index, a)


def test_no_change_shape():
    index1 = eqx.experimental.StateIndex()
    index2 = eqx.experimental.StateIndex()

    @jax.jit
    def set_state1():
        eqx.experimental.set_state(index1, jnp.array(1))
        eqx.experimental.set_state(index1, jnp.array([2]))

    @jax.jit
    def set_state2():
        eqx.experimental.set_state(index2, jnp.array(1))
        eqx.experimental.set_state(index2, [jnp.array(1)])

    with pytest.raises(RuntimeError):
        set_state1()
    with pytest.raises(RuntimeError):
        set_state2()


def test_index_jittable():
    index1 = eqx.experimental.StateIndex()
    index2 = eqx.experimental.StateIndex()

    @eqx.filter_jit
    def get_state(i, x):
        return eqx.experimental.get_state(i, x)

    a = [jnp.array(3), jnp.array(2)]
    b = [jnp.array(4), jnp.array(5)]
    c = [jnp.array(6), jnp.array(9)]
    d = [jnp.array(7), jnp.array(8)]
    eqx.experimental.set_state(index1, a)
    eqx.experimental.set_state(index2, b)
    assert get_state(index1, c) == a
    assert get_state(index2, d) == b


@pytest.mark.parametrize("with_jit", (False, True))
@pytest.mark.parametrize("with_pytree", (False, True))
def test_vmap(with_jit, with_pytree):
    index1 = eqx.experimental.StateIndex()
    index2 = eqx.experimental.StateIndex()

    @ft.partial(jax.vmap, in_axes=(None, 0))
    def vmap_set_state(i, x):
        eqx.experimental.set_state(i, x)

    @ft.partial(jax.vmap, in_axes=(None, 0))
    def vmap_get_state(i, x):
        return eqx.experimental.get_state(i, x)

    if with_jit:
        vmap_set_state = eqx.filter_jit(vmap_set_state, donate="none")
        vmap_get_state = eqx.filter_jit(vmap_get_state, donate="none")

    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    if with_pytree:
        set_ = (a, a)
        get_ = (b, b)
    else:
        set_ = a
        get_ = b
    vmap_set_state(index1, set_)
    assert jnp.array_equal(vmap_get_state(index1, get_), set_)

    with pytest.raises(RuntimeError):
        # setting state without vmap, after setting state with vmap
        eqx.experimental.set_state(index1, set_)

    with pytest.raises(RuntimeError):
        # getting state without vmap, after setting state with vmap
        eqx.experimental.get_state(index1, get_)

    eqx.experimental.set_state(index2, set_)

    with pytest.raises(RuntimeError):
        # setting state with vmap, after setting state without vmap
        vmap_set_state(index2, set_)

    with pytest.raises(RuntimeError):
        # getting state with vmap, after setting state without vmap
        vmap_get_state(index2, get_)


@pytest.mark.parametrize("with_jit", (False, True))
@pytest.mark.parametrize("with_pytree", (False, True))
def test_multi_vmap(with_jit, with_pytree):
    index = eqx.experimental.StateIndex()

    @jax.vmap
    @jax.vmap
    def set_state(x):
        eqx.experimental.set_state(index, x)

    @jax.vmap
    @jax.vmap
    def get_state(y):
        return eqx.experimental.get_state(index, y)

    @ft.partial(jax.vmap, in_axes=(1,))
    @ft.partial(jax.vmap, in_axes=(0,))
    def get_state_bad(y):
        return eqx.experimental.get_state(index, y)

    if with_jit:
        set_state = eqx.filter_jit(set_state)
        get_state = eqx.filter_jit(get_state)
        get_state_bad = eqx.filter_jit(get_state_bad)

    a = jnp.array([[1, 2]])
    b = jnp.array([[3, 4]])
    if with_pytree:
        set_ = lambda: (jnp.copy(a), jnp.copy(a))
        get_ = lambda: (jnp.copy(b), jnp.copy(b))
    else:
        set_ = lambda: jnp.copy(a)
        get_ = lambda: jnp.copy(b)
    set_state(set_())
    assert jnp.array_equal(get_state(get_()), set_())

    with pytest.raises(RuntimeError):
        eqx.experimental.get_state(index, get_())

    with pytest.raises(RuntimeError):
        get_state_bad(get_())


def test_inference_not_set_state():
    index = eqx.experimental.StateIndex(inference=True)
    with pytest.raises(RuntimeError):
        eqx.experimental.set_state(index, jnp.array(1))


def test_inference_no_state():
    index = eqx.experimental.StateIndex(inference=True)
    with pytest.raises(RuntimeError):
        eqx.experimental.get_state(index, jnp.array(1))


def test_inference_not_set_under_jit():
    index = eqx.experimental.StateIndex()

    @jax.jit
    def f(i):
        eqx.tree_inference(i, True)

    with pytest.raises(RuntimeError):
        f(index)


def test_inference_can_set():
    index = eqx.experimental.StateIndex()
    eqx.tree_at(lambda i: i.inference, index, True)


def test_inference_no_jit():
    index = eqx.experimental.StateIndex()
    eqx.experimental.set_state(index, jnp.array(1))

    def f(i):
        x = eqx.experimental.get_state(i, jnp.array(0))
        return x + 1

    index_inference = eqx.tree_at(lambda i: i.inference, index, True)
    out = f(index_inference)
    assert jnp.array_equal(out, jnp.array(2))
    eqx.experimental.set_state(index, jnp.array(2))
    out = f(index_inference)
    assert jnp.array_equal(out, jnp.array(3))


def test_inference_jit_input():
    num_jits = 0

    index = eqx.experimental.StateIndex()
    eqx.experimental.set_state(index, jnp.array(1))

    @eqx.filter_jit
    def f(i):
        nonlocal num_jits
        num_jits = num_jits + 1
        x = eqx.experimental.get_state(i, jnp.array(0))
        return x + 1

    index_inference = eqx.tree_at(lambda i: i.inference, index, True)
    out = f(index_inference)
    assert jnp.array_equal(out, jnp.array(2))
    eqx.experimental.set_state(index, jnp.array(2))
    out = f(index_inference)
    assert jnp.array_equal(out, jnp.array(3))

    assert num_jits == 1

    (hlo,) = f.lower(index_inference).compile().runtime_executable().hlo_modules()
    assert "custom-call" not in hlo.to_string()
    # Test the test: just in case the use of custom-call ever changes.
    (hlo,) = f.lower(index).compile().runtime_executable().hlo_modules()
    assert "custom-call" in hlo.to_string()


def test_inference_jit_closure():
    num_jits = 0

    index = eqx.experimental.StateIndex()
    eqx.experimental.set_state(index, jnp.array(1))
    index_inference = eqx.tree_at(lambda i: i.inference, index, True)

    @eqx.filter_jit
    def f():
        nonlocal num_jits
        num_jits = num_jits + 1
        x = eqx.experimental.get_state(index_inference, jnp.array(0))
        return x + 1

    out = f()
    assert jnp.array_equal(out, jnp.array(2))
    eqx.experimental.set_state(index, jnp.array(2))
    out = f()
    # not updated because passed in via closure
    assert jnp.array_equal(out, jnp.array(2))

    assert num_jits == 1

    (hlo,) = f.lower().compile().runtime_executable().hlo_modules()
    assert "custom-call" not in hlo.to_string()

    # Test the test: just in case the use of custom-call ever changes.
    @eqx.filter_jit
    def g():
        x = eqx.experimental.get_state(index, jnp.array(0))
        return x + 1

    (hlo,) = g.lower().compile().runtime_executable().hlo_modules()
    assert "custom-call" in hlo.to_string()


def test_inference_fixed_wrapper():
    class M:
        def __init__(self, value):
            self.value = value

    index = eqx.experimental.StateIndex()
    eqx.experimental.set_state(index, jnp.array(1))

    index_inference = eqx.tree_at(lambda i: i.inference, index, True)
    m = M(index_inference)

    num_jits = 0

    @eqx.filter_jit
    def f(k):
        nonlocal num_jits
        num_jits = num_jits + 1
        return eqx.experimental.get_state(k.value, jnp.array(2))

    assert jnp.array_equal(f(m), jnp.array(1))

    # update ignored; result baked in.
    eqx.experimental.set_state(index, jnp.array(3))
    assert jnp.array_equal(f(m), jnp.array(1))

    assert num_jits == 1


def test_inference_hashable_wrapper():
    class M:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return self.value == other.value

        def __hash__(self):
            return hash(self.value)

    index = eqx.experimental.StateIndex()
    eqx.experimental.set_state(index, jnp.array(1))
    m = M(index)

    @eqx.filter_jit
    def f(k):
        return eqx.experimental.get_state(k.value, jnp.array(2))

    with pytest.raises(ValueError):
        f(m)


@pytest.mark.parametrize("with_jit", (False, True))
@pytest.mark.parametrize("with_pytree", (False, True))
def test_inference_vmap(with_jit, with_pytree):
    index1 = eqx.experimental.StateIndex()
    index2 = eqx.experimental.StateIndex()
    index1_inference = eqx.tree_at(lambda i: i.inference, index1, True)
    index2_inference = eqx.tree_at(lambda i: i.inference, index2, True)

    @ft.partial(jax.vmap, in_axes=(None, 0))
    def vmap_set_state(i, x):
        eqx.experimental.set_state(i, x)

    @ft.partial(jax.vmap, in_axes=(None, 0))
    def vmap_get_state(i, x):
        return eqx.experimental.get_state(i, x)

    if with_jit:
        vmap_set_state = eqx.filter_jit(vmap_set_state)
        vmap_get_state = eqx.filter_jit(vmap_get_state)

    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    if with_pytree:
        set_ = (a, a)
        get_ = (b, b)
    else:
        set_ = a
        get_ = b
    vmap_set_state(index1, set_)
    assert jnp.array_equal(vmap_get_state(index1_inference, get_), set_)

    with pytest.raises(RuntimeError):
        # getting state without vmap, after setting state with vmap
        eqx.experimental.get_state(index1_inference, get_)

    eqx.experimental.set_state(index2, set_)

    with pytest.raises(RuntimeError):
        # getting state with vmap, after setting state without vmap
        vmap_get_state(index2_inference, get_)


@pytest.mark.parametrize("with_jit", (False, True))
@pytest.mark.parametrize("with_pytree", (False, True))
def test_inference_multi_vmap(with_jit, with_pytree):
    index = eqx.experimental.StateIndex()
    index_inference = eqx.tree_at(lambda i: i.inference, index, True)

    @jax.vmap
    @jax.vmap
    def set_state(x):
        eqx.experimental.set_state(index, x)

    @jax.vmap
    @jax.vmap
    def get_state(y):
        return eqx.experimental.get_state(index_inference, y)

    @ft.partial(jax.vmap, in_axes=(1,))
    @ft.partial(jax.vmap, in_axes=(0,))
    def get_state_bad(y):
        return eqx.experimental.get_state(index_inference, y)

    if with_jit:
        set_state = eqx.filter_jit(set_state, donate="none")
        get_state = eqx.filter_jit(get_state, donate="none")
        get_state_bad = eqx.filter_jit(get_state_bad, donate="none")

    a = jnp.array([[1, 2]])
    b = jnp.array([[3, 4]])
    if with_pytree:
        set_ = (a, a)
        get_ = (b, b)
    else:
        set_ = a
        get_ = b
    set_state(set_)
    assert jnp.array_equal(get_state(get_), set_)

    with pytest.raises(RuntimeError):
        eqx.experimental.get_state(index_inference, get_)

    with pytest.raises(RuntimeError):
        get_state_bad(get_)


def test_equality():
    index = eqx.experimental.StateIndex()
    eqx.experimental.set_state(index, jnp.array(1))
    leaves, treedef = jtu.tree_flatten(index)
    index2 = jtu.tree_unflatten(treedef, leaves)
    assert index == index2

    index3 = eqx.tree_at(lambda i: i.inference, index, True)
    assert index != index3
    index4 = eqx.tree_at(lambda i: i.inference, index2, True)
    assert index4 == index3
