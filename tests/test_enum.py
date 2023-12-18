import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from .helpers import tree_allclose


def test_equality():
    class A(eqxi.Enumeration):
        x = "foo"
        y = "bar"

    class B(eqxi.Enumeration):
        z = "qux"

    assert A.x == A.x
    assert A.x != A.y
    with pytest.raises(ValueError):
        A.x == 0  # pyright: ignore
    with pytest.raises(ValueError):
        A.x == B.z  # pyright: ignore
    assert not A.x.is_traced()

    @jax.jit
    def run1():
        return A.x == A.y

    @jax.jit
    def run2():
        return A.x == A.x

    @jax.jit
    def run3(a):
        assert a.is_traced()
        return a == A.x

    assert tree_allclose(run1(), jnp.array(False))
    assert tree_allclose(run2(), jnp.array(True))
    assert tree_allclose(run3(A.x), jnp.array(True))
    assert tree_allclose(run3(A.y), jnp.array(False))


def test_where():
    class A(eqxi.Enumeration):
        x = "foo"
        y = "bar"

    class B(eqxi.Enumeration):
        z = "qux"

    class C(A):  # pyright: ignore
        w = "quux"

    assert A.where(True, A.x, A.y) == A.x
    assert A.where(False, A.x, A.y) == A.y

    with pytest.raises(ValueError):
        A.where(True, A.x, B.z)  # pyright: ignore

    with pytest.raises(ValueError):
        A.where(True, A.x, C.w)

    @jax.jit
    def run(pred, a):
        return A.where(pred, a, A.x)

    true = jnp.array(True)
    false = jnp.array(False)
    assert run(true, A.x) == A.x
    assert run(false, A.x) == A.x
    assert run(true, A.y) == A.y
    assert run(false, A.y) == A.x


def test_repr():
    class A(eqxi.Enumeration):
        x = "foo"

    assert str(A.x).endswith("test_enum.test_repr.<locals>.A<foo>")

    @jax.jit
    def run(a):
        assert str(a).endswith("test_enum.test_repr.<locals>.A<traced>")

    run(A.x)


def test_inheritance_and_len():
    class A(eqxi.Enumeration):
        a = "foo"

    class B(eqxi.Enumeration):
        b = "bar"

    class C1(A, B):  # pyright: ignore
        c = "qux"

    class C2(B, A):  # pyright: ignore
        c = "qux"

    class D1(C1, A):  # pyright: ignore
        d = "quux"

    class D2(C2, A):  # pyright: ignore
        d = "quux"

    with pytest.raises(Exception):

        class D3(A, C1):  # pyright: ignore
            d = "quux"

    with pytest.raises(Exception):

        class D4(A, C2):  # pyright: ignore
            d = "quux"

    A.a
    B.b
    C1.a
    C1.b
    C1.c
    C2.a
    C2.b
    C2.c
    D1.a
    D1.b
    D1.c
    D1.d
    D2.a
    D2.b
    D2.c
    D2.d

    with pytest.raises(ValueError):
        D1.a == A.a  # pyright: ignore

    with pytest.raises(ValueError):
        D1.d == A.a  # pyright: ignore

    assert C1.promote(A.a) == C1.a
    assert C2.promote(A.a) == C2.a
    assert C1.promote(B.b) == C1.b
    assert C2.promote(B.b) == C2.b

    with pytest.raises(ValueError):
        C1.promote(C1.c)

    with pytest.raises(ValueError):
        C1.promote(C2.c)

    assert D1.promote(A.a) == D1.a
    assert D1.promote(B.b) == D1.b
    assert D1.promote(C1.a) == D1.a
    assert D1.promote(C1.b) == D1.b
    assert D1.promote(C1.c) == D1.c

    assert D2.promote(A.a) == D2.a
    assert D2.promote(B.b) == D2.b
    assert D2.promote(C2.a) == D2.a
    assert D2.promote(C2.b) == D2.b
    assert D2.promote(C2.c) == D2.c

    assert C1.promote(B.b) != C1.a
    assert D2.promote(A.a) != D2.b

    assert len(A) == 1
    assert len(B) == 1
    assert len(C1) == 3
    assert len(C2) == 3
    assert len(D1) == 4
    assert len(D2) == 4


def test_inheritance2():
    class A(eqxi.Enumeration):
        a = "a"
        b = "b"

    class B(eqxi.Enumeration):
        a = "a"
        c = "c"

    class C(A, B):  # pyright: ignore
        pass

    assert C.promote(A.a) == C.a
    assert C.promote(A.b) == C.b
    assert C.promote(B.a) == C.a
    assert C.promote(B.c) == C.c


def test_getitem():
    class A(eqxi.Enumeration):
        a = "hi"

    class B(eqxi.Enumeration):
        b = "bye"

    A[A.a]
    with pytest.raises(ValueError):
        A[0]
    with pytest.raises(ValueError):
        A[B.b]


def test_isinstance():
    class A(eqxi.Enumeration):
        a = "hi"

    class B(eqxi.Enumeration):
        b = "bye"

    assert isinstance(A.a, A)
    assert not isinstance(0, A)
    assert not isinstance(B.b, A)


def test_duplicate_fields():
    class A(eqxi.Enumeration):
        a = "hi"

    class B(eqxi.Enumeration):
        a = "hi"

    class C(A, B):  # pyright: ignore
        pass

    class D(A):  # pyright: ignore
        a = "hi"

    with pytest.warns():

        class E(A):  # pyright: ignore
            a = "not hi"

        assert E[E.a] == "not hi"

    with pytest.warns():

        class F(A, B):  # pyright: ignore
            a = "not hi"

        assert F[F.a] == "not hi"


def test_error_if():
    class A(eqxi.Enumeration):
        a = "hi"

    token = jnp.array(True)
    A.a.error_if(token, False)
    eqx.filter_jit(A.a.error_if)(token, False)
    with pytest.raises(Exception):
        A.a.error_if(token, True)
    with pytest.raises(Exception):
        eqx.filter_jit(A.a.error_if)(token, True)


def test_compile_time_eval():
    class A(eqxi.Enumeration):
        a = "hi"
        b = "bye"

    @jax.jit
    def f(pred):
        x = A.where(True, A.a, A.b)
        assert not x.is_traced()
        y = x == A.a
        assert y
        z = A.where(pred, A.a, A.b)
        assert z.is_traced()

    f(True)


def test_where_traced_bool_same_branches():
    class A(eqxi.Enumeration):
        a = "hi"
        b = "bye"

    @jax.jit
    def f(pred, foo):
        leaves, treedef = jtu.tree_flatten(foo)
        bar = jtu.tree_unflatten(treedef, leaves)
        out = A.where(pred, foo, bar)
        assert out is foo

    f(True, A.a)
