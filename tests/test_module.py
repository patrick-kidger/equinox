import abc
import dataclasses
import functools as ft
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import equinox as eqx
import equinox.internal as eqxi

from .helpers import shaped_allclose


def test_module_not_enough_attributes():
    class MyModule1(eqx.Module):
        weight: Any

    with pytest.raises(TypeError):
        MyModule1()  # pyright: ignore

    class MyModule2(eqx.Module):
        weight: Any

        def __init__(self) -> None:
            pass

    with pytest.raises(ValueError):
        MyModule2()
    with pytest.raises(TypeError):
        MyModule2(1)  # pyright: ignore


def test_module_too_many_attributes():
    class MyModule1(eqx.Module):
        weight: Any

    with pytest.raises(TypeError):
        MyModule1(1, 2)  # pyright: ignore

    class MyModule2(eqx.Module):
        weight: Any

        def __init__(self, weight: Any):
            self.weight = weight
            self.something_else = True

    with pytest.raises(AttributeError):
        MyModule2(1)


def test_module_setattr_after_init():
    class MyModule(eqx.Module):
        weight: Any

    m = MyModule(1)
    with pytest.raises(AttributeError):
        m.asdf = True  # pyright: ignore


def test_wrong_attribute():
    class MyModule(eqx.Module):
        weight: Any

        def __init__(self, value: Any):
            self.not_weight = value

    with pytest.raises(AttributeError):
        MyModule(1)


# The main part of this test is to check that __init__ works correctly.
def test_inheritance():
    # no custom init / no custom init

    class MyModule(eqx.Module):
        weight: Any

    class MyModule2(MyModule):
        weight2: Any

    m = MyModule2(1, 2)
    assert m.weight == 1
    assert m.weight2 == 2
    m = MyModule2(1, weight2=2)
    assert m.weight == 1
    assert m.weight2 == 2
    m = MyModule2(weight=1, weight2=2)
    assert m.weight == 1
    assert m.weight2 == 2
    with pytest.raises(TypeError):
        m = MyModule2(2, weight=2)  # pyright: ignore

    # not custom init / custom init

    class MyModule3(MyModule):
        weight3: Any

        def __init__(self, *, weight3: Any, **kwargs):
            self.weight3 = weight3
            super().__init__(**kwargs)

    m = MyModule3(weight=1, weight3=3)
    assert m.weight == 1
    assert m.weight3 == 3

    # custom init / no custom init

    class MyModule4(eqx.Module):
        weight4: Any

        def __init__(self, value4: Any, **kwargs):
            self.weight4 = value4
            super().__init__(**kwargs)

    class MyModule5(MyModule4):
        weight5: Any

    with pytest.raises(TypeError):
        m = MyModule5(value4=1, weight5=2)  # pyright: ignore

    class MyModule6(MyModule4):
        pass

    m = MyModule6(value4=1)  # pyright: ignore
    assert m.weight4 == 1

    # custom init / custom init

    class MyModule7(MyModule4):
        weight7: Any

        def __init__(self, value7: Any, **kwargs):
            self.weight7 = value7
            super().__init__(**kwargs)

    m = MyModule7(value4=1, value7=2)
    assert m.weight4 == 1
    assert m.weight7 == 2


# This used to be allowed, but was frankly very sketchy -- it introduced cycles into
# things. We now explicitly try to catch this as an error.
def test_method_assignment():
    class A(eqx.Module):
        foo: Callable
        x: jax.Array

        def __init__(self, x):
            # foo is assigned before x! We have that `self.bar.__self__` is a copy of
            # `self`, but for which self.bar.__self__.x` doesn't exist yet. Then later
            # calling `self.foo()` would raise an error.
            self.foo = self.bar
            self.x = x

        def bar(self):
            return self.x + 1

    x = jnp.array(3)
    with pytest.raises(ValueError, match="Cannot assign methods in __init__"):
        A(x)


def test_method_assignment2():
    # From https://github.com/patrick-kidger/equinox/issues/327
    class Component(eqx.Module):
        transform: Callable[[float], float]
        validator: Callable[[Callable], Callable]

        def __init__(self, transform=lambda x: x, validator=lambda f: f) -> None:
            self.transform = transform
            self.validator = validator

        def __call__(self, x):
            return self.validator(self.transform)(x)

    class SubComponent(Component):
        test: Callable[[float], float]

        def __init__(self, test=lambda x: 2 * x) -> None:
            self.test = test
            super().__init__(self._transform)

        def _transform(self, x):
            return self.test(x)

    with pytest.raises(ValueError, match="Cannot assign methods in __init__"):
        SubComponent()


def test_method_access_during_init():
    class Foo(eqx.Module):
        def __init__(self):
            self.method()

        def method(self):
            pass

    Foo()


@pytest.mark.parametrize("new", (False, True))
def test_static_field(new):
    if new:
        static_field = ft.partial(eqx.field, static=True)
    else:
        static_field = eqx.static_field

    class MyModule(eqx.Module):
        field1: int
        field2: int = static_field()
        field3: int = static_field(default=3)

    m = MyModule(1, 2)
    flat, treedef = jtu.tree_flatten(m)
    assert len(flat) == 1
    assert flat[0] == 1
    rm = jtu.tree_unflatten(treedef, flat)
    assert rm.field1 == 1
    assert rm.field2 == 2
    assert rm.field3 == 3


def test_converter():
    class MyModule(eqx.Module):
        field: jax.Array = eqx.field(converter=jnp.asarray)

    assert shaped_allclose(MyModule(1.0).field, jnp.array(1.0))  # pyright: ignore

    class MyModuleWithInit(eqx.Module):
        field: jax.Array = eqx.field(converter=jnp.asarray)

        def __init__(self, a):
            self.field = a

    assert shaped_allclose(MyModule(1.0).field, jnp.array(1.0))  # pyright: ignore


def test_wrap_method():
    class MyModule(eqx.Module):
        a: int

        def f(self, b):
            return self.a + b

    m = MyModule(13)
    assert isinstance(m.f, eqx.Module)
    flat, treedef = jtu.tree_flatten(m.f)
    assert len(flat) == 1
    assert flat[0] == 13
    assert jtu.tree_unflatten(treedef, flat)(2) == 15


def test_eq_method():
    # Expected behaviour from non-Module methods
    class A:
        def f(self):
            pass

    a = A()
    assert a.f == a.f

    class B(eqx.Module):
        def f(self):
            pass

    assert B().f == B().f


def test_init_subclass():
    ran = []

    class MyModule(eqx.Module):
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            ran.append(True)

    class AnotherModule(MyModule):
        pass

    assert ran == [True]


def test_wrapper_attributes():
    def f(x: int) -> str:
        """some doc"""
        return "hi"

    def h(x: int) -> str:
        """some other doc"""
        return "bye"

    noinline = lambda x: eqxi.noinline(h, abstract_fn=x)

    for wrapper in (
        eqx.filter_jit,
        eqx.filter_grad,
        eqx.filter_value_and_grad,
        eqx.filter_vmap,
        eqx.filter_pmap,
        noinline,
    ):
        f_wrap = wrapper(f)
        # Gets __name__ attribute from module_update_wrapper

        called = False

        @eqx.filter_jit  # Flattens and unflattens
        def g(k):
            nonlocal called
            called = True
            assert k.__name__ == "f"
            assert k.__doc__ == "some doc"
            assert k.__qualname__ == "test_wrapper_attributes.<locals>.f"
            assert k.__annotations__ == {"x": int, "return": str}

        g(f_wrap)
        assert called


# https://github.com/patrick-kidger/equinox/issues/337
def test_subclass_static():
    class A(eqx.Module):
        foo: int = eqx.field(static=True)

    class B(A):
        pass

    b = B(1)
    assert len(jtu.tree_leaves(b)) == 0


def test_flatten_with_keys():
    class A(eqx.Module):
        foo: int
        bar: int = eqx.field(static=True)
        qux: list

    a = A(1, 2, [3.0])
    leaves, metadata = jtu.tree_flatten_with_path(a)
    ((path1,), value1), ((path2a, path2b), value2) = leaves
    assert value1 == 1
    assert value2 == 3.0
    assert isinstance(path1, jtu.GetAttrKey) and path1.name == "foo"
    assert isinstance(path2a, jtu.GetAttrKey) and path2a.name == "qux"
    assert isinstance(path2b, jtu.SequenceKey) and path2b.idx == 0


def test_wrapped(getkey):
    # https://github.com/patrick-kidger/equinox/issues/377
    x = eqx.filter_vmap(eqx.nn.Linear(2, 2, key=getkey()))
    y = eqx.filter_vmap(eqx.nn.Linear(2, 2, key=getkey()))
    x, y = eqx.filter((x, y), eqx.is_array)
    jtu.tree_map(lambda x, y: x + y, x, y)


def test_class_creation_kwargs():
    called = False

    class A(eqx.Module):
        def __init_subclass__(cls, foo, **kwargs) -> None:
            nonlocal called
            assert not called
            called = True
            assert foo
            assert len(kwargs) == 0

    class B(A, foo=True):
        pass

    assert called


def test_check_init():
    class FooException(Exception):
        pass

    called_a = False
    called_b = False

    class A(eqx.Module):
        a: int

        def __check_init__(self):
            nonlocal called_a
            called_a = True
            if self.a >= 0:
                raise FooException

    class B(A):
        def __check_init__(self):
            nonlocal called_b
            called_b = True

    class C(A):
        pass

    assert not called_a
    assert not called_b
    A(-1)
    assert called_a
    assert not called_b

    called_a = False
    with pytest.raises(FooException):
        A(1)
    assert called_a
    assert not called_b

    called_a = False
    B(-1)
    assert called_a
    assert called_b

    called_a = False
    called_b = False
    with pytest.raises(FooException):
        B(1)
    assert called_a
    assert called_b  # B.__check_init__ is called before A.__check_init__

    called_a = False
    called_b = False
    C(-1)
    assert called_a
    assert not called_b

    called_a = False
    with pytest.raises(FooException):
        C(1)
    assert called_a
    assert not called_b


def test_check_init_order():
    called_a = False
    called_b = False
    called_c = False

    class A(eqx.Module):
        def __check_init__(self):
            nonlocal called_a
            called_a = True

    class B(A):
        def __check_init__(self):
            nonlocal called_b
            called_b = True
            raise ValueError

    class C(B):
        def __check_init__(self):  # pyright: ignore
            nonlocal called_c
            called_c = True

    with pytest.raises(ValueError):
        C()

    assert called_c
    assert called_b
    assert not called_a


def test_check_init_no_assignment():
    class A(eqx.Module):
        x: int

        def __check_init__(self):
            self.x = 4

    with pytest.raises(dataclasses.FrozenInstanceError):
        A(1)


def test_strict_noerrors():
    class Abstract(eqx.Module, strict=True):
        @abc.abstractmethod
        def foo(self, x):
            pass

    class Concrete1(Abstract, strict=True):
        def foo(self, x):
            return x + 1

    class Concrete2(Abstract):
        def foo(self, x):
            return x + 1


def test_strict_non_module_base():
    class NotAModule:
        pass

    with pytest.raises(TypeError, match="subclasses of `eqx.Module`"):

        class MyModule(eqx.Module, NotAModule, strict=True):
            pass


def test_strict_method_reoverride():
    class AbstractA(eqx.Module, strict=True):
        @abc.abstractmethod
        def foo(self, x):
            pass

    class B(AbstractA, strict=True):
        def foo(self, x):
            pass

    with pytest.raises(TypeError, match="concrete methods"):

        class C(B, strict=True):
            def foo(self, x):
                pass


def test_strict_init():
    with pytest.raises(TypeError, match="__init__"):

        class Abstract(eqx.Module, strict=True):
            def __init__(self):
                pass

            @abc.abstractmethod
            def foo(self):
                pass


def test_strict_fields():
    class Abstract1(eqx.Module, strict=True):
        bar: eqx.AbstractVar[int]

        @abc.abstractmethod
        def foo(self):
            pass

    class Abstract2(eqx.Module, strict=True):
        bar: eqx.AbstractClassVar[int]

        @abc.abstractmethod
        def foo(self):
            pass

    with pytest.raises(TypeError, match="fields"):

        class Abstract3(eqx.Module, strict=True):
            bar: int

            @abc.abstractmethod
            def foo(self):
                pass


def test_post_init_warning():
    class A(eqx.Module):
        called = False

        def __post_init__(self):
            type(self).called = True

    with pytest.warns(
        UserWarning, match="test_module.test_post_init_warning.<locals>.B"
    ):

        class B(A):
            def __init__(self):
                pass

    with pytest.warns(
        UserWarning, match="test_module.test_post_init_warning.<locals>.C"
    ):

        class C(B):
            pass

    B()
    C()
    assert not A.called
