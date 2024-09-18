import abc
import dataclasses
import functools as ft
import inspect
from collections.abc import Callable
from dataclasses import InitVar
from typing import Any, Optional

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from .helpers import tree_allclose


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

    assert tree_allclose(MyModule(1.0).field, jnp.array(1.0))  # pyright: ignore

    class MyModuleWithInit(eqx.Module):
        field: jax.Array = eqx.field(converter=jnp.asarray)

        def __init__(self, a):
            self.field = a
            assert type(self.field) is float

    assert tree_allclose(MyModuleWithInit(1.0).field, jnp.array(1.0))

    called = False

    class MyModuleWithPostInit(eqx.Module):
        field: jax.Array = eqx.field(converter=jnp.asarray)

        def __post_init__(self):
            nonlocal called
            assert not called
            called = True
            assert tree_allclose(self.field, jnp.array(1.0))

    MyModuleWithPostInit(1.0)  # pyright: ignore
    assert called


def test_converter_monkeypatched_init():
    class Foo(eqx.Module):
        field: jax.Array = eqx.field(converter=jnp.asarray)

    assert tree_allclose(Foo(1.0).field, jnp.array(1.0))  # pyright: ignore

    called = False
    init = Foo.__init__

    def __init__(self, *args, **kwargs):
        nonlocal called
        assert not called
        called = True
        init(self, *args, **kwargs)

    Foo.__init__ = __init__
    assert tree_allclose(Foo(1.0).field, jnp.array(1.0))  # pyright: ignore
    assert called


# Note that `Foo` had to start with a `__post_init__` method for this to work.
# Dataclasses check for the presence of a `__post_init__` method when the class is
# created, and at that time creates a flag declaring whether to run `__post_init__` at
# initialisation time.
def test_converter_monkeypatched_postinit():
    called1 = False

    class Foo(eqx.Module):
        field: jax.Array = eqx.field(converter=jnp.asarray)

        def __post_init__(self):
            nonlocal called1
            assert not called1
            called1 = True
            assert tree_allclose(self.field, jnp.array(1.0))

    assert tree_allclose(Foo(1.0).field, jnp.array(1.0))  # pyright: ignore
    assert called1

    called2 = False
    post_init = Foo.__post_init__

    def __post_init__(self):
        nonlocal called1
        nonlocal called2
        assert not called2
        called1 = False
        called2 = True
        post_init(self)

    Foo.__post_init__ = __post_init__  # pyright: ignore
    assert tree_allclose(Foo(1.0).field, jnp.array(1.0))  # pyright: ignore
    assert called2
    assert called1


@pytest.mark.parametrize("base_is_module", (False, True))
def test_converter_init_hierarchy(base_is_module):
    class A(eqx.Module if base_is_module else object):  # pyright: ignore
        def __init__(self, x):
            nonlocal called
            assert not called
            called = True
            self.x = x

    class B(eqx.Module):
        x: jax.Array = eqx.field(converter=jnp.asarray)

    class C(A, B):
        # Use `A.__init__`
        pass

    class D(B, A):
        # Use the autogenerated `B.__init__`
        pass

    # In either case, conversion should happen.

    called = False
    assert tree_allclose(C(1).x, jnp.array(1))
    assert called

    assert tree_allclose(D(1).x, jnp.array(1))  # pyright: ignore
    # No `called` check, we're not using `A.__init__`.


@pytest.mark.parametrize("base_is_module", (False, True))
def test_converter_post_init_hierarchy(base_is_module):
    class A(eqx.Module if base_is_module else object):  # pyright: ignore
        def __post_init__(self):
            nonlocal called
            assert not called
            called = True

    class B(eqx.Module):
        x: jax.Array = eqx.field(converter=jnp.asarray)

    class C(A, B):
        pass

    class D(B, A):
        pass

    called = False
    assert tree_allclose(C(1).x, jnp.array(1))  # pyright: ignore
    assert called

    called = False
    assert tree_allclose(D(1).x, jnp.array(1))  # pyright: ignore
    assert called


def test_init_and_postinit():
    class Foo(eqx.Module):
        field: jax.Array = eqx.field(converter=jnp.asarray)

        def __post_init__(self):
            assert False

    with pytest.warns(UserWarning, match="__init__` method and a `__post_init__"):

        class Bar(Foo):
            def __init__(self):
                self.field = 1  # pyright: ignore

    assert tree_allclose(Bar().field, jnp.array(1))

    class Qux(eqx.Module):
        field: jax.Array = eqx.field(converter=jnp.asarray)

        def __init__(self):
            self.field = 1  # pyright: ignore

    with pytest.warns(UserWarning, match="__init__` method and a `__post_init__"):

        class Quux(Qux):
            def __post_init__(self):
                assert False

    assert tree_allclose(Quux().field, jnp.array(1))  # pyright: ignore


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
            return x + 2

    class Concrete3(Concrete2):
        def foo(self, x):
            return x + 3

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

    class Abstract3(eqx.Module, strict=True):
        bar: int

        @abc.abstractmethod
        def foo(self):
            pass


def test_strict_non_module_base():
    class NotAModule:
        pass

    with pytest.raises(
        TypeError, match="Strict `eqx.Module`s must only inherit from other strict"
    ):

        class MyModule(NotAModule, eqx.Module, strict=True):
            pass

    class NotAStrictModule(eqx.Module):
        pass

    with pytest.raises(
        TypeError, match="Strict `eqx.Module`s must only inherit from other strict"
    ):

        class MyModule2(NotAStrictModule, eqx.Module, strict=True):
            pass


def test_strict_concrete_is_final():
    class Concrete(eqx.Module, strict=True):
        pass

    with pytest.raises(
        TypeError, match="very strict `eqx.Module` must be either abstract or final"
    ):

        class Concrete2(Concrete, strict=True):
            pass


@pytest.mark.parametrize("super_init_or_attr", ("init", "attr"))
@pytest.mark.parametrize("sub_init_or_attr", ("init", "attr"))
@pytest.mark.parametrize("abstract_flag", (False, True))
def test_strict_init(super_init_or_attr, sub_init_or_attr, abstract_flag):
    class Abstract(eqx.Module, strict=eqx.StrictConfig(force_abstract=abstract_flag)):
        if super_init_or_attr == "init":

            def __init__(self):
                pass

        else:
            x: int

        if not abstract_flag:

            @abc.abstractmethod
            def foo(self):
                pass

    with pytest.raises(
        TypeError, match="For readability, any custom `__init__` method, and all fields"
    ):

        class Concrete1(Abstract, strict=True):
            if sub_init_or_attr == "init":

                def __init__(self):
                    pass

            else:
                y: str

            def foo(self):
                pass


def test_strict_init_in_abstract():
    class AbstractA(eqx.Module):
        a: int
        b: int

        def __init__(self, x):
            self.a = x
            self.b = x

        @abc.abstractmethod
        def foo(self):
            pass

    class ConcreteB(AbstractA):
        def foo(self):
            pass

    # No way to teach pyright about Equinox's different behaviour for `__init__` than
    # is provided by dataclasses.
    instance = ConcreteB(x=1)  # pyright: ignore
    assert instance.a == 1
    assert instance.b == 1


def test_strict_init_transitive():
    class AbstractA(eqx.Module, strict=eqx.StrictConfig(force_abstract=True)):
        x: int

    class AbstractB(AbstractA, strict=eqx.StrictConfig(force_abstract=True)):
        pass

    with pytest.raises(
        TypeError, match="For readability, any custom `__init__` method, and all fields"
    ):

        class C(AbstractB, strict=True):
            y: str


def test_strict_abstract_name():
    class Abstract(eqx.Module, strict=eqx.StrictConfig(force_abstract=True)):
        pass

    class _Abstract(eqx.Module, strict=eqx.StrictConfig(force_abstract=True)):
        pass

    with pytest.raises(
        TypeError, match="Abstract strict `eqx.Module`s must be named starting with"
    ):

        class NotAbstract(eqx.Module, strict=eqx.StrictConfig(force_abstract=True)):
            pass


def test_strict_method_reoverride():
    class AbstractA(eqx.Module, strict=True):
        @abc.abstractmethod
        def foo(self, x):
            pass

    class AbstractB(AbstractA, strict=True):
        def foo(self, x):
            pass

        @abc.abstractmethod
        def bar(self, x):
            pass

    with pytest.raises(
        TypeError, match="Strict `eqx.Module`s cannot override concrete"
    ):

        class C(AbstractB, strict=True):
            def foo(self, x):
                pass

            def bar(self, x):
                pass

    class AbstractD(eqx.Module, strict=eqx.StrictConfig(force_abstract=True)):
        foo = 4

    class AbstractD2(AbstractD, strict=eqx.StrictConfig(force_abstract=True)):
        pass

    with pytest.raises(
        TypeError, match="Strict `eqx.Module`s cannot override non-methods"
    ):

        class E(AbstractD, strict=True):
            def foo(self):
                pass

    with pytest.raises(
        TypeError, match="Strict `eqx.Module`s cannot override non-methods"
    ):

        class E2(AbstractD2, strict=True):
            def foo(self):
                pass


def test_strict_default():
    class AbstractA(eqx.Module, strict=eqx.StrictConfig(force_abstract=True)):
        def foo(self) -> int:
            return 4

    class AbstractB(
        AbstractA,
        strict=eqx.StrictConfig(force_abstract=True, allow_method_override=True),
    ):
        def foo(self) -> int:
            return 5

    with pytest.raises(
        TypeError, match="Strict `eqx.Module`s cannot override concrete"
    ):

        class C1(AbstractB, strict=True):
            def foo(self):
                return 6

    class C2(AbstractB, strict=eqx.StrictConfig(allow_method_override=True)):
        def foo(self):
            return 7


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


def test_inherit_doc():
    # This is not what dataclasses do by default -- they would set `B.__init__.__doc__`
    # to `None`

    class A(eqx.Module):
        pass

    A.__init__.__doc__ = "Hey there!"

    class B(A):
        pass

    assert B.__init__.__doc__ == "Hey there!"


@pytest.mark.parametrize("a_post_init", (False, True))
@pytest.mark.parametrize("b_post_init", (False, True))
@pytest.mark.parametrize("c_post_init", (False, True))
def test_conversion_once(a_post_init, b_post_init, c_post_init):
    def converter(x):
        nonlocal called
        assert not called
        called = True
        return x

    class A(eqx.Module):
        x: int = eqx.field(converter=converter)

        if a_post_init:

            def __post_init__(self):
                pass

    class B(A):
        if b_post_init:

            def __post_init__(self):
                pass

    class C(B):
        if c_post_init:

            def __post_init__(self):
                pass

    called = False
    A(1)
    assert called

    called = False
    B(1)
    assert called

    called = False
    C(1)
    assert called


def test_init_fields():
    class A(eqx.Module):
        x: int = eqx.field(init=False)

    with pytest.raises(ValueError, match="The following fields were not initialised"):
        A()

    class B(eqx.Module):
        x: int = eqx.field(init=False)

        def __post_init__(self):
            pass

    with pytest.raises(ValueError, match="The following fields were not initialised"):
        B()

    class C(eqx.Module):
        x: int = eqx.field(init=False)
        flag: InitVar[bool]

        def __post_init__(self, flag):
            if flag:
                self.x = 1

    C(flag=True)
    with pytest.raises(ValueError, match="The following fields were not initialised"):
        C(flag=False)


@pytest.mark.parametrize("field", (dataclasses.field, eqx.field))
def test_init_as_abstract(field):
    # Before the introduction of AbstractVar, it was possible to sort-of get the same
    # behaviour by marking it as an `init=False` field. Here we check that we don't
    # break that, in particular when it's overridden by a non-field.

    class Abstract(eqx.Module):
        foo: int = field(init=False)

    class Concrete(Abstract):
        @property
        def foo(self):
            return 1

    x = Concrete()
    leaves, treedef = jtu.tree_flatten(x)
    assert len(leaves) == 0
    y = jtu.tree_unflatten(treedef, leaves)
    assert y.foo == 1


# https://github.com/patrick-kidger/equinox/issues/522
def test_custom_field():
    def my_field(*, foo: Optional[bool] = None, **kwargs: Any):
        metadata = kwargs.pop("metadata", {})
        if foo is not None:
            metadata["foo"] = foo
        return eqx.field(metadata=metadata, **kwargs)

    class ExampleModel(eqx.Module):
        dynamic: jax.Array = my_field(foo=True)
        static: int = my_field(foo=False, static=True)

    model = ExampleModel(dynamic=jnp.array(1), static=1)
    dynamic_field, static_field = dataclasses.fields(model)
    assert dynamic_field.metadata == dict(foo=True)
    assert static_field.metadata == dict(foo=False, static=True)


def signature_test_cases():
    @dataclasses.dataclass
    class FooDataClass:
        a: int

    class FooModule(eqx.Module):
        a: int

    @dataclasses.dataclass
    class CustomInitDataClass:
        def __init__(self, a: int):
            pass

    class CustomInitModule(eqx.Module):
        def __init__(self, a: int):
            pass

    @dataclasses.dataclass
    class CallableDataClass:
        a: int

        def __call__(self, b: int):
            pass

    class CallableModule(eqx.Module):
        a: int

        def __call__(self, b: int):
            pass

    test_cases = [
        (FooDataClass, FooModule),
        (CustomInitDataClass, CustomInitModule),
        (CallableDataClass, CallableModule),
        (CallableDataClass(1), CallableModule(1)),
    ]
    return test_cases


@pytest.mark.parametrize(("dataclass", "module"), signature_test_cases())
def test_signature(dataclass, module):
    # Check module signature matches dataclass signatures.
    assert inspect.signature(dataclass) == inspect.signature(module)


def test_module_setattr():
    class Foo(eqx.Module):
        def f(self):
            pass

    def f2(self):
        pass

    def g(self):
        pass

    Foo.f = f2
    Foo.g = g  # pyright: ignore
    assert Foo.f is f2
    assert Foo.g is g  # pyright: ignore
    assert type(Foo.__dict__["f"]).__name__ == "_wrap_method"
    assert type(Foo.__dict__["g"]).__name__ == "_wrap_method"


# See https://github.com/patrick-kidger/equinox/issues/206
def test_jax_transform_warn(getkey):
    class A(eqx.Module):
        linear: Callable

    class B(eqx.Module):
        linear: Callable

        def __init__(self, linear):
            self.linear = linear

    for cls in (A, B):
        for transform in (
            jax.jit,
            jax.grad,
            jax.vmap,
            jax.value_and_grad,
            jax.jacfwd,
            jax.jacrev,
            jax.hessian,
            jax.custom_jvp,
            jax.custom_vjp,
            jax.checkpoint,  # pyright: ignore
            jax.pmap,
        ):
            with pytest.warns(
                match="Possibly assigning a JAX-transformed callable as an attribute"
            ):
                transformed = transform(eqx.nn.Linear(2, 2, key=getkey()))
                cls(transformed)


def test_converter_annotations():
    def converter1(x):
        return x

    def converter2(x: int):
        return "hi"

    def converter3(x: bool, y=1):
        return "bye"

    class Foo1(eqx.Module):
        x: str = eqx.field(converter=converter1)

    class Foo2(eqx.Module):
        x: str = eqx.field(converter=converter2)

    class Foo3(eqx.Module):
        x: str = eqx.field(converter=converter3)

    assert Foo1.__init__.__annotations__["x"] is Any
    assert Foo2.__init__.__annotations__["x"] is int
    assert Foo3.__init__.__annotations__["x"] is bool


def test_no_jax_array_static():
    class Valid(eqx.Module):
        a: tuple
        b: jax.Array

    class InvalidTuple(eqx.Module):
        a: tuple = eqx.field(static=True)
        b: jax.Array

    class InvalidArr(eqx.Module):
        a: tuple
        b: jax.Array = eqx.field(static=True)

    Valid((), jnp.ones(2))

    with pytest.warns(
        UserWarning,
        match="A JAX array is being set as static!",
    ):
        InvalidTuple((jnp.ones(10),), jnp.ones(10))

    with pytest.warns(
        UserWarning,
        match="A JAX array is being set as static!",
    ):
        InvalidArr((), jnp.ones(10))


# https://github.com/patrick-kidger/equinox/issues/832
def test_cooperative_multiple_inheritance():
    called_a = False
    called_b = False
    called_d = False

    class A(eqx.Module):
        def __post_init__(self) -> None:
            nonlocal called_a
            called_a = True

    class B(A):
        def __post_init__(self) -> None:
            nonlocal called_b
            called_b = True
            super().__post_init__()

    class C(A):
        pass

    class D(C, A):
        def __post_init__(self) -> None:
            nonlocal called_d
            called_d = True
            super().__post_init__()

    class E(D, B):
        pass

    E()
    assert called_a
    assert called_b
    assert called_d


# https://github.com/patrick-kidger/equinox/issues/858
def test_init_subclass_and_abstract_class_var():
    class Parent(eqx.Module):
        abs_cls_var: eqx.AbstractClassVar[str]

        def __init__(self):
            pass

        def __init_subclass__(cls):
            cls.abs_cls_var

    class Child(Parent):
        abs_cls_var = "foo"

    Child()  # pyright: ignore[reportCallIssue]
