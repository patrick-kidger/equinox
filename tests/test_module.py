from typing import Any

import jax.tree_util as jtu
import pytest

import equinox as eqx


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


def test_static_field():
    class MyModule(eqx.Module):
        field1: int
        field2: int = eqx.static_field()
        field3: int = eqx.static_field(default=3)

    m = MyModule(1, 2)
    flat, treedef = jtu.tree_flatten(m)
    assert len(flat) == 1
    assert flat[0] == 1
    rm = jtu.tree_unflatten(treedef, flat)
    assert rm.field1 == 1
    assert rm.field2 == 2
    assert rm.field3 == 3


def test_wrap_method():
    class MyModule(eqx.Module):
        a: int

        def f(self, b):
            return self.a + b

    m = MyModule(13)
    assert isinstance(m.f, jtu.Partial)
    flat, treedef = jtu.tree_flatten(m.f)
    assert len(flat) == 1
    assert flat[0] == 13
    assert jtu.tree_unflatten(treedef, flat)(2) == 15


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
    def f(x):
        pass

    fjit = eqx.filter_jit(f)
    # Gets __name__ attribute from module_update_wrapper

    @eqx.filter_jit  # Flattens and unflattens
    def g(k):
        k.__name__

    g(fjit)


# https://github.com/patrick-kidger/equinox/issues/337
def test_subclass_static():
    class A(eqx.Module):
        foo: int = eqx.static_field()

    class B(A):
        pass

    b = B(1)
    assert len(jtu.tree_leaves(b)) == 0


def test_flatten_with_keys():
    class A(eqx.Module):
        foo: int
        bar: int = eqx.static_field()
        qux: list

    a = A(1, 2, [3.0])
    leaves, metadata = jtu.tree_flatten_with_path(a)
    ((path1,), value1), ((path2a, path2b), value2) = leaves
    assert value1 == 1
    assert value2 == 3.0
    assert isinstance(path1, jtu.GetAttrKey) and path1.name == "foo"
    assert isinstance(path2a, jtu.GetAttrKey) and path2a.name == "qux"
    assert isinstance(path2b, jtu.SequenceKey) and path2b.idx == 0
