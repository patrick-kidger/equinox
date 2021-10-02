from typing import Any

import jax
import pytest

import equinox as eqx


def test_module_not_enough_attributes():
    class MyModule1(eqx.Module):
        weight: Any

    with pytest.raises(TypeError):
        MyModule1()

    class MyModule2(eqx.Module):
        weight: Any

        def __init__(self):
            pass

    with pytest.raises(ValueError):
        MyModule2()
    with pytest.raises(TypeError):
        MyModule2(1)


def test_module_too_many_attributes():
    class MyModule1(eqx.Module):
        weight: Any

    with pytest.raises(TypeError):
        MyModule1(1, 2)

    class MyModule2(eqx.Module):
        weight: Any

        def __init__(self, weight):
            self.weight = weight
            self.something_else = True

    with pytest.raises(AttributeError):
        MyModule2(1)


def test_module_setattr_after_init():
    class MyModule(eqx.Module):
        weight: Any

    m = MyModule(1)
    with pytest.raises(AttributeError):
        m.asdf = True


def test_wrong_attribute():
    class MyModule(eqx.Module):
        weight: Any

        def __init__(self, value):
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
        m = MyModule2(2, weight=2)

    # not custom init / custom init

    class MyModule3(MyModule):
        weight3: Any

        def __init__(self, *, weight3, **kwargs):
            self.weight3 = weight3
            super().__init__(**kwargs)

    m = MyModule3(weight=1, weight3=3)
    assert m.weight == 1
    assert m.weight3 == 3

    # custom init / no custom init

    class MyModule4(eqx.Module):
        weight4: Any

        def __init__(self, value4, **kwargs):
            self.weight4 = value4
            super().__init__(**kwargs)

    class MyModule5(MyModule4):
        weight5: Any

    with pytest.raises(TypeError):
        m = MyModule5(value4=1, weight5=2)

    class MyModule6(MyModule4):
        pass

    m = MyModule6(value4=1)
    assert m.weight4 == 1

    # custom init / custom init

    class MyModule7(MyModule4):
        weight7: Any

        def __init__(self, value7, **kwargs):
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
    flat, treedef = jax.tree_flatten(m)
    assert len(flat) == 1
    assert flat[0] == 1
    rm = jax.tree_unflatten(treedef, flat)
    assert rm.field1 == 1
    assert rm.field2 == 2
    assert rm.field3 == 3


def test_wrap_method():
    class MyModule(eqx.Module):
        a: int

        def f(self, b):
            return self.a + b

    m = MyModule(13)
    assert isinstance(m.f, jax.tree_util.Partial)
    flat, treedef = jax.tree_flatten(m.f)
    assert len(flat) == 1
    assert flat[0] == 13
    assert jax.tree_unflatten(treedef, flat)(2) == 15


def test_init_subclass():
    ran = []

    class MyModule(eqx.Module):
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            ran.append(True)

    class AnotherModule(MyModule):
        pass

    assert ran == [True]
