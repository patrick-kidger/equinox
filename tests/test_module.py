import equinox as eqx
import pytest
from typing import Any


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
