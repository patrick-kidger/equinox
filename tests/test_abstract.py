import abc
from typing import ClassVar, TYPE_CHECKING

import pytest

import equinox as eqx


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar, ClassVar as AbstractVar
else:
    from equinox.internal import AbstractClassVar, AbstractVar


def test_abstract_method():
    class A(eqx.Module):
        @abc.abstractmethod
        def x(self):
            ...

    with pytest.raises(TypeError, match="abstract method"):
        A()


def test_abstract_attribute():
    class A(eqx.Module):
        x: AbstractVar[bool]

    with pytest.raises(TypeError, match="abstract attributes"):
        A()

    class B(eqx.Module):
        x: "AbstractVar[bool]"

    with pytest.raises(TypeError, match="abstract attributes"):
        B()

    class C(A):
        y: int

    with pytest.raises(TypeError, match="abstract attributes"):
        C(y=2)

    class D(A):
        x: bool
        y: str

    D(x=True, y="hi")

    class E(A):
        y: str
        x: bool  # different order

    e = E("hi", True)
    assert e.x is True
    assert e.y == "hi"

    class F(A):
        y: str

        @property
        def x(self):
            return True

    f = F(y="hi")
    assert f.x is True
    assert f.y == "hi"

    with pytest.raises(TypeError, match="unsubscripted"):

        class G(eqx.Module):
            x: AbstractVar

    with pytest.raises(TypeError, match="mismatched type annotations"):

        class H(A):
            x: str

    class I(A):  # noqa: E742
        x: AbstractVar[bool]

    class J(A):
        x: AbstractClassVar[bool]

    class K(A):
        x: bool

    with pytest.raises(TypeError, match="cannot have value"):

        class L(eqx.Module):
            x: AbstractVar[bool] = True

    class M1(A):
        x = True
        y: bool = False

    class M2(A):
        x = True


def test_abstract_class_attribute():
    class A(eqx.Module):
        x: AbstractClassVar[bool]

    with pytest.raises(TypeError, match="abstract class attributes"):
        A()

    class B(eqx.Module):
        x: "AbstractClassVar[bool]"

    with pytest.raises(TypeError, match="abstract class attributes"):
        B()

    class C(A):
        y: int

    with pytest.raises(TypeError, match="abstract class attributes"):
        C(y=2)

    with pytest.raises(TypeError, match="mismatched type annotations"):

        class D(A):
            x: bool
            y: str

    with pytest.raises(TypeError, match="mismatched type annotations"):

        class E(A):
            y: str
            x: bool  # different order

    with pytest.raises(TypeError, match="unsubscripted"):

        class G(eqx.Module):
            x: AbstractClassVar

    with pytest.raises(TypeError, match="mismatched type annotations"):

        class H1(A):
            x: str

    with pytest.raises(TypeError, match="mismatched type annotations"):

        class H2(A):
            x: ClassVar[str]

    with pytest.raises(TypeError, match="mismatched type annotations"):

        class I(A):  # noqa: E742
            x: AbstractVar[bool]

    class J(A):
        x: AbstractClassVar[bool]

    class K(A):
        x: ClassVar[bool]

    with pytest.raises(TypeError, match="cannot have value"):

        class L(eqx.Module):
            x: AbstractClassVar[bool] = True

    class M1(A):
        x = True
        y: bool = False

    class M2(A):
        x = True
