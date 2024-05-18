import abc
from typing import ClassVar, TYPE_CHECKING

import equinox as eqx
import pytest


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar, ClassVar as AbstractVar
else:
    from equinox.internal import AbstractClassVar, AbstractVar


def test_abstract_method():
    class A(eqx.Module):
        @abc.abstractmethod
        def x(self): ...

    class B(A):
        pass

    with pytest.raises(TypeError, match="abstract method"):
        A()

    with pytest.raises(TypeError, match="abstract method"):
        B()


def test_abstract_attribute_stringified():
    with pytest.raises(NotImplementedError):

        class A(eqx.Module):
            x: "AbstractVar[bool]"


def test_abstract_attribute():
    class A(eqx.Module):
        x: AbstractVar[bool]

    assert A.__abstractvars__ == frozenset({"x"})
    assert A.__abstractclassvars__ == frozenset()

    with pytest.raises(TypeError, match="abstract attributes"):
        A()

    class B(A):
        y: int

    assert B.__abstractvars__ == frozenset({"x"})
    assert B.__abstractclassvars__ == frozenset()

    with pytest.raises(TypeError, match="abstract attributes"):
        B(y=2)

    class C(A):
        x: bool
        y: str

    assert C.__abstractvars__ == frozenset()
    assert C.__abstractclassvars__ == frozenset()

    C(x=True, y="hi")

    class D(A):
        y: str
        x: bool  # different order

    assert D.__abstractvars__ == frozenset()
    assert D.__abstractclassvars__ == frozenset()

    d = D("hi", True)
    assert d.x is True
    assert d.y == "hi"

    class E(A):
        y: str

        @property
        def x(self):
            return True

    assert E.__abstractvars__ == frozenset()
    assert E.__abstractclassvars__ == frozenset()

    e = E(y="hi")
    assert e.x is True
    assert e.y == "hi"

    with pytest.raises(TypeError, match="unsubscripted"):

        class F(eqx.Module):
            x: AbstractVar

    class G(A):
        x: AbstractVar[bool]

    assert G.__abstractvars__ == frozenset({"x"})
    assert G.__abstractclassvars__ == frozenset()

    class H(A):
        x: AbstractClassVar[bool]

    assert H.__abstractvars__ == frozenset()
    assert H.__abstractclassvars__ == frozenset({"x"})

    class I(A):  # noqa: E742
        x: bool

    assert I.__abstractvars__ == frozenset()
    assert I.__abstractclassvars__ == frozenset()
    I(True)

    with pytest.raises(TypeError, match="cannot have value"):

        class J(eqx.Module):
            x: AbstractVar[bool] = True

    class K(A):
        x = True
        y: bool = False

    assert K.__abstractvars__ == frozenset()
    assert K.__abstractclassvars__ == frozenset()
    K()

    class L(A):
        x = True

    assert L.__abstractvars__ == frozenset()
    assert L.__abstractclassvars__ == frozenset()
    L()


def test_abstract_class_attribute():
    class A(eqx.Module):
        x: AbstractClassVar[bool]

    assert A.__abstractvars__ == frozenset()
    assert A.__abstractclassvars__ == frozenset({"x"})

    with pytest.raises(TypeError, match="abstract class attributes"):
        A()

    class B(A):
        y: int

    assert B.__abstractvars__ == frozenset()
    assert B.__abstractclassvars__ == frozenset({"x"})

    with pytest.raises(TypeError, match="abstract class attributes"):
        B(y=2)

    with pytest.raises(TypeError, match="unsubscripted"):

        class C(eqx.Module):
            x: AbstractClassVar

    class D(A):
        x: AbstractClassVar[bool]

    assert D.__abstractvars__ == frozenset()
    assert D.__abstractclassvars__ == frozenset({"x"})

    class E(A):
        x: ClassVar[bool]

    assert E.__abstractvars__ == frozenset()
    assert E.__abstractclassvars__ == frozenset()

    with pytest.raises(TypeError, match="cannot have value"):

        class F(eqx.Module):
            x: AbstractClassVar[bool] = True

    class G(A):
        x = True
        y: bool = False

    assert G.__abstractvars__ == frozenset()
    assert G.__abstractclassvars__ == frozenset()

    class H(A):
        x = True

    assert H.__abstractvars__ == frozenset()
    assert H.__abstractclassvars__ == frozenset()

    class I(A):  # noqa: E742
        x: AbstractVar[bool]

    assert I.__abstractvars__ == frozenset()
    assert I.__abstractclassvars__ == frozenset({"x"})


def test_abstract_multiple_inheritance():
    class A(eqx.Module):
        x: AbstractVar[int]

    class B(eqx.Module):
        x: int

    class C(B, A):
        pass

    C(1)
