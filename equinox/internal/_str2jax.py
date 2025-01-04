import wadler_lindig as wl

from .._module import Module


def str2jax(msg: str):
    """Creates a JAXable object whose `str(...)` is the specified string."""

    class M(Module):
        def __pdoc__(self, **kwargs):
            del kwargs
            return wl.TextDoc(msg)

        def __repr__(self):
            return msg

    M.__name__ = M.__qualname__ = msg
    return M()
