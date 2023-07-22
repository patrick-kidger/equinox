from .._module import Module
from .._pretty_print import text as pp_text


def str2jax(msg: str):
    """Creates a JAXable object whose `str(...)` is the specified string."""

    class M(Module):
        def __tree_pp__(self, **kwargs):
            return pp_text(msg)

        def __repr__(self):
            return msg

    M.__name__ = M.__qualname__ = msg
    return M()
