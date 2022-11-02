from ..module import Module


def str2jax(msg: str):
    """Creates a JAXable object whose `str(...)` is the specified string."""

    class M(Module):
        def __repr__(self):
            return msg

    return M()
