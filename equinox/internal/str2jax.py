from ..module import Module


def str2jax(msg: str):
    class M(Module):
        def __repr__(self):
            return msg

    return M()
