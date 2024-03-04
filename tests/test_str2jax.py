import equinox.internal as eqxi
import jax


def test_basic():
    @jax.jit
    def concat(a, b):
        return eqxi.str2jax(str(a) + str(b))

    assert str(concat(eqxi.str2jax("hello"), eqxi.str2jax("world"))) == "helloworld"
