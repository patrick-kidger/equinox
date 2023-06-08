import random
import typing

import jax.random as jr
import pytest


typing.TESTING = True  # pyright: ignore


@pytest.fixture()
def getkey():
    def _getkey():
        # Not sure what the maximum actually is but this will do
        return jr.PRNGKey(random.randint(0, 2**31 - 1))

    return _getkey
