import random

import jax.random as jrandom
import pytest


@pytest.fixture()
def getkey():
    def _getkey():
        # Not sure what the maximum actually is but this will do
        return jrandom.PRNGKey(random.randint(0, 2**31 - 1))

    return _getkey
