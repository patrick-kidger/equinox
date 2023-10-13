import dataclasses
import random
import typing

import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray


typing.TESTING = True  # pyright: ignore


# This offers reproducability -- the initial seed is printed in the repr so we can see
# it when a test fails.
# Note the `eq=False`, which means that `_GetKey `objects have `__eq__` and `__hash__`
# based on object identity.
@dataclasses.dataclass(eq=False)
class _GetKey:
    seed: int
    call: int
    key: PRNGKeyArray

    def __init__(self, seed: int):
        self.seed = seed
        self.call = 0
        self.key = jr.PRNGKey(seed)

    def __call__(self):
        self.call += 1
        return jr.fold_in(self.key, self.call)


@pytest.fixture
def getkey():
    return _GetKey(random.randint(0, 2**31 - 1))
