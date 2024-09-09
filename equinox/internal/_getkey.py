import dataclasses
import random
from typing import Optional

import jax.random as jr
from jaxtyping import PRNGKeyArray

from .._config import EQX_GETKEY_SEED


# This offers reproducibility -- the initial seed is printed in the repr so we can see
# it when a test fails.
# Note the `eq=False`, which means that `GetKey `objects have `__eq__` and `__hash__`
# based on object identity.
@dataclasses.dataclass(eq=False)
class GetKey:
    """Designed for use as a fixture in tests.

    !!! Example

        ```python
        # tests/conftest.py

        @pytest.fixture
        def getkey():
            return eqxi.GetKey()
        ```

    Do not use this in any other context; the random seed generation gives deliberate
    non-determinism.
    """

    seed: int
    call: int
    key: PRNGKeyArray

    def __init__(self, seed: Optional[int] = EQX_GETKEY_SEED):
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        self.seed = seed
        self.call = 0
        self.key = jr.PRNGKey(seed)

    def __call__(self):
        self.call += 1
        return jr.fold_in(self.key, self.call)
