import os
import random
import typing
import warnings

import beartype
import jax.random as jrandom
import pytest


typing.TESTING = True  # pyright: ignore
warnings.filterwarnings(
    "ignore",
    category=beartype.roar.BeartypeDecorHintPep585DeprecationWarning,  # pyright: ignore
)
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"


@pytest.fixture()
def getkey():
    def _getkey():
        # Not sure what the maximum actually is but this will do
        return jrandom.PRNGKey(random.randint(0, 2**31 - 1))

    return _getkey
