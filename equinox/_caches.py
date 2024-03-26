cache_clears = []


def clear_caches():
    """Clears internal Equinox caches.

    Best used before calling `jax.clear_caches()` or `jax.clear_backends()`.

    **Arguments:**

    None.

    **Returns:**

    None.
    """
    for cache_clear in cache_clears:
        cache_clear()
