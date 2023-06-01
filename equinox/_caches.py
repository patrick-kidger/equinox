internal_caches = []
internal_lru_caches = []


def clear_caches():
    """Clears internal Equinox caches.

    Best used before calling `jax.clear_caches()` and `jax.clear_backends()`.

    **Arguments:**

    None.

    **Returns:**

    None.
    """
    for cache in internal_caches:
        cache.clear()
    for cache in internal_lru_caches:
        cache.cache_clear()
