internal_caches = []
internal_lru_caches = []
internal_rope_embedding_cache = {}
internal_sinusoidal_positional_encoding_cache = {}


def clear_caches():
    """Clears internal Equinox caches.

    Best used before calling `jax.clear_caches()` or `jax.clear_backends()`.

    **Arguments:**

    None.

    **Returns:**

    None.
    """
    for cache in internal_caches:
        cache.clear()
    for cache in internal_lru_caches:
        cache.cache_clear()
    internal_rope_embedding_cache.clear()
    internal_sinusoidal_positional_encoding_cache.clear()
