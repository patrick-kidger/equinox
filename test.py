import equinox as eqx
import jax
from equinox.nn._attention import MultiheadAttention
from equinox.nn._kv_cache import StandardKVCache


query_size = 6
num_heads = 1
state_length = 8
seq_len = 3

standard_kv_cache = StandardKVCache(
    key_shape=(state_length, num_heads, query_size),
    value_shape=(state_length, num_heads, query_size),
)

key = jax.random.PRNGKey(0)
mha, state = eqx.nn.make_with_state(MultiheadAttention)(
    query_size=query_size,
    num_heads=num_heads,
    state_length=state_length,
    kv_cache=standard_kv_cache,
    key=key,
)

for states in range(state_length - 3):
    x = jax.numpy.ones(shape=(seq_len, query_size)) * (states + 1)
    y, state = mha(x, x, x, mask="causal", state=state)
