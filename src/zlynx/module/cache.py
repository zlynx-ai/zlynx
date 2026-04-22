





import jax
import jax.numpy as jnp
from flax import nnx


class KVCacheState(nnx.Variable):
    pass


class CacheIndex(nnx.Variable):
    pass


class KVCache(nnx.Module):
    """
    A standalone Key-Value Cache component for autoregressive generation.
    Instantiated functionally within an Attention module.
    """
    def __init__(self, kv_head: int, head_dim: int, dtype=jnp.float32):
        self.kv_head = kv_head
        self.head_dim = head_dim
        self.dtype = dtype

    def init_cache_state(self, batch_size: int, max_seq_len: int):
        """Pre-allocates the JAX arrays for KV Cache tracing."""
        self.k_cache = KVCacheState(
            jnp.zeros(
                (batch_size, max_seq_len, self.kv_head, self.head_dim),
                dtype=self.dtype,
            )
        )
        self.v_cache = KVCacheState(
            jnp.zeros(
                (batch_size, max_seq_len, self.kv_head, self.head_dim),
                dtype=self.dtype,
            )
        )
        self.cache_index = CacheIndex(jnp.zeros((), dtype=jnp.int32))

    def update_cache(self, key: jax.Array, value: jax.Array):
        if not hasattr(self, "k_cache") or self.k_cache is None:
            return key, value

        # Modern NNX dictates we read array variables via [...] or .get_value()
        curr_k = self.k_cache[...]
        curr_v = self.v_cache[...]
        curr_index = self.cache_index[...]
        
        key = key.astype(curr_k.dtype)
        value = value.astype(curr_v.dtype)

        S = key.shape[1]

        # Splice in the new tokens into the cache
        new_k = jax.lax.dynamic_update_slice(curr_k, key, (0, curr_index, 0, 0))
        new_v = jax.lax.dynamic_update_slice(curr_v, value, (0, curr_index, 0, 0))
        
        # Write back via modern NNX syntax
        self.k_cache[...] = new_k
        self.v_cache[...] = new_v
        
        # Update index
        # For non-Array variables or simple scalars wrapped in jax arrays, we update
        self.cache_index[...] = curr_index + S

        return self.k_cache[...], self.v_cache[...]
