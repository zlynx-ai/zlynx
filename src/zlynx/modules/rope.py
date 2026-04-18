




import jax
import jax.numpy as jnp
import math
from flax import nnx
from dataclasses import dataclass


def compute_rope(positions: jax.Array, head_dim: int, base_frequency: float) -> tuple[jax.Array, jax.Array]:
    """Compute RoPE cos/sin from absolute positions.

    Args:
        positions: (B, S) integer position indices
        head_dim: dimension of each attention head
        base_frequency: RoPE base frequency (e.g. 10_000 for local, 1_000_000 for global)

    Returns:
        cos, sin: each of shape (B, S, head_dim)
    """
    inv_freq = 1.0 / (base_frequency ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = positions[..., None].astype(jnp.float32) * inv_freq[None, None, :]  # (B, S, head_dim//2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # (B, S, head_dim)
    return jnp.cos(emb), jnp.sin(emb)


def apply_partial_rope(
    query: jax.Array,
    key: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    proportion: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Apply RoPE to only the first `proportion` of head dimensions.

    Args:
        query, key: (..., head_dim)
        cos, sin: (..., head_dim) — full head_dim freqs
        proportion: fraction of head_dim to rotate (e.g. 0.25 for global attn)

    Returns:
        rotated query, key
    """
    if proportion >= 1.0:
        return apply_rope(query, key, cos, sin)

    head_dim = query.shape[-1]
    rope_dim = int(head_dim * proportion)

    q_rope, q_pass = query[..., :rope_dim], query[..., rope_dim:]
    k_rope, k_pass = key[..., :rope_dim], key[..., rope_dim:]

    cos_r, sin_r = cos[..., :rope_dim], sin[..., :rope_dim]
    q_rope, k_rope = apply_rope(q_rope, k_rope, cos_r, sin_r)

    return jnp.concatenate([q_rope, q_pass], axis=-1), jnp.concatenate([k_rope, k_pass], axis=-1)


def rotate_half(x: jax.Array):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concat([-x2, x1], axis=-1)


def apply_rope(query: jax.Array, key: jax.Array, cos: jax.Array, sin: jax.Array):
    # [B, S, E]
    cos = jnp.expand_dims(cos, axis=-2).astype(query.dtype)
    sin = jnp.expand_dims(sin, axis=-2).astype(query.dtype)

    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    return query, key


def init_rope_llama3(base, head_dim, rope_scaling: dict):
    partial_rotary_factor = rope_scaling.get("partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / base ** (jnp.arange(0, dim, 2) / dim)

    factor = rope_scaling["factor"]  # `8` in the original implementation
    low_freq_factor = rope_scaling["low_freq_factor"]  # `1` in the original implementation
    high_freq_factor = rope_scaling["high_freq_factor"]  # `4` in the original implementation
    old_context_len = rope_scaling["original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor

def init_rope_linear(): ...

def init_rope_dynamic(): ...

def init_rope_yarn(): ...

def init_rope_longrope(): ...

def init_rope_hirope(): ...

ROPE_TYPE_FN = {
    "linear": ...,
    "dynamic": ...,
    "yarn": ...,
    "longrope": ...,
    "llama3": init_rope_llama3,
    "hirope": ...
}

@dataclass(frozen=True)
class RoPEConfig:
    base: float = 10000.0
    dim: int = 256
    head_dim: int = 64
    max_position_embeddings: int = 8192
    original_max_position_embeddings: int = 8192
    # Hierachical RoPE params
    K: int = 3              # number of hierarchy levels (gears)
    B: int = 32             # base for position decomposition (gear capacity)
    rope_type: str = "standard"  # "standard" or "hirope"


class RotaryEmbedding(nnx.Module):
    def __init__(
        self, base: float, 
        head_dim: int, 
        max_position_embedding: int,
        rope_scaling: dict | None = None
    ):
        super().__init__()
        
        self.original_max_position_embedding = rope_scaling.get("original_max_position_embedding", max_position_embedding) if rope_scaling else max_position_embedding
        self.max_position_embedding = max_position_embedding

        init_rope_fn = self.rope_default_fn
        if rope_scaling is not None:
            rope_type = rope_scaling.get("rope_type")
            init_rope_fn = ROPE_TYPE_FN[rope_type]
        
        inv_freq, attention_factor = init_rope_fn(base, head_dim, rope_scaling)
        self.inv_freq = nnx.Cache(inv_freq)
        self.attention_factor = nnx.Cache(attention_factor)

    def rope_default_fn(self, base, head_dim, rope_scaling):
        attention_factor = 1.0
        inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        return inv_freq, attention_factor

    def __call__(self, hidden_states: jax.Array, position_ids: jax.Array | None = None):
        if position_ids is None:
            B, S, _ = hidden_states.shape
            position_ids = jnp.expand_dims(jnp.arange(S), axis=(0, 1)).repeat(B, axis=0).astype(float)
        else:
            B = hidden_states.shape[0]
            position_ids = jnp.expand_dims(position_ids, axis=1).astype(float) # from (B, S) -> (B, 1, S)
            
        inv_freq = jnp.expand_dims(self.inv_freq, axis=(0, -1)).repeat(B, axis=0).astype(float)

        scale = 1.0
        if self.max_position_embedding > self.original_max_position_embedding:
            scale = self.max_position_embedding / self.original_max_position_embedding

        position_ids = position_ids / scale

        freq = (inv_freq @ position_ids).transpose(0, 2, 1)
        embed = jnp.concat([freq, freq], axis=-1)
        cos = jax.lax.cos(embed)
        sin = jax.lax.sin(embed)
        return cos, sin