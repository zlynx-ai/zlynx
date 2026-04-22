# Modules

The `zlynx.modules` package provides reusable building-block layers for constructing models in Flax NNX. These modules are architecture-agnostic and compose into higher-level models.

```python
from zlynx.modules import Attention, MLP, RMSNorm, RotaryEmbedding
```

---

## `Attention`

Multi-head attention with support for Grouped-Query Attention (GQA), Multi-Query Attention (MQA), rotary position embeddings, and KV caching.

```python
class Attention(nnx.Module):
    def __init__(
        self, key,
        hidden_size: int,
        attention_head: int,
        head_dim: int,
        kv_head: int | None = None,
        bias: bool = False,
        layer_idx: int | None = None,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    )
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `key` | `jax.Array` | — | PRNG key for parameter initialization. |
| `hidden_size` | `int` | — | Input/output dimension. |
| `attention_head` | `int` | — | Number of query attention heads. |
| `head_dim` | `int` | — | Dimension per attention head. |
| `kv_head` | `int \| None` | `None` | Number of key/value heads. `None` = same as `attention_head`. Set lower for GQA/MQA. |
| `bias` | `bool` | `False` | Use bias in Q/K/V/O projections. |
| `layer_idx` | `int \| None` | `None` | Layer index (for debugging/cache). |
| `dtype` | `jnp.dtype` | `bfloat16` | Compute dtype. |
| `param_dtype` | `jnp.dtype` | `float32` | Parameter storage dtype. |

**Projections:** `q_proj`, `k_proj`, `v_proj`, `o_proj` — all `nnx.Linear`.

### `__call__`

```python
def __call__(
    self, hidden_states,
    attention_mask=None,
    position_embedding=None,    # (cos, sin) from RotaryEmbedding
    past_key_value=None,        # (k_cache, v_cache, cache_index)
) -> tuple[jax.Array, tuple | None]
```

Returns `(output, present_key_value)`.

- When `position_embedding` is provided, RoPE is applied to Q and K.
- When `past_key_value` is provided, KV caching is used for autoregressive decoding.
- GQA: K/V heads are repeated to match Q heads automatically.

---

## `KVCache`

Standalone Key-Value cache for autoregressive generation.

```python
class KVCache(nnx.Module):
    def __init__(self, kv_head: int, head_dim: int, dtype=jnp.float32)
```

| Parameter | Type | Description |
|---|---|---|
| `kv_head` | `int` | Number of KV heads. |
| `head_dim` | `int` | Dimension per head. |
| `dtype` | `jnp.dtype` | Cache array dtype. |

### Methods

| Method | Description |
|---|---|
| `init_cache_state(batch_size, max_seq_len)` | Pre-allocates K/V cache arrays of shape `(B, S, kv_head, head_dim)`. |
| `update_cache(key, value)` | Splices new key/value tensors into the cache at the current index. Returns the updated full K/V arrays. |

**State variables:** `k_cache` (`KVCacheState`), `v_cache` (`KVCacheState`), `cache_index` (`CacheIndex`).

---

## `MLP`

Gated MLP (SwiGLU-style) with gate, up, and down projections.

```python
class MLP(nnx.Module):
    def __init__(
        self, key,
        hidden_size: int,
        intermediate_size: int,
        act_fn=jax.nn.silu,
        bias: bool = False,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    )
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `key` | `jax.Array` | — | PRNG key. |
| `hidden_size` | `int` | — | Input/output dimension. |
| `intermediate_size` | `int` | — | Hidden dimension of the gate/up projections. |
| `act_fn` | `Callable` | `silu` | Activation function. |
| `bias` | `bool` | `False` | Use bias in projections. |
| `dtype` | `jnp.dtype` | `bfloat16` | Compute dtype. |
| `param_dtype` | `jnp.dtype` | `float32` | Parameter storage dtype. |

**Forward:** `act_fn(gate_proj(x)) * up_proj(x)` → `down_proj(...)`.

---

## `RMSNorm`

Root Mean Square Layer Normalization.

```python
class RMSNorm(nnx.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-9)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hidden_size` | `int` | — | Feature dimension. |
| `eps` | `float` | `1e-9` | Epsilon for numerical stability. |

Computes in `float32` internally, casts output back to input dtype.

---

## `AdaLayerNormZero`

Adaptive Layer Normalization with zero initialization, used in Diffusion Transformers (DiT). Projects a conditioning vector into shift, scale, and gate parameters for both attention and MLP branches.

```python
class AdaLayerNormZero(nnx.Module):
    def __init__(
        self, hidden_size: int, eps: float = 1e-6,
        dtype=jnp.float32, param_dtype=jnp.float32,
        rngs: nnx.Rngs | None = None
    )
```

### `__call__`

```python
def __call__(self, x: jax.Array, c: jax.Array) -> tuple
    # x: (B, S, D) input sequence
    # c: (B, D) conditioning vector (e.g. timestep embedding)
    # Returns: (x_norm, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
```

All shift/scale/gate parameters are initialized to zero, so the block starts as an identity function.

---

## `RotaryEmbedding`

Rotary Position Embedding (RoPE) module with support for multiple scaling strategies.

```python
class RotaryEmbedding(nnx.Module):
    def __init__(
        self, base: float,
        head_dim: int,
        max_position_embedding: int,
        rope_scaling: dict | None = None,
    )
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base` | `float` | — | Base frequency for inverse frequency computation. |
| `head_dim` | `int` | — | Dimension per attention head. |
| `max_position_embedding` | `int` | — | Maximum sequence length. |
| `rope_scaling` | `dict \| None` | `None` | Scaling configuration. Must contain `"rope_type"` key. |

### `__call__`

```python
def __call__(self, hidden_states, position_ids=None) -> tuple[cos, sin]
```

Returns `(cos, sin)` tensors to be passed to `Attention` via `position_embedding`.

### Supported RoPE Types

| Type | Status |
|---|---|
| `"llama3"` | ✅ Implemented |
| `"linear"` | 🚧 Placeholder |
| `"dynamic"` | 🚧 Placeholder |
| `"yarn"` | 🚧 Placeholder |
| `"longrope"` | 🚧 Placeholder |
| `"hirope"` | 🚧 Placeholder |

### `RoPEConfig`

```python
@dataclass(frozen=True)
class RoPEConfig:
    base: float = 10000.0
    dim: int = 256
    head_dim: int = 64
    max_position_embeddings: int = 8192
    original_max_position_embeddings: int = 8192
    K: int = 3            # hierarchy levels (HiRoPE)
    B: int = 32           # base for position decomposition (HiRoPE)
    rope_type: str = "standard"
```

### Utility Functions

| Function | Description |
|---|---|
| `apply_rope(query, key, cos, sin)` | Applies RoPE to query and key tensors using the `rotate_half` method. |
| `rotate_half(x)` | Splits the last dimension in half and rotates (used by `apply_rope`). |

---

## `TimestepEmbedder`

Embeds scalar timesteps into dense vectors using sinusoidal encodings followed by a 2-layer MLP. Standard in diffusion models (DiT).

```python
class TimestepEmbedder(nnx.Module):
    def __init__(
        self, hidden_size: int,
        frequency_embedding_size: int = 256,
        dtype=jnp.float32, param_dtype=jnp.float32,
        rngs: nnx.Rngs | None = None,
    )
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hidden_size` | `int` | — | Output embedding dimension. |
| `frequency_embedding_size` | `int` | `256` | Sinusoidal frequency dimension. |

### `__call__`

```python
def __call__(self, t: jax.Array) -> jax.Array
    # t: (B,) timestep values
    # Returns: (B, hidden_size)
```

---

## `PatchEmbed`

2D image to patch embedding (ViT-style stem). Slices an image into non-overlapping patches and projects each to the hidden size.

```python
class PatchEmbed(nnx.Module):
    def __init__(
        self, img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        dtype=jnp.float32, param_dtype=jnp.float32,
        rngs: nnx.Rngs | None = None,
    )
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `img_size` | `int` | `256` | Input image size (assumed square). |
| `patch_size` | `int` | `16` | Patch size. |
| `in_channels` | `int` | `3` | Number of input channels. |
| `embed_dim` | `int` | `768` | Output embedding dimension. |

### `__call__`

```python
def __call__(self, x: jax.Array) -> jax.Array
    # x: (B, H, W, C)
    # Returns: (B, N, embed_dim) where N = (H/P) * (W/P)
```

**Attributes:** `num_patches` — total number of patches.

---

## PEFT (Parameter-Efficient Fine-Tuning)

The `peft` module provides drop-in adapter wrappers for `nnx.Linear` layers and a utility function to apply them across a model.

### `apply_peft`

```python
def apply_peft(
    model: nnx.Module,
    method: str = "lora",
    r: int = 8,
    alpha: int = 16,
    target_modules: Sequence[str] = ("q_proj", "v_proj"),
    rngs: nnx.Rngs | None = None,
) -> nnx.Module
```

Traverses the model and replaces target `nnx.Linear` layers with PEFT adapters **in-place**.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nnx.Module` | — | Model to modify. |
| `method` | `str` | `"lora"` | Adapter method (see table below). |
| `r` | `int` | `8` | Rank. |
| `alpha` | `int` | `16` | Scaling factor (`scaling = alpha / r`). |
| `target_modules` | `Sequence[str]` | `("q_proj", "v_proj")` | Layer name substrings to match. |
| `rngs` | `nnx.Rngs \| None` | `None` | Random number generators. |

### Supported Adapters

| Method | Class | Description |
|---|---|---|
| `"lora"` | `LoraLinear` | Low-Rank Adaptation. Trains A and B matrices. |
| `"dora"` | `DoraLinear` | Weight-Decomposed LoRA. Adds a learnable magnitude vector `m` for directional updates. |
| `"vera"` | `VeraLinear` | Vector-based Random Adaptation. Freezes random A/B matrices, trains only scaling vectors `d` and `b`. |
| `"loha"` | `LohaLinear` | Hadamard Product Adaptation. Update = `(A1 @ B1) ⊙ (A2 @ B2)`. |
| `"lokr"` | `LokrLinear` | Kronecker Product Adaptation. Update = `kron(A @ B, O)`. |
| `"adalora"` | `AdaloraLinear` | Adaptive LoRA. Parameterizes as `P @ diag(E) @ Q` with prunable singular values. |

### Adapter Constructors

All adapters share the same constructor signature:

```python
class <Adapter>Linear(nnx.Module):
    def __init__(self, base_layer: nnx.Linear, r: int, alpha: int, rngs: nnx.Rngs)
```

| Parameter | Type | Description |
|---|---|---|
| `base_layer` | `nnx.Linear` | The original linear layer to wrap. Its parameters are frozen (`nnx.Variable`). |
| `r` | `int` | Rank of the low-rank decomposition. |
| `alpha` | `int` | Scaling factor. `scaling = alpha / r`. |
| `rngs` | `nnx.Rngs` | Random number generators for adapter parameter initialization. |

**Note:** Base layer weights are stored as `nnx.Variable` (frozen), while adapter weights are `nnx.Param` (trainable). This means only adapter parameters appear in the optimizer state.
