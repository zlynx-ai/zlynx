# Models

The `zlynx.models` package provides model architectures, a base class for checkpointing, output dataclasses, configuration structs, and inference utilities.

```python
from zlynx.models import Z, LanguageModel, ModelOutput, CausalLMOutput
```

---

## `Z`

Base class for all Zlynx models. Inherit from `Z` to get save/load/push functionality for free. Supports Orbax and SafeTensors formats, and integrates with HuggingFace Hub and Kaggle.

```python
class Z(nnx.Module)
```

### Quick Start

```python
from zlynx import Z

class MyModel(Z):
    def __init__(self, config):
        self.config = config
        self.embed = nnx.Embed(num_embeddings=1000, features=256)
        self.linear = nnx.Linear(256, 1000)

# Save
model = MyModel(config)
model.save("./my_model")

# Load
model = MyModel.load("./my_model")
```

### `Z.save`

```python
def save(
    self, path: str | Path, *,
    format: Literal["orbax", "safetensors"] = "orbax",
    max_shard_size_gb: float = 3.0,
) -> None
```

Saves model weights and `config.json` to disk. Uses atomic writes via a temporary directory.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | — | Output directory path. |
| `format` | `"orbax" \| "safetensors"` | `"orbax"` | Checkpoint format. |
| `max_shard_size_gb` | `float` | `3.0` | Maximum shard size for SafeTensors (GiB). |

### `Z.load`

```python
@classmethod
def load(
    cls, path: str | Path, *,
    dtype: str | None = None,
    config=None,
    config_map: dict | None = None,
    module_map: dict | None = None,
    sharding: int | str | None = None,
    format: Literal["orbax", "safetensors"] = "orbax",
    **kwargs,
) -> Z
```

Loads model weights from disk. Uses `nnx.eval_shape` for memory-efficient initialization.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | — | Checkpoint directory. |
| `dtype` | `str \| None` | `None` | Cast all floating-point params to this dtype. |
| `config` | `Any \| None` | `None` | Override config. `None` = load from `config.json`. |
| `config_map` | `dict \| None` | `None` | Remap config key names. |
| `module_map` | `dict \| None` | `None` | Remap module/weight names (SafeTensors). |
| `sharding` | `int \| str \| None` | `None` | `"ddp"`, `"fsdp"`, or `None`. |
| `format` | `"orbax" \| "safetensors"` | `"orbax"` | Checkpoint format. |

When called on `Z` directly (not a subclass), auto-detects the architecture from `config.arch`.

### `Z.load_config`

```python
@classmethod
def load_config(cls, path, asdict=False, config_map=None) -> Any
```

Loads and returns the config from `config.json`. If `asdict=True`, returns a plain dict.

### `Z.load_hf`

```python
@classmethod
def load_hf(
    cls, repo_id: str, *,
    dtype=None, config=None, config_map=None,
    module_map=None, sharding=None,
    format="safetensors", hf_kwargs=None,
    **model_kwargs,
) -> Z
```

Downloads a model from HuggingFace Hub via `snapshot_download`, then loads it.

### `Z.push_hf`

```python
def push_hf(
    self, repo_id: str,
    private: bool = False, *,
    format="safetensors",
    max_shard_size_gb=3.0,
    **kwargs,
)
```

Pushes model weights and config to HuggingFace Hub.

### `Z.load_kaggle`

```python
@classmethod
def load_kaggle(
    cls, repo_id: str,
    variation="default", *,
    dtype=None, config=None, config_map=None,
    module_map=None, sharding=None,
    format="safetensors", kaggle_kwargs=None,
    **model_kwargs,
) -> Z
```

Downloads a model from Kaggle via `kagglehub.model_download`, then loads it.

### `Z.push_kaggle`

```python
def push_kaggle(
    self, repo_id: str,
    variation="default", *,
    format="safetensors",
    max_shard_size_gb=3.0,
    **kwargs,
)
```

Pushes model weights and config to Kaggle Models.

---

## `LanguageModel`

Mixin class that adds autoregressive `.generate()` to any model.

```python
class LanguageModel:
    def __init__(self, **kwargs)
```

Models that inherit both `LanguageModel` and `Z` (e.g., `LlamaLanguageModel`) gain both generation and checkpoint capabilities.

### `LanguageModel.generate`

```python
def generate(
    self,
    input_ids: jax.Array,
    attention_mask: jax.Array = None,
    key: jax.Array | None = None,
    max_new_tokens: int = 64,
    ctxlen: int = 2048,
    batch: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    eos_token_id: int | None = None,
    suppress_tokens: list[int] | None = None,
) -> jax.Array
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_ids` | `jax.Array` | — | Input token IDs `(B, S)`. |
| `attention_mask` | `jax.Array \| None` | `None` | Attention mask. `None` = all ones. |
| `key` | `jax.Array \| None` | `None` | PRNG key for sampling. `None` = greedy decoding. |
| `max_new_tokens` | `int` | `64` | Maximum tokens to generate. |
| `ctxlen` | `int` | `2048` | KV cache context length. |
| `batch` | `int \| None` | `None` | Batch size override. `None` = inferred from `input_ids`. |
| `temperature` | `float` | `1.0` | Sampling temperature. `0.0` = greedy. |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold. |
| `top_k` | `int` | `50` | Top-K sampling. |
| `repetition_penalty` | `float` | `1.0` | Repetition penalty factor. |
| `eos_token_id` | `int \| None` | `None` | Stop token ID. |
| `suppress_tokens` | `list[int] \| None` | `None` | Token IDs to suppress during generation. |

**Returns:** `jax.Array` of shape `(B, ctxlen)` containing prompt + generated tokens.

**Behavior:**
1. **Prefill** — runs a full forward pass on the prompt to populate KV caches.
2. **Decode** — JIT-compiled step-by-step generation with KV cache reuse.
3. **Sampling** — supports greedy, top-k, nucleus (top-p), temperature, and repetition penalty via `sample_token`.

---

## Configuration

### `Config`

Base configuration struct using `flax.struct.dataclass`.

```python
@struct.dataclass
class Config:
    arch: str | None = None    # Architecture class name (e.g. "LlamaLanguageModel")
    conf: str | None = None    # Config class name (e.g. "LlamaConfig")
```

### `LanguageConfig`

Extends `Config` with common language model fields.

```python
@struct.dataclass
class LanguageConfig(Config):
    vocab_size: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    act_fn: str | None = None
    num_hidden_layers: int | None = None
    norm_eps: float | None = None
    bias: bool | None = None
    dtype: str | None = None
    param_dtype: str | None = None
    use_cache: bool | None = None

    # Attention
    attention_head: int | None = None
    kv_head: int | None = None
    head_dim: int | None = None
    attention_bias: bool | None = None

    # RoPE
    base: float | None = None
    original_max_position_embedding: int | None = None
    max_position_embedding: int | None = None
    rope_scaling: dict | None = None
```

---

## Output Dataclasses

### `ModelOutput`

```python
@struct.dataclass
class ModelOutput:
    loss: jax.Array | None = None
    hidden_states: tuple[jax.Array, ...] | None = None
    auxiliary: dict | None = None
```

### `CausalLMOutput`

Extends `ModelOutput` for autoregressive language models.

```python
@struct.dataclass
class CausalLMOutput(ModelOutput):
    logits: jax.Array | None = None           # (B, S, V)
    past_key_values: list | None = None       # KV cache
```

---

## Inference Utilities

### `sample_token`

```python
@jax.jit
def sample_token(
    logits, input_ids, input_mask, key,
    temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0,
) -> tuple[jax.Array, jax.Array]
```

JIT-compiled token sampling with temperature, top-k, nucleus (top-p), and repetition penalty.

Returns `(sampled_token_ids, new_key)`.

---

## Model Architectures

### Llama

A Llama-style decoder-only transformer implementation.

#### `LlamaConfig`

```python
@struct.dataclass
class LlamaConfig(Config):
    arch: str = "LlamaLanguageModel"
    conf: str = "LlamaConfig"
    vocab_size: int = 80000
    hidden_size: int = 1024
    intermediate_size: int = 2048
    act_fn: str = "silu"
    num_hidden_layers: int = 4
    norm_eps: float = 1e-6
    bias: bool = False
    dtype: str = "bfloat16"
    param_dtype: str = "float32"
    use_cache: bool = True
    attention_head: int = 16
    kv_head: int = 8
    head_dim: int = 32
    attention_bias: bool = False
    base: float = 10_000
    original_max_position_embedding: int = 2048
    max_position_embedding: int = 2048
    rope_scaling: dict | None = None
```

#### `LlamaLanguageModel`

```python
class LlamaLanguageModel(LanguageModel, Z):
    def __init__(self, config: LlamaConfig, key=jax.random.key(42))
```

Inherits both `LanguageModel` (for `.generate()`) and `Z` (for `.save()` / `.load()`).

**Architecture:** `Embed` → `N × LlamaTransformer` blocks → `RMSNorm` → `Linear` (LM head)

Each `LlamaTransformer` block: `RMSNorm` → `Attention` (GQA + RoPE) → residual → `RMSNorm` → `MLP` (SwiGLU) → residual

**`__call__` signature:**

```python
def __call__(
    self, input_ids, attention_mask=None,
    position_ids=None, labels=None,
    past_key_values=None,
) -> CausalLMOutput
```

When `labels` is provided, computes shifted cross-entropy loss internally.

---

### DiT (Diffusion Transformer)

> [!WARNING]
> Experimental. Subject to change.

A Diffusion Transformer for image generation, following the DiT paper architecture.

#### `DiTConfig`

```python
@struct.dataclass
class DiTConfig:
    img_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    frequency_embedding_size: int = 256
    use_bias: bool = True
```

#### `DiT`

```python
class DiT(nnx.Module):
    def __init__(self, config: DiTConfig, dtype=jnp.float32, param_dtype=jnp.float32, rngs=None)
```

**Architecture:** `PatchEmbed` + positional embedding → `N × DiTBlock` (AdaLN-Zero conditioned) → `FinalLayer`

**`__call__` signature:**

```python
def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array
    # x: (B, H, W, C) noisy images
    # t: (B,) timesteps
    # Returns: (B, N, patch_size² × C) predicted noise
```
