# Module: Layers

`zlynx.module` provides the main reusable model-building blocks.

## Core Layers

```python
from zlynx.module import Attention, MLP, RMSNorm, RotaryEmbedding
```

- `Attention`
- `MLP`
- `RMSNorm`
- `RotaryEmbedding`

## Other Building Blocks

```python
from zlynx.module import AdaLayerNormZero, TimestepEmbedder, PatchEmbed
```

- `AdaLayerNormZero`
- `TimestepEmbedder`
- `PatchEmbed`

## Cache Helpers

From `zlynx.module.cache`:

```python
from zlynx.module.cache import KVCache, KVCacheState, CacheIndex
```

## Notes

- the current module layer APIs are still mixed between `key`-style and `rngs`-style initialization depending on the file
- use the concrete layer source when you need exact constructor signatures
- RoPE helpers such as `compute_rope(...)` and `apply_rope(...)` live in `zlynx.module.rope`
