# Module

`zlynx.module` contains reusable Flax NNX building blocks.

See also:

- [Layers](./module-layers.md)
- [PEFT](./module-peft.md)

## Common Imports

```python
from zlynx.module import (
    Attention,
    MLP,
    RMSNorm,
    AdaLayerNormZero,
    TimestepEmbedder,
    PatchEmbed,
    RotaryEmbedding,
)
```

## Main Building Blocks

- `Attention`
- `MLP`
- `RMSNorm`
- `AdaLayerNormZero`
- `TimestepEmbedder`
- `PatchEmbed`
- `RotaryEmbedding`

`zlynx.module.rope` also provides lower-level RoPE helpers such as:

- `compute_rope`
- `apply_rope`
- `apply_partial_rope`

## PEFT Adapters

PEFT adapters live under `zlynx.module.peft`:

```python
from zlynx.module.peft import apply_peft, LoraLinear, DoraLinear
```

Available adapter wrappers include:

- `LoraLinear`
- `DoraLinear`
- `VeraLinear`
- `LohaLinear`
- `LokrLinear`
- `AdaloraLinear`
- `apply_peft(...)`

## Cache Helpers

Key/value cache helpers live under `zlynx.module.cache`:

```python
from zlynx.module.cache import KVCache, KVCacheState, CacheIndex
```

Use the root `zlynx.module` package for common layers, and the submodules when you need PEFT or cache-specific helpers.
