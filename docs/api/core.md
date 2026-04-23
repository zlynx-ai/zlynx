# Core

`zlynx.core` contains the base APIs that the rest of the library builds on:

- `Z` for save/load and hub helpers
- config dataclasses such as `Config` and `LanguageConfig`
- output dataclasses such as `ModelOutput` and `CausalLMOutput`
- inference helpers such as `LanguageModel.generate(...)`

See also:

- [Z](./core-z.md)
- [Configs and Outputs](./core-configs-outputs.md)
- [Inference](./core-inference.md)

## Common Imports

```python
from zlynx.core import (
    Z,
    Config,
    LanguageConfig,
    ModelOutput,
    CausalLMOutput,
    LanguageModel,
)
```

## `Z`

`Z` is the checkpoint and model-sharing base class.

```python
from flax import nnx
from zlynx.core import Z


class MyModel(Z):
    def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(8, 8, rngs=rngs)
```

Key methods:

- `save(path, fmt="orbax")`
- `load(path, fmt="orbax", **kwargs)`
- `load_config(path, asdict=False, config_map=None)`
- `push_hf(repo_id, fmt="safetensors", **kwargs)`
- `load_hf(repo_id, fmt="safetensors", **kwargs)`
- `push_kaggle(repo_id, fmt="safetensors", **kwargs)`
- `load_kaggle(repo_id, fmt="safetensors", **kwargs)`

The local default is `orbax`. Remote-sharing defaults use `safetensors`.

## Config Structs

`Config` is the minimal base config:

```python
from flax import struct
from zlynx.core import Config


@struct.dataclass
class MyConfig(Config):
    hidden_size: int = 256
```

`LanguageConfig` extends that pattern with common language-model fields such as:

- `vocab_size`
- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `attention_head`
- `kv_head`
- `head_dim`
- `dtype`
- `param_dtype`

## Output Structs

```python
from zlynx.core import ModelOutput, CausalLMOutput
```

- `ModelOutput` is the generic base output.
- `CausalLMOutput` adds `logits` and `past_key_values` for autoregressive models.

## Generation Helper

`LanguageModel` adds `.generate(...)` for language-model style decoding.

```python
from zlynx.core import LanguageModel
```

Use it when your model should support token generation on top of checkpoint helpers.
