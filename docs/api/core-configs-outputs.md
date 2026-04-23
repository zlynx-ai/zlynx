# Core: Configs and Outputs

`zlynx.core` provides small dataclass-style structs for configs and model outputs.

## Configs

```python
from zlynx.core import Config, LanguageConfig
```

### `Config`

Base config with two metadata fields:

- `arch`
- `conf`

Use it as the root config type for save/load reconstruction.

### `LanguageConfig`

Extends `Config` with common language-model fields such as:

- `vocab_size`
- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `attention_head`
- `kv_head`
- `head_dim`
- `dtype`
- `param_dtype`
- `use_cache`
- `base`
- `max_position_embedding`
- `rope_scaling`

## Outputs

```python
from zlynx.core import ModelOutput, CausalLMOutput
```

### `ModelOutput`

Generic output container with:

- `loss`
- `hidden_states`
- `auxiliary`

### `CausalLMOutput`

Adds language-model fields:

- `logits`
- `past_key_values`

Use these output structs when you want model returns to stay explicit and stable.
