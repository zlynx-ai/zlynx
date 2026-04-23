# Core: Z

`Z` is the base class for checkpointing and model sharing.

```python
from flax import nnx
from zlynx.core import Z


class MyModel(Z):
    def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(8, 8, rngs=rngs)
```

## Main Methods

- `save(path, fmt="orbax")`
- `load(path, fmt="orbax", **kwargs)`
- `load_config(path, asdict=False, config_map=None)`
- `push_hf(repo_id, fmt="safetensors", **kwargs)`
- `load_hf(repo_id, fmt="safetensors", **kwargs)`
- `push_kaggle(repo_id, fmt="safetensors", **kwargs)`
- `load_kaggle(repo_id, fmt="safetensors", **kwargs)`

## Format Defaults

- local save/load: `orbax`
- remote sharing: `safetensors`

## Notes

- `Z.load(...)` can reconstruct a model when the saved config contains `arch` and `conf`.
- `load_config(...)` supports `config_map` when config field names changed.
- weight loading supports `module_map` for compatible path renames.
- constructor requirements still belong to the model author. If your model needs `rngs`, `key`, or other init arguments, provide them explicitly or give them defaults in your model class.

For end-to-end examples, see [Save, Load & Share](../getting-started/ckpt.md).
