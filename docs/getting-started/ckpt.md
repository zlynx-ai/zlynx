# Save, Load & Share

How to persist your trained models and restore them later.

## Saving a Model

Every `Z` model has a built-in `save()` method:

```python
model.save("./my_model")
```

This creates a directory containing:

- **Orbax checkpoint files** — the serialized model weights
- **`config.json`** (if the model has a `config` attribute) — architecture metadata

```
my_model/
├── config.json          ← only if model has a config
├── _METADATA
└── ...                  ← Orbax weight files
```

## Saving in Different Formats

### Safetensors Format

Save in HuggingFace-compatible safetensors format:

```python
model.save("./my_model", fmt="safetensors")
```

This creates:

```
my_model/
├── config.json
├── model.safetensors.index.json
├── model-00001-of-xxxxx.safetensors
└── ...
```

### Max Shard Size

Control the max size per safetensors file:

```python
model.save("./my_model", fmt="safetensors", max_shard_size_gb=1)
```

## Loading a Model

Use the **class method** `load()` on your model class:

```python
model = CNN.load(
    "./my_model",
    key=jax.random.key(0),
    num_classes=10,
)
```

### Why pass `key` and `num_classes` again?

Flax NNX needs to reconstruct the **model structure** (shapes, layers, etc.) before it can load the saved weights into it. The `load()` method:

1. Calls your `__init__` with the provided args to rebuild the model structure
2. Loads the checkpoint into that instance

The `key` can be **any key** — it's only used for structural initialization. The saved weights overwrite all parameters.

## Loading Trainer Checkpoints

When you use the `Trainer`, it saves checkpoints automatically using Orbax's `CheckpointManager`:

```
output/
├── 500/                    ← checkpoint at step 500
│   └── ...
├── 1000/                   ← checkpoint at step 1000
│   └── ...
└── 2814/                   ← final checkpoint
    └── ...
```

To load a Trainer checkpoint, point to the step directory itself:

```python
model = CNN.load(
    "./output/2814",
    key=jax.random.key(0),
    num_classes=10,
)
```

> [!NOTE]
> The Trainer auto-rotates checkpoints based on `save_total_limit` in `TrainerConfig`. If set to `2`, only the 2 most recent checkpoints are kept.

## Loading Config-Based Models

If your own model saves a config dataclass into `config.json`, `Z.load(...)` can infer the model class and config class automatically. Constructor arguments are still your responsibility.

```python
import jax
from flax import nnx, struct
from zlynx import Z

@struct.dataclass
class MLPConfig:
    arch: str = "MLP"
    conf: str = "MLPConfig"
    in_features: int = 32
    hidden: int = 64
    out_features: int = 2

class MLP(Z):
    def __init__(self, config: MLPConfig, rngs: nnx.Rngs | None = None):
        if rngs is None:
            rngs = nnx.Rngs(42)
        self.config = config
        self.linear1 = nnx.Linear(config.in_features, config.hidden, rngs=rngs)
        self.linear2 = nnx.Linear(config.hidden, config.out_features, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.relu(self.linear1(x))
        return self.linear2(x)
```

If the checkpoint directory contains `config.json`, `Z.load(...)` can rebuild the architecture automatically:

```python
from zlynx import Z

model = Z.load("./mlp-checkpoint")
```

That works because the saved config contains the config class name and architecture name. In practice, for config-based loading:

- your model should store `self.config`
- your config should include `arch` and `conf`
- if your constructor requires extra arguments such as `rngs` or `key`, either give them defaults or pass them explicitly to `load(...)`

### With dtype casting

```python
import jax.numpy as jnp

# Load in bfloat16 for memory efficiency
model = Z.load("./mlp-checkpoint", dtype=jnp.bfloat16)
```

### With `config_map` and `module_map`

Use these when the checkpoint matches the same underlying model but uses different config field names or parameter/module names.

`config_map` remaps keys from `config.json` before the config object is reconstructed:

```python
model = Z.load(
    "./other-impl-checkpoint",
    config_map={
        "hidden_dim": "hidden",
        "num_classes": "out_features",
    },
)
```

`module_map` remaps checkpoint parameter/module paths during weight loading:

```python
model = Z.load(
    "./other-impl-checkpoint",
    module_map={
        "dense_in": "linear1",
        "dense_out": "linear2",
    },
)
```

Use them only when you already know the two implementations are structurally compatible. They are compatibility helpers, not architecture converters.

## Loading from HuggingFace

Load models directly from HuggingFace Hub:

```python
from zlynx import Z

class MyModel(Z): ...

model = MyModel.load_hf("username/my-model")
```

### With token

```python
model = MyModel.load_hf("username/my-model", token="hf_...")
```

### With sharding

```python
# Load with FSDP sharding across devices
model = MyModel.load_hf("username/my-model", sharding="fsdp")

# Load with data parallel sharding
model = MyModel.load_hf("username/my-model", sharding="ddp")
```

### With dtype

```python
import jax.numpy as jnp

model = MyModel.load_hf("username/my-model", dtype=jnp.bfloat16)
```

### Format options

```python
# Load safetensors format (default for HF)
model = MyModel.load_hf("username/my-model", fmt="safetensors")

# Load orbax format
model = MyModel.load_hf("username/my-model", fmt="orbax")
```

## Loading from Kaggle

Load models directly from Kaggle:

```python
import kagglehub
kagglehub.login()

class MyModel(Z): ...

model = MyModel.load_kaggle("username/my-model")
```

### With variation

```python
model = MyModel.load_kaggle("username/my-model", variation="v1")
```

### With sharding

```python
model = MyModel.load_kaggle("username/my-model", sharding="fsdp")
```

## Pushing to HuggingFace

### Authentication

Choose any one of these authentication methods when you need access to a private repo or want to push without relying on existing local login state.

```python
from huggingface_hub import login
login()
```

Or set a token in your environment:

```bash
export HF_TOKEN=your_token_here
```

Or pass a token directly in the API call:

```python
model.push_hf("username/my-model", private=False, token="hf_...")
```

### Push your model

```python
model.push_hf("username/my-model", private=False)
```

### Private model

```python
model.push_hf("username/my-private-model", private=True)
```

### Format options

```python
# Push safetensors format (default)
model.push_hf("username/my-model", fmt="safetensors")

# Push orbax format
model.push_hf("username/my-model", fmt="orbax")
```

### Safetensors shard size

```python
model.push_hf(
    "username/my-model",
    fmt="safetensors",
    max_shard_size_gb=1,
)
```

## Pushing to Kaggle

### Prerequisites

```python
import kagglehub
kagglehub.login()
```

### Push your model

```python
model.push_kaggle("username/my-model")
```

This uploads to `username/my-model` under the `flax` framework.

### With variation

```python
model.push_kaggle("username/my-model", variation="v2")
```

### Format options

```python
# Push safetensors format (default)
model.push_kaggle("username/my-model", fmt="safetensors")

# Push orbax format
model.push_kaggle("username/my-model", fmt="orbax")
```

## Full Example: Train and Push

```python
from zlynx import Z
from zlynx.trainer import Trainer, TrainerConfig


class MyModel(Z): ...


model = MyModel.load_hf("base-model/weights")

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    train_dataset=train_dataset,
    config=TrainerConfig(num_epochs=3),
)
trainer.train()

model.push_hf("username/my-finetuned-model", private=False)
model.push_kaggle("username/my-finetuned-model")
```

> [!TIP]
> As a rule of thumb, use `orbax` for local checkpoints and `safetensors` for network sharing. In ZLynx, local `save()` / `load()` default to `orbax`, while Hub and Kaggle flows default to `safetensors`.

## Next Steps

Learn how to scale to multiple devices in [Sharding](../useful-stuff/sharding.md).
