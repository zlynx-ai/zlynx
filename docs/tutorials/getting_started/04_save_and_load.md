# Save & Load

How to persist your trained models and restore them later.

---

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

---

## Saving in Different Formats

### Safetensors Format

Save in HuggingFace-compatible safetensors format:

```python
model.save("./my_model", format="safetensors")
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
model.save("./my_model", format="safetensors", max_shard_size_gb=1)
```

---

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

1. Calls your `__init__` with the provided args to build an abstract model skeleton via `nnx.eval_shape`
2. Loads the checkpoint into that skeleton

The `key` can be **any key** — it's only used for structural initialization. The saved weights overwrite all parameters.

---

## Loading Trainer Checkpoints

When you use the `Trainer`, it saves checkpoints automatically using Orbax's `CheckpointManager`:

```
output/checkpoints/
├── 500/                    ← checkpoint at step 500
│   └── default/
│       └── ...
├── 1000/                   ← checkpoint at step 1000
│   └── default/
│       └── ...
└── 2814/                   ← final checkpoint
    └── default/
        └── ...
```

To load a Trainer checkpoint, point to the `default` subdirectory:

```python
model = CNN.load(
    "./output/checkpoints/2814/default",
    key=jax.random.key(0),
    num_classes=10,
)
```

> [!NOTE]
> The Trainer auto-rotates checkpoints based on `save_total_limit` in `TrainerConfig`. If set to `2`, only the 2 most recent checkpoints are kept.

---

## Loading Config-Based Models

For models that use a config dataclass (like `LlamaLanguageModel`), loading is simpler — no extra kwargs needed:

```python
from zlynx import Z

# If the checkpoint directory contains config.json, Z figures out the architecture:
model = Z.load("./llama-checkpoint")
```

Or load with a specific class:

```python
from zlynx.models.llama import LlamaLanguageModel, LlamaConfig

config = LlamaConfig(vocab_size=32000, hidden_size=2048)
model = LlamaLanguageModel.load("./llama-checkpoint", config=config)
```

### With dtype casting

```python
# Load in bfloat16 for memory efficiency
model = LlamaLanguageModel.load("./llama-checkpoint", dtype="bfloat16")
```

---

## Loading from HuggingFace

Load models directly from HuggingFace Hub:

```python
from zlynx import Z

class MyModel(Z): ...

model = MyModel.load_hf("username/my-model")
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
model = MyModel.load_hf("username/my-model", dtype="bfloat16")
```

### Format options

```python
# Load safetensors format (default for HF)
model = MyModel.load_hf("username/my-model", format="safetensors")

# Load orbax format
model = MyModel.load_hf("username/my-model", format="orbax")
```

---

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

---

## Next Steps

Learn how to scale to multiple devices in [Sharding](./05_sharding.md) or how to [Push to Hub](./06_push_to_hub.md).
