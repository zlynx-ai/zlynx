# Zlynx

> [!CAUTION]
> **Zlynx is currently an experimental library.** APIs are subject to change without notice. We recommend pinning versions for production use.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zlynx-ai/zlynx)

An experimental, lightweight deep learning library built on **JAX** and **Flax NNX**. It explores providing researchers and developers with fine-grained control over model architectures, training loops, and distributed setups.

## Install

```bash
uv pip install zlynx
```

## Experimental Model Interface

Zlynx introduces `Z` as an experimental base class for models, aiming to provide built-in utilities for saving, loading, and sharing model artifacts.

```python
import jax
from flax import nnx
from zlynx import Z

class MyModel(Z):
    def __init__(self, rngs: nnx.Rngs, in_features: int, hidden: int, out_features: int):
        self.linear1 = nnx.Linear(in_features, hidden, rngs=rngs)
        self.linear2 = nnx.Linear(hidden, out_features, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.relu(self.linear1(x))
        return self.linear2(x)

# Initialize
model = MyModel(nnx.Rngs(42), in_features=784, hidden=256, out_features=10)

# Save & Load
model.save("./my-model", format="safetensors")
restored = MyModel.load("./my-model", rngs=nnx.Rngs(0), in_features=784, hidden=256, out_features=10)

# Push to Hubs (Experimental)
model.push_hf("username/my-model")
model.push_kaggle("username/my-model")
```

## Training

The `Trainer` is designed to handle common tasks of the training loop, including optimization, checkpointing, and logging.

```python
from zlynx.trainer import Trainer, TrainerConfig

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    train_dataset=dataset,
    config=TrainerConfig(
        batch_size=32,
        learning_rate=5e-5,
        num_epochs=3,
        sharding="auto", # Experimental sharding
    ),
)
trainer.train()
```

## PEFT (LoRA, DoRA, VeRA, LoHa, LoKr, AdaLoRA)

Zlynx provides utilities to apply various parameter-efficient fine-tuning (PEFT) methods.

```python
from zlynx.module.peft import apply_peft

model = apply_peft(
    model, 
    method="lora", 
    r=16, 
    alpha=32, 
    target_modules=["linear1", "linear2"]
)
```

## Goals & Features

- **JAX + Flax NNX Integration** — Explores combining XLA speed with the flexible NNX module system.
- **Checkpointing Utilities** — Experimental support for Orbax + SafeTensors, HuggingFace Hub, and Kaggle.
- **Training Helpers** — Gradient accumulation, LR scheduling, and multi-backend logging.
- **Sharding Support** — Initial support for transitioning between single-device and distributed (DDP, FSDP) training.
- **PEFT Methods** — Implementation of 6 adapter methods via `apply_peft()`.
- **GaLore** — Experimental gradient low-rank projection for memory-efficient fine-tuning.

## Documentation

For full guides and current API references, visit the [Documentation](./docs/index.md).

- [Installation](./docs/getting-started/installation.md)
- [Quick Start](./docs/getting-started/quick-start.md)
- [MNIST Tutorial](./docs/examples/mnist.md)
- [API Reference](./docs/api/index.md)
