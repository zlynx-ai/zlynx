# GaLore Optimizer

Reduce optimizer memory by up to 65% using Gradient Low-Rank Projection — without sacrificing model quality.

## The Problem

Standard AdamW stores **two momentum buffers** (first and second moment) for every parameter. For a 7B parameter model in float32, that's:

```
Model params:    7B × 4 bytes = 28 GB
Adam moments:    7B × 4 bytes × 2 = 56 GB
Total:           84 GB just for parameters + optimizer
```

GaLore solves this by projecting gradients into a low-rank subspace **before** they enter the optimizer. The optimizer only needs to track moments for the projected (much smaller) gradients.

## Quick Start

Just change your optimizer string — everything else stays the same:

```python
from zlynx.trainer import TrainerConfig

config = TrainerConfig(
    optimizer="galore_adamw",
    learning_rate=2e-5,
    optimizer_kwargs={
        "galore_r": 128,                # projection rank
        "galore_update_proj_gap": 200,   # re-compute SVD every N steps
        "galore_scale": 1.0,             # scaling factor
    },
)
```

That's it. The Trainer wraps AdamW with GaLore automatically.

## How It Works

For each 2D weight matrix (≥ 2 dimensions):

1. **Project down**: Compute a low-rank projection of the gradient using SVD
2. **Optimize**: Run AdamW on the small projected gradient (moments are `r`-dimensional, not full-dimensional)
3. **Project up**: Map the optimizer update back to the full parameter space

For 1D parameters (biases, norms), GaLore skips projection and uses standard AdamW.

```
Full gradient G: (m × n)
    ↓ SVD projection
Low-rank gradient: (m × r)  or  (r × n)     ← optimizer runs here
    ↓ project back
Full update: (m × n)
```

### Memory Savings

For a weight matrix of shape `(m, n)`:

|          | AdamW   | GaLore AdamW       |
| -------- | ------- | ------------------ |
| Moment 1 | `m × n` | `m × r` or `r × n` |
| Moment 2 | `m × n` | `m × r` or `r × n` |

With `r = 128` and a `(4096, 4096)` weight: **32× memory reduction** for optimizer states on that layer.

## Configuration Options

| Option                   | Default | Description                                                              |
| ------------------------ | ------- | ------------------------------------------------------------------------ |
| `galore_r`               | `128`   | Rank of the SVD projection. Lower = more compression, less fidelity      |
| `galore_update_proj_gap` | `200`   | Steps between SVD recomputations. Higher = faster but staler projections |
| `galore_scale`           | `1.0`   | Scaling factor applied to the final projected updates                    |

### Choosing the Rank

- **`r = 128`** — good default for most large models
- **`r = 64`** — more aggressive compression, use for very large models
- **`r = 256`** — less compression, closer to full AdamW quality

The projection is only applied to 2D+ tensors (weight matrices). Parameters with `min(shape) ≤ r` use standard AdamW automatically.

### Choosing the Update Gap

- **`200`** — good default. SVD is expensive, so doing it every step would be slow
- **`100`** — more frequent updates, better quality but slower
- **`500`** — less frequent, faster training but projections may get stale

## Full Example

```python
from zlynx.models.llama import LlamaConfig, LlamaLanguageModel
from zlynx.trainer import Trainer, TrainerConfig

config = LlamaConfig(vocab_size=32000, hidden_size=2048, num_hidden_layers=16, head_dim=64)
model = LlamaLanguageModel(config)

trconfig = TrainerConfig(
    optimizer="galore_adamw",
    learning_rate=2e-5,
    weight_decay=0.01,
    lr_scheduler="warmup_cosine_decay",
    warmup_steps=100,
    num_epochs=3,
    batch_size=32,
    optimizer_kwargs={
        "galore_r": 128,
        "galore_update_proj_gap": 200,
        "galore_scale": 1.0,
    },
    sharding="fsdp",
    processing_train_dataset_fn=preprocess,
)

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    train_dataset=train_data,
    config=trconfig,
)
trainer.train()
```

## GaLore vs PEFT

These are complementary techniques that solve different problems:

|                     | GaLore                                   | PEFT (LoRA etc.)                        |
| ------------------- | ---------------------------------------- | --------------------------------------- |
| **What it reduces** | Optimizer memory                         | Trainable parameter count               |
| **What it trains**  | All parameters (full fine-tuning)        | Only adapter parameters                 |
| **Where it acts**   | Inside the optimizer                     | Inside the model architecture           |
| **Quality**         | Closest to full fine-tuning              | Slightly lower (depends on rank/method) |
| **Can combine?**    | Yes — you can use GaLore + LoRA together | Yes                                     |

**Use GaLore when**: You want full fine-tuning quality but can't fit AdamW moments in memory.

**Use PEFT when**: You want to train a tiny fraction of parameters (fast, portable adapters).

**Use both when**: You're applying PEFT but even the adapter optimizer states are too large (very constrained hardware).

## Next Steps

Related pages:

- [PEFT](./peft.md) — parameter-efficient fine-tuning methods
- [Sharding](./sharding.md) — multi-device execution and placement
- [Logging Backend](./logging-backend.md) — optional experiment logging integrations
- [MNIST](../examples/mnist.md) — complete end-to-end example
