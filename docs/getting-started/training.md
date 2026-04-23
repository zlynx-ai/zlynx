# Training

Learn how to train any nnx model using zlynx `Trainer` — from data loading to checkpointing.

## Overview

The Zlynx training workflow has three components:

```
Dataset → Loss Function → TrainerConfig → Trainer.train()
```

1. **Dataset** — your raw data (list, HF dataset, or path)
2. **Loss function** — `(model, batch) → scalar` or `(model, batch) → {"loss": ..., ...}`
3. **TrainerConfig** — hyperparameters, data processing, logging, checkpointing

## Step 1: Prepare Your Dataset

Zlynx accepts several dataset formats, including columnar dict datasets, and more generally any iterable dataset, including streaming datasets:

| Format                       | Example                          |
| ---------------------------- | -------------------------------- |
| **List of dicts**            | `[{"x": array, "y": 1}, ...]`    |
| **HF `datasets.Dataset`**    | `datasets.load_dataset("ylecun/mnist")` |
| **HF dataset name** (string) | `"openai/gsm8k"`                 |
| **Dict of arrays**           | `{"x": big_array, "y": labels}`  |

For dict datasets, each field must have the same leading length. For example, if `x`
has shape `(N, features)` and `y` has shape `(N,)`, ZLynx treats that as `N` samples.

The simplest format is a list of dicts:

```python
train_data = [
    {"input": x_train[i], "target": y_train[i]}
    for i in range(len(x_train))
]
```

You can use any key names — just be consistent with your preprocessing and loss functions.

## Step 2: Configure Data Processing

Data processing is configured directly inside `TrainerConfig`:

```python
from zlynx.trainer import TrainerConfig

config = TrainerConfig(
    processing_train_dataset_fn=preprocess,    # per-batch transform
    seed=42,
    num_workers=4,
)
```

### The `processing_train_dataset_fn`

This function preprocesses each batch before it reaches `loss_fn`:

```python
def preprocess(batch):
    images = batch["input"].astype(jnp.float32) / 255.0
    images = images[..., None]   # add channel dim
    return {"input": images, "target": batch["target"]}
```

### Loading HF Datasets Directly

Pass a Hugging Face dataset repository ID string as the dataset, and configure with `TrainerConfig`:

```python
config = TrainerConfig(
    train_split="train",
    train_subset="default",
    processing_train_dataset_fn=preprocess,
)

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    train_dataset="ylecun/mnist",
    config=config,
)
```

## Step 3: Define the Loss Function

The loss function receives the **model** and a **batched** dict. It must return a scalar and be JIT-compatible:

```python
import optax

def loss_fn(model, batch):
    logits = model(batch["input"])       # forward pass
    labels = batch["target"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean()
```

If your loss path or batch structure is not JIT-friendly yet, you can disable loss-function
JIT in `TrainerConfig`:

```python
config = TrainerConfig(
    ...,
    jit_loss_fn=False,
)
```

> [!IMPORTANT]
> By the time data reaches `loss_fn`, Grain has already batched it. So `batch["input"]` has shape `(batch_size, ...)`.

### Returning Extra Metrics

Your loss function can return a **dict** instead of a scalar. The `"loss"` key is used for backpropagation:

```python
def loss_fn(model, batch):
    logits = model(batch["input"])
    labels = batch["target"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == labels)

    return {"loss": loss, "accuracy": accuracy}
```

By default, `logging_aux=True`, so scalar auxiliary values such as `"accuracy"` are logged automatically.

If you set `logging_aux=False`, auxiliary values are not logged automatically. In that case, use
`logging_fn` to choose what to log or to compute derived metrics:

```python
config = TrainerConfig(
    ...,
    logging_aux=False,
    logging_fn={
        "accuracy": lambda **kw: kw["accuracy"],
    },
)
```

## Step 4: Configure the Trainer

```python
from zlynx.trainer import TrainerConfig

config = TrainerConfig(
    # Batch
    batch_size=64,
    gradient_accumulation_steps=1,

    # Optimizer
    optimizer="adamw",
    learning_rate=1e-3,
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Schedule
    lr_scheduler="warmup_cosine_decay",
    warmup_steps=100,

    # Duration
    num_epochs=3,
    # max_steps=-1,               # set to override epochs

    # Checkpointing
    output_dir="./output",
    save_steps=500,
    save_total_limit=2,

    # Logging
    logging_steps=100,

    # Device
    sharding=False,                # single device (see sharding tutorial)
)
```

### Optimizers

All Optax core and contrib optimizers are supported. Common ones:

| Value            | Optimizer                            |
| ---------------- | ------------------------------------ |
| `"adamw"`        | AdamW (default)                      |
| `"adam"`         | Adam                                 |
| `"sgd"`          | SGD                                  |
| `"lion"`         | Lion                                 |
| `"muon"`         | Muon                                 |
| `"galore_adamw"` | AdamW + Gradient Low-Rank Projection |

### Learning Rate Schedules

| Value                     | Behavior                     |
| ------------------------- | ---------------------------- |
| `"constant"`              | Fixed learning rate          |
| `"linear"`                | Linear decay to 0            |
| `"cosine"`                | Cosine decay to 0 (default)  |
| `"warmup_cosine_decay"`   | Linear warmup → cosine decay |
| `"warmup_constant"`       | Linear warmup → constant     |

### Gradient Accumulation

Simulate larger batch sizes on limited hardware:

```python
config = TrainerConfig(
    batch_size=16,            # micro-batch
    gradient_accumulation_steps=4,       # effective batch = 16 × 4 = 64
)
```

The Trainer handles accumulating gradients across micro-batches and averaging them before each optimizer step.

<!-- ### Logging Backends

```python
config = TrainerConfig(
    log_to=["wandb", "tensorboard"],
    run_name="my-experiment",
)
```

| Backend         | Output                                     |
| --------------- | ------------------------------------------ |
| `"wandb"`       | Weights & Biases (auto-inits a run)        |
| `"tensorboard"` | TensorBoard logs in `output_dir/tb_logs/`  | -->

## Step 5: Train

```python
from zlynx.trainer import Trainer

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    train_dataset=train_data,
    config=config,
)

trainer.train()
```

The Trainer handles everything:

- Building the optimizer and LR schedule
- JIT-compiled training steps
- Gradient accumulation
- Periodic logging and checkpointing
- Orbax checkpoint rotation (keeps only `save_total_limit` most recent)

### Expected Output

```bash
════════════════════════════ Zlynx ═════════════════════════════
Model            : MLP
Parameters       : 2.24K total | 2.24K trainable
Devices          : 1 x cpu
Sharding         : False
Remat            : disabled

Batch size       : 8
Grad accum       : 1
Effective batch  : 8

Optimizer        : adamw
Learning rate    : 1.00e-03
Scheduler        : cosine
Epochs           : 3
Max steps        : 48

Output dir       : ./output
════════════════════════════════════════════════════════════════
{'step': 10, 'loss': 0.6934, 'learning_rate': '8.9668e-04', 'epoch': 0.625, 'grad_norm': 1.8421, 'steps_per_sec': 5.7312}
{'step': 20, 'loss': 0.6712, 'learning_rate': '6.2941e-04', 'epoch': 1.25, 'grad_norm': 1.5038, 'steps_per_sec': 34.2285}
saved checkpoint → step 20
...
training complete — 48 steps | saved → ./output
```

## Quick Reference

```python
# Minimal training setup
from zlynx.trainer import Trainer, TrainerConfig

trainer = Trainer(
    model=model,
    loss_fn=lambda model, batch: ...,       # (model, batch) → scalar
    train_dataset=data,                     # list, HF dataset, or string
    config=TrainerConfig(batch_size=32, num_epochs=5),
)
trainer.train()
```

## Next Steps

Learn how to save and load your trained model in [Save & Load](./ckpt.md).
