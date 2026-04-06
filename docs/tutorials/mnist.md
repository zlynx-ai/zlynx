# Train an MNIST Classifier with Zlynx

A step-by-step guide to training a CNN on the MNIST handwritten digit dataset using Zlynx's `Trainer`.

> **What you'll learn**
>
> - Defining a model that inherits from `Z`
> - Preparing a dataset for Zlynx's grain-based data pipeline
> - Writing a JIT-compatible loss function
> - Configuring `TrainerConfig`
> - Running training, evaluating accuracy, and saving/loading checkpoints

---

## Prerequisites

```bash
uv pip install zlynx torchvision
```

> [!NOTE]
> Torchvision is only used to download MNIST. Zlynx has **no dependency on PyTorch**.

---

## 1 — Imports

```python
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torchvision import datasets

from flax import nnx
from zlynx import Z
from zlynx.trainer import Trainer, TrainerConfig
```

| Import                      | Purpose                                                   |
| --------------------------- | --------------------------------------------------------- |
| `jax` / `jnp`               | Numerical computing and array operations                  |
| `optax`                     | Loss functions and optimizers                             |
| `torchvision.datasets`      | One-liner MNIST download                                  |
| `nnx`                       | Flax NNX neural network layers (`Conv`, `Linear`, …)      |
| `Z`                         | Zlynx base model — adds `save()` and `load()`             |
| `Trainer` / `TrainerConfig` | Turn-key training loop with logging & checkpointing       |

---

## 2 — Define the Model

```python
class CNNClassifier(Z):
    def __init__(self, key, num_classes: int = 10):
        super().__init__()

        # Split random key for each layer
        conv1_key, conv2_key, fc_key = jax.random.split(key, 3)

        # Conv block 1:  28×28×1 → 26×26×32 → 13×13×32
        self.conv1 = nnx.Conv(
            in_features=1,
            out_features=32,
            kernel_size=(3, 3),
            padding="VALID",
            rngs=nnx.Rngs(conv1_key),
        )

        # Conv block 2:  13×13×32 → 11×11×64 → 5×5×64
        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=64,
            kernel_size=(3, 3),
            padding="VALID",
            rngs=nnx.Rngs(conv2_key),
        )

        # Classifier head:  5×5×64 = 1600 → num_classes
        self.fc = nnx.Linear(
            in_features=64 * 5 * 5,
            out_features=num_classes,
            rngs=nnx.Rngs(fc_key),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (batch, 28, 28, 1)
        x = jax.nn.relu(self.conv1(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = jax.nn.relu(self.conv2(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape(x.shape[0], -1)   # flatten → (batch, 1600)
        return self.fc(x)                # logits  → (batch, num_classes)
```

**Key points:**

- Inherit from **`Z`** to get `save()` / `load()` for free. Only the outermost model needs `Z` — inner layers like `nnx.Conv` are plain Flax modules.
- We use **`jax.random.split`** to create separate random keys for each layer, then wrap each with `nnx.Rngs(...)` for Flax NNX.
- The forward pass: **Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Linear**.

---

## 3 — Load MNIST

```python
print("Loading MNIST dataset …")

train_ds = datasets.MNIST(root="./data", train=True,  download=True)
test_ds  = datasets.MNIST(root="./data", train=False, download=True)

# Extract raw NumPy arrays
x_train = train_ds.data.numpy().astype(np.float32)   # (60000, 28, 28)
y_train = np.array(train_ds.targets.numpy())           # (60000,)
x_test  = test_ds.data.numpy().astype(np.float32)      # (10000, 28, 28)
y_test  = np.array(test_ds.targets.numpy())             # (10000,)

print(f"Train: {len(x_train)} | Test: {len(x_test)}")
```

---

## 4 — Prepare the Dataset for Zlynx

Zlynx wraps datasets with [**Google Grain**](https://github.com/google/grain) for high-performance batching and shuffling. The simplest input format is a **list of dicts**:

```python
train_data = [
    {"image": x_train[i], "label": y_train[i]}
    for i in range(len(x_train))
]
```

Each element is a `dict` with:

- `"image"` — a `(28, 28)` float array
- `"label"` — an integer `0–9`

> [!TIP]
> You can also pass a Hugging Face `datasets.Dataset`, a local dataset path string, or a plain Python `dict` of arrays.

---

## 5 — Define Preprocessing

The `processing_train_dataset_fn` in `TrainerConfig` is called on **each batch** before it reaches the loss function:

```python
def preprocess(batch):
    image = batch["image"].astype(jnp.float32) / 255.0   # normalize [0,255] → [0,1]
    image = image[..., None]                               # add channel dim: (B,28,28) → (B,28,28,1)
    label = batch["label"]
    return {"image": image, "label": label}
```

---

## 6 — Define the Loss Function

The Trainer expects a callable with signature `(model, batch) → scalar`. By the time it reaches the loss function, data **has** been batched by Grain:

```python
def loss_fn(model, batch):
    logits = model(batch["image"])       # (B, 10)
    labels = batch["label"]              # (B,)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean()
```

> [!IMPORTANT]
> The loss function is **JIT-compiled** under the hood. Avoid Python side effects like `print()` or list mutations inside it.

---

## 7 — Create the Model

```python
key = jax.random.key(42)
model_key, _ = jax.random.split(key)

model = CNNClassifier(model_key, num_classes=10)
```

JAX requires explicit PRNG keys for reproducibility. We split a key and pass one to the model constructor.

---

## 8 — Configure the Trainer

```python
config = TrainerConfig(
    per_device_batch_size=64,
    learning_rate=1e-3,
    num_epochs=3,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    output_dir="./output",
    sharding=False,                        # single device
    processing_train_dataset_fn=preprocess,
    seed=42,
)
```

### Key TrainerConfig Options

| Option                        | Default      | Description                                                        |
| ----------------------------- | ------------ | ------------------------------------------------------------------ |
| `per_device_batch_size`       | `1`          | Samples per device per training step                               |
| `gradient_accumulation_steps` | `1`          | Micro-batches per optimizer update                                 |
| `optimizer`                   | `"adamw"`    | Any Optax optimizer name or callable                               |
| `learning_rate`               | `5e-5`       | Peak learning rate                                                 |
| `weight_decay`                | `0.0`        | L2 regularization strength                                         |
| `max_grad_norm`               | `1.0`        | Global gradient clipping (`None` to disable)                       |
| `lr_scheduler`                | `"cosine"`   | `"cosine"` · `"linear"` · `"constant"` · `"warmup_cosine_decay"`   |
| `warmup_steps`                | `0`          | LR warmup steps (or use `warmup_ratio`)                            |
| `num_epochs`                  | `1`          | Training epochs (ignored if `max_steps > 0`)                       |
| `max_steps`                   | `-1`         | Hard step limit (`-1` = use epochs)                                |
| `sharding`                    | `"auto"`     | `"auto"` · `"ddp"` · `"fsdp"` · `False` · `None` · `<int>`       |
| `save_steps`                  | `500`        | Checkpoint interval                                                |
| `save_total_limit`            | `3`          | Max checkpoints kept (auto-rotated via Orbax)                      |
| `log_to`                      | `[]`         | `"wandb"` · `"tensorboard"`                                       |
| `logging_fn`                  | `None`       | Custom metrics, e.g. `{"ppl": lambda **kw: jnp.exp(kw["loss"])}`   |
| `processing_train_dataset_fn` | `None`       | Per-batch preprocessing function                                   |

---

## 9 — Train

```python
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    train_dataset=train_data,
    config=config,
)

print("Starting training …")
trainer.train()
```

Expected output:

```
step 100 | loss: 0.6132 | step: 100 | epoch: 0 | steps_per_sec: 7.39
step 200 | loss: 0.1854 | step: 200 | epoch: 0 | steps_per_sec: 8.12
...
training complete — 2814 steps | saved → ./output
```

---

## 10 — Evaluate

After training, measure accuracy on the test set:

```python
# Preprocess test images (same normalization as training)
x_test_norm = jnp.array(x_test, dtype=jnp.float32) / 255.0
x_test_norm = x_test_norm[..., None]                         # (10000, 28, 28, 1)
y_test_jnp  = jnp.array(y_test)

# Forward pass on first 1000 test samples
logits = model(x_test_norm[:1000])
preds  = jnp.argmax(logits, axis=-1)
accuracy = jnp.mean(preds == y_test_jnp[:1000])

print(f"Test accuracy: {accuracy:.4f}")
# Expected: ~0.99 after 3 epochs
```

> [!TIP]
> For larger test sets, loop over batches to avoid OOM:
>
> ```python
> correct, total = 0, 0
> for i in range(0, len(x_test_norm), 256):
>     logits = model(x_test_norm[i:i+256])
>     preds = jnp.argmax(logits, axis=-1)
>     correct += int(jnp.sum(preds == y_test_jnp[i:i+256]))
>     total += len(x_test_norm[i:i+256])
> print(f"Test accuracy: {correct / total:.4f}")
> ```

---

## 11 — Save & Load the Model

### Manual Save

```python
model.save("./my_mnist_model")
# Saves weights via Orbax + config.json (if the model has a config attribute)
```

### Load

Because Flax NNX needs to reconstruct the model structure before loading weights, you must pass the **same init arguments** you used when creating the model:

```python
model = CNNClassifier.load(
    "./my_mnist_model",
    key=jax.random.key(0),    # any key will do — weights get overwritten
    num_classes=10,
)
```

> [!NOTE]
> The `key` argument can be any key — it's only used to initialize the model skeleton so Orbax knows what shapes to load. The saved weights overwrite everything.

### Loading a Trainer Checkpoint

The Trainer saves checkpoints with Orbax's `CheckpointManager` in this structure:

```
output/checkpoints/
├── 500/
│   └── default/
│       └── ...          ← weight files
├── 1000/
│   └── default/
│       └── ...
└── 2814/               ← final checkpoint
    └── default/
        └── ...
```

To load a specific checkpoint:

```python
model = CNNClassifier.load(
    "./output/checkpoints/2814/default",
    key=jax.random.key(0),
    num_classes=10,
)
```

---

## Full Script

<details>
<summary>Click to expand the complete, copy-pasteable script</summary>

```python
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torchvision import datasets

from flax import nnx
from zlynx import Z
from zlynx.trainer import Trainer, TrainerConfig


# ── Model ────────────────────────────────────────────────────
class CNNClassifier(Z):
    def __init__(self, key, num_classes: int = 10):
        super().__init__()
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding="VALID", rngs=nnx.Rngs(k1))
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="VALID", rngs=nnx.Rngs(k2))
        self.fc    = nnx.Linear(64 * 5 * 5, num_classes, rngs=nnx.Rngs(k3))

    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = jax.nn.relu(self.conv2(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


# ── Dataset ──────────────────────────────────────────────────
train_ds = datasets.MNIST(root="./data", train=True,  download=True)
test_ds  = datasets.MNIST(root="./data", train=False, download=True)

x_train, y_train = train_ds.data.numpy().astype(np.float32), np.array(train_ds.targets.numpy())
x_test,  y_test  = test_ds.data.numpy().astype(np.float32),  np.array(test_ds.targets.numpy())

train_data = [{"image": x_train[i], "label": y_train[i]} for i in range(len(x_train))]


# ── Preprocessing (per-batch) & loss (per-batch) ────────────
def preprocess(batch):
    image = batch["image"].astype(jnp.float32) / 255.0
    return {"image": image[..., None], "label": batch["label"]}

def loss_fn(model, batch):
    logits = model(batch["image"])
    return optax.softmax_cross_entropy_with_integer_labels(logits, batch["label"]).mean()


# ── Config ───────────────────────────────────────────────────
config = TrainerConfig(
    per_device_batch_size=64, learning_rate=1e-3, num_epochs=3,
    logging_steps=100, save_steps=500, save_total_limit=2,
    output_dir="./output", sharding=False,
    processing_train_dataset_fn=preprocess, seed=42,
)

# ── Train ────────────────────────────────────────────────────
key = jax.random.key(42)
model_key, _ = jax.random.split(key)
model = CNNClassifier(model_key, num_classes=10)

trainer = Trainer(
    model=model, loss_fn=loss_fn,
    train_dataset=train_data, config=config,
)
trainer.train()


# ── Evaluate ─────────────────────────────────────────────────
x_t = jnp.array(x_test, dtype=jnp.float32) / 255.0
x_t = x_t[..., None]
y_t = jnp.array(y_test)

preds = jnp.argmax(model(x_t[:1000]), axis=-1)
print(f"Test accuracy: {jnp.mean(preds == y_t[:1000]):.4f}")


# ── Save ─────────────────────────────────────────────────────
model.save("./my_mnist_model")
```

</details>

---

## What's Next?

Now that you've trained your first model with Zlynx, try these:

- **Bigger architectures** — add more conv blocks, batch normalization, or residual connections
- **Hyperparameter tuning** — experiment with `learning_rate`, `per_device_batch_size`, `lr_scheduler`, `warmup_steps`
- **Multi-device training** — set `sharding="auto"` (or `"ddp"` / `"fsdp"`) to distribute across GPUs or TPUs
- **PEFT fine-tuning** — apply LoRA, DoRA, VeRA, and more with `apply_peft()` from `zlynx.modules.peft`
- **GaLore optimizer** — use `optimizer="galore_adamw"` to reduce optimizer memory via gradient low-rank projection
- **Logging** — add `"wandb"` or `"tensorboard"` to `log_to` for richer experiment tracking

Check out the [Advanced Tutorials](./advanced/01_peft.md) to dive deeper!
