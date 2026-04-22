# Quick Start

Get a model running with Zlynx in a few minutes.

This page shows the shortest useful path:

1. Install Zlynx
2. Define a model
3. Train it with `Trainer`
4. Save the checkpoint

Zlynx is currently experimental, so expect some APIs to keep evolving.

## Install

For CPU:

```bash
pip install zlynx
```

For accelerator-specific JAX installs:

```bash
pip install "zlynx[cuda13]"
pip install "zlynx[cuda12]"
pip install "zlynx[tpu]"
# `jax[rocm7-local]` depends on `jax-rocm7-plugin==0.10.0` but currently published is 0.9.1.post3
# pip install "zlynx[rocm7-local]" 
````

If you want the full installation guide, see [Installation](./installation.md).

## Define a Model

```python
import jax
from flax import nnx
from zlynx import Z

class MLP(Z):
    def __init__(self, key, in_features: int, hidden: int, out_features: int):
        super().__init__()
        k1, k2 = jax.random.split(key)

        self.linear1 = nnx.Linear(in_features, hidden, rngs=nnx.Rngs(k1))
        self.linear2 = nnx.Linear(hidden, out_features, rngs=nnx.Rngs(k2))

    def __call__(self, x):
        x = jax.nn.relu(self.linear1(x))
        return self.linear2(x)
```

The important part is that the top-level model inherits from `Z`. That gives you built-in save/load helpers.

If you want more detail, see [Create a model](./create-a-model.md).

## Prepare Data

Zlynx accepts simple Python data containers, Hugging Face datasets, and other iterable or random-access sources.

Here is a tiny in-memory dataset:

```python
import jax
import jax.numpy as jnp

train_data = [
    {
        "input": jax.random.normal(jax.random.key(i), (32,)),
        "target": jnp.int32(i % 2),
    }
    for i in range(128)
]
```

## Define a Loss Function

```python
import optax

def loss_fn(model, batch):
    logits = model(batch["input"])
    labels = batch["target"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean()
```

## Train

```python
from zlynx.trainer import Trainer, TrainerConfig

model = MLP(
    key=jax.random.key(42),
    in_features=32,
    hidden=64,
    out_features=2,
)

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    train_dataset=train_data,
    config=TrainerConfig(
        batch_size=8,
        learning_rate=1e-3,
        num_epochs=3,
        output_dir="./output",
    ),
)

trainer.train()
```

This gives you:

- batching and data pipeline handling
- optimizer construction
- checkpointing
- checkpoints and logs written under `./output`

For a fuller walkthrough, see [Training](./training.md).

## Result

You should see a training summary, periodic metric logs, and a final checkpoint message in the console.

```terminal
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
{'step': 10, 'loss': 0.5651, 'learning_rate': '8.9668e-04', 'epoch': 0.625, 'grad_norm': 1.423, 'steps_per_sec': 5.9827}
{'step': 20, 'loss': 0.711, 'learning_rate': '6.2941e-04', 'epoch': 1.25, 'grad_norm': 1.7601, 'steps_per_sec': 120.8998}
{'step': 30, 'loss': 0.6041, 'learning_rate': '3.0866e-04', 'epoch': 1.875, 'grad_norm': 2.5709, 'steps_per_sec': 185.578}
{'step': 40, 'loss': 0.7822, 'learning_rate': '6.6987e-05', 'epoch': 2.5, 'grad_norm': 1.7409, 'steps_per_sec': 121.2556}
step 48/48: 100%|█████████████████████████████████████████████| 48/48 [00:01<00:00, 24.58it/s]
training complete — 48 steps | saved → ./output
```

## Save and Load

Save a model checkpoint:

```python
model.save("./my_model")
```

Load it back:

```python
restored = MLP.load(
    "./my_model",
    key=jax.random.key(0),
    in_features=32,
    hidden=64,
    out_features=2,
)
```

For format options and checkpoint details, see [Checkpoint](./ckpt.md).

## Next Steps

- Read [Installation](./installation.md) for backend-specific setup
- Read [Create a model](./create-a-model.md) for model structure patterns
- Read [Training](./training.md) for the full trainer workflow
- Read [Sharding](./sharding.md) if you want multi-device execution
- Read [PEFT](./peft.md) and [GaLore](./galore.md) for parameter-efficient training
