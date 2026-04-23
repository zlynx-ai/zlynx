# Logging Backend

ZLynx logs training progress to the console by default, and can also send metrics to TensorBoard or Weights & Biases.

## Default Behavior

Even with `log_to=[]`, the trainer still prints:

- the training banner at startup
- periodic metric logs every `logging_steps`
- checkpoint save messages
- the final `training complete` line

```python
from zlynx.trainer import TrainerConfig

config = TrainerConfig(
    logging_steps=10,
)
```

Console logs look like:

```python
{'step': 10, 'loss': 0.6934, 'learning_rate': '8.9668e-04', 'epoch': 0.625, 'grad_norm': 1.8421, 'steps_per_sec': 5.7312}
```

## `logging_steps`

`logging_steps` controls how often training metrics are emitted:

```python
config = TrainerConfig(
    logging_steps=100,
)
```

This affects:

- console metric frequency
- TensorBoard logging frequency
- W&B logging frequency

## `logging_aux`

If your `loss_fn` returns a dict, ZLynx always uses the `"loss"` key for backpropagation.

When `logging_aux=True` (default), scalar auxiliary values are logged automatically:

```python
def loss_fn(model, batch):
    logits = model(batch["input"])
    loss = ...
    accuracy = ...
    return {"loss": loss, "accuracy": accuracy}
```

```python
config = TrainerConfig(
    logging_aux=True,
)
```

That will log `"accuracy"` automatically if it is a scalar.

## `logging_fn`

Use `logging_fn` when you want derived metrics or when `logging_aux=False`.

```python
import jax.numpy as jnp

config = TrainerConfig(
    logging_aux=False,
    logging_fn={
        "perplexity": lambda **kw: jnp.exp(kw["loss"]),
    },
)
```

Each function in `logging_fn` receives the full auxiliary dict returned by `loss_fn` as `**kwargs`.

## TensorBoard

Enable TensorBoard logging with:

```python
config = TrainerConfig(
    log_to=["tensorboard"],
)
```

Current implementation details:

- uses `torch.utils.tensorboard.SummaryWriter`
- writes logs under `output_dir/tb_logs`
- logs scalar metrics only

Example:

```python
config = TrainerConfig(
    output_dir="./output",
    log_to=["tensorboard"],
)
```

## Weights & Biases

Enable W&B logging with:

```python
config = TrainerConfig(
    log_to=["wandb"],
    run_name="my-experiment",
)
```

Current implementation details:

- imports `wandb` directly
- calls `wandb.init(project=run_name or "zlynx", name=run_name)` if no run exists yet
- logs metrics with `wandb.log(metrics, step=step)`
- calls `wandb.finish()` when training ends

## Multiple Backends

You can combine backends:

```python
config = TrainerConfig(
    log_to=["tensorboard", "wandb"],
    run_name="my-experiment",
)
```

This does not disable console logging. It adds the selected backends on top of it.

## Training Metrics File

ZLynx also writes a summary file to:

```text
output_dir/training_metrics.json
```

This file is maintained separately from `log_to` and includes:

- config values such as optimizer, batch size, scheduler, and sharding
- per-step logged metrics
- evaluation metrics when eval is enabled
- start and end timestamps

## Notes

- TensorBoard support requires `torch` to be available because the current implementation imports `torch.utils.tensorboard`.
- W&B support requires the `wandb` package to be installed and configured.
- Notebook environments are handled differently: ZLynx displays a live table instead of printing plain console dicts.

## Next Steps

- [Training](../getting-started/training.md) — trainer configuration and loss functions
- [Checkpoint](../getting-started/ckpt.md) — save, load, and sharing
