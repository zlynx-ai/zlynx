# Trainer

`zlynx.trainer` contains the training loop and its configuration surface.

See also:

- [TrainerConfig](./trainer-config.md)
- [Optimizer and Schedulers](./trainer-optim.md)

## Common Imports

```python
from zlynx.trainer import Trainer, TrainerConfig
```

## `Trainer`

```python
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    config=TrainerConfig(),
)
```

Accepted dataset forms include:

- any iterable of batches or samples
- dict-of-columns datasets such as `{"input": x, "target": y}`
- Hugging Face dataset repository IDs such as `"ylecun/mnist"`
- local dataset paths handled through the configured loaders

Common methods:

- `train()`
- `eval()`

The trainer uses Orbax-managed checkpoints internally under `output_dir`.

## `TrainerConfig`

`TrainerConfig` is a dataclass with the main knobs for training.

Common fields:

- `batch_size`
- `gradient_accumulation_steps`
- `optimizer`
- `optimizer_kwargs`
- `learning_rate`
- `weight_decay`
- `lr_scheduler`
- `warmup_steps`
- `warmup_ratio`
- `max_steps`
- `num_epochs`
- `sharding`
- `remat`
- `output_dir`
- `save_steps`
- `log_to`
- `logging_steps`
- `logging_aux`
- `logging_fn`
- `jit_loss_fn`

Example:

```python
from zlynx.trainer import TrainerConfig

config = TrainerConfig(
    batch_size=32,
    learning_rate=3e-4,
    num_epochs=3,
    sharding="auto",
    output_dir="./output",
)
```

## Notes

- `max_steps` is the hard stop when it is set.
- `logging_aux=True` logs scalar auxiliary values returned by `loss_fn`.
- `logging_fn` is for derived or custom metrics.
- `sharding="auto"` is a convenience mode, not a guarantee of optimal partitioning.

For end-to-end usage, see the getting-started pages for [Training](../getting-started/training.md) and [Checkpointing](../getting-started/ckpt.md).
