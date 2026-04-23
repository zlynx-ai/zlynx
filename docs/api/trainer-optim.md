# Trainer: Optimizer and Schedulers

Optimizer helpers live in `zlynx.trainer.optim`.

## Main Helpers

```python
from zlynx.trainer.optim import build_optimizer, get_optimizer, get_scheduler
```

- `get_optimizer(config)`
- `get_scheduler(config)`
- `build_optimizer(config, total_steps)`

## Optimizers

`config.optimizer` can be:

- a string such as `"adamw"` or `"sgd"`
- a callable
- a `galore_*` optimizer name such as `"galore_adamw"`

GaLore support is implemented through:

- `galore_wrapper(...)`
- `update_galore_projectors(...)`

Using a `galore_*` optimizer requires `scikit-learn` for randomized SVD.

## Schedulers

`config.lr_scheduler` can be a string or a callable.

Built-in scheduler names include:

- `constant`
- `linear`
- `cosine`
- `cosine_onecycle`
- `exponential`
- `linear_onecycle`
- `piecewise_constant`
- `piecewise_interpolate`
- `polynomial`
- `sgdr`
- `warmup_constant`
- `warmup_cosine_decay`
- `warmup_exponential_decay`

## Important Detail

Warmup is not generic across every scheduler name.

Example:

- `lr_scheduler="cosine"` uses cosine decay only
- `lr_scheduler="warmup_cosine_decay"` uses warmup and cosine decay

That distinction matters when you set `warmup_steps` or `warmup_ratio`.
