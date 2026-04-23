# Trainer: TrainerConfig

`TrainerConfig` is the main dataclass that controls training behavior.

```python
from zlynx.trainer import TrainerConfig
```

## Common Fields

### Batch and duration

- `batch_size`
- `gradient_accumulation_steps`
- `max_steps`
- `num_epochs`

### Dataset loading

- `train_subset`
- `eval_subset`
- `train_split`
- `eval_split`
- `streaming`
- `train_streaming`
- `eval_streaming`
- `load_train_dataset_fn`
- `load_eval_dataset_fn`

### Processing

- `processing_train_dataset_fn`
- `processing_eval_dataset_fn`
- `seed`

### Optimizer and schedule

- `optimizer`
- `optimizer_kwargs`
- `learning_rate`
- `weight_decay`
- `max_grad_norm`
- `lr_scheduler`
- `warmup_steps`
- `warmup_ratio`

### Evaluation

- `eval_strategy`
- `eval_max_steps`
- `eval_steps`
- `eval_epochs`
- `eval_batch_size`

### System behavior

- `jit_loss_fn`
- `remat`
- `sharding`

### Checkpointing and logging

- `output_dir`
- `save_steps`
- `save_total_limit`
- `resume_from`
- `logging_steps`
- `log_to`
- `run_name`
- `logging_aux`
- `logging_fn`

### Extension hooks

- `build_optimizer_fn`
- `return_optimizer_extra_args_fn`

## Notes

- `max_steps` overrides epoch-based stopping when set.
- plain `cosine` does not automatically use warmup; use `warmup_cosine_decay` if you want warmup plus cosine decay.
- `logging_aux=True` logs scalar auxiliary values returned by `loss_fn` automatically.
