# Trainer

The `zlynx.trainer` module provides a flexible, JAX-native training framework for Flax NNX models. It handles optimizer construction, gradient accumulation, multi-backend logging, checkpointing, sharding, and data pipeline creation via [Grain](https://github.com/google/grain).

```python
from zlynx.trainer import Trainer, TrainerConfig, build_optimizer
```

---

## `Trainer`

The core training class. Accepts a model, a loss function, datasets, and a configuration, then runs a complete training loop with gradient accumulation, logging, checkpointing, and optional evaluation.

```python
class Trainer(
    model,
    loss_fn: Callable,
    train_dataset,
    eval_dataset: Optional[Any] = None,
    config: TrainerConfig | None = None,
)
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `model` | `nnx.Module` | The Flax NNX model to train. |
| `loss_fn` | `Callable` | A callable with signature `loss_fn(model, batch)`. May return a scalar loss **or** a dict containing at least `"loss"` (extra keys are forwarded to `logging_fn`). |
| `train_dataset` | `Any` | Training data — can be a HuggingFace dataset, a list, a dict, a string identifier, or any iterable/random-access source. |
| `eval_dataset` | `Any \| None` | Optional evaluation dataset (same types as `train_dataset`). |
| `config` | `TrainerConfig \| None` | Trainer configuration. Defaults to `TrainerConfig()`. |

### `loss_fn` Contract

The loss function receives `(model, batch)` and may return:

1. **A scalar loss** — used directly for backpropagation.
2. **A dictionary** — must contain `"loss"`. Extra keys are forwarded to custom `logging_fn` callables as `**kwargs`.

```python
# Scalar return
def loss_fn(model, batch):
    logits = model(batch["input_ids"])
    return cross_entropy(logits, batch["labels"])

# Dict return (enables custom logging)
def loss_fn(model, batch):
    logits = model(batch["input_ids"])
    loss = cross_entropy(logits, batch["labels"])
    return {"loss": loss, "per_token_loss": loss}
```

### `Trainer.train()`

Runs the full training loop:

- Builds the optimizer and LR schedule from config (or via `build_optimizer_fn`).
- JIT-compiles the loss+grad computation (controllable via `jit_loss_fn`).
- Runs gradient accumulation over micro-batches.
- Logs to stdout, TensorBoard, and/or W&B.
- Saves Orbax checkpoints with automatic rotation.
- Runs evaluation at configurable intervals (by steps or epochs).
- Saves `training_metrics.json` alongside checkpoints.

---

## `TrainerConfig`

Base configuration for `Trainer`. All specialized configs inherit from this. Defined as a Python `dataclass`.

```python
from zlynx.trainer import TrainerConfig
```

### JAX

| Field | Type | Default | Description |
|---|---|---|---|
| `jit_loss_fn` | `bool` | `True` | JIT-compile the loss+grad computation. Disable if your batch contains non-JAX types. |

### Batch & Accumulation

| Field | Type | Default | Description |
|---|---|---|---|
| `per_device_batch_size` | `int` | `1` | Batch size per device. |
| `gradient_accumulation_steps` | `int` | `1` | Number of micro-batches per optimization step. Effective batch = `per_device_batch_size × gradient_accumulation_steps`. |

### Dataset

| Field | Type | Default | Description |
|---|---|---|---|
| `train_subset` | `str` | `"default"` | HuggingFace dataset config name for training. |
| `eval_subset` | `str` | `"default"` | HuggingFace dataset config name for evaluation. |
| `train_split` | `str` | `"train"` | Dataset split for training. |
| `eval_split` | `str` | `"test"` | Dataset split for evaluation. |
| `streaming` | `bool` | `False` | Enable streaming mode for HuggingFace datasets. |
| `train_streaming` | `bool \| None` | `None` | Override `streaming` for train set. `None` = use `streaming`. |
| `eval_streaming` | `bool \| None` | `None` | Override `streaming` for eval set. `None` = use `streaming`. |
| `load_train_dataset_fn` | `Callable \| None` | `None` | Custom loading function for the training dataset. |
| `load_eval_dataset_fn` | `Callable \| None` | `None` | Custom loading function for the eval dataset. |

### Dataset Processing

| Field | Type | Default | Description |
|---|---|---|---|
| `processing_train_dataset_fn` | `Callable \| None` | `None` | Per-batch preprocessing function applied to each train batch before `loss_fn`. |
| `processing_eval_dataset_fn` | `Callable \| None` | `None` | Per-batch preprocessing function applied to each eval batch. |
| `seed` | `int \| None` | `None` | Random seed for dataset shuffling. `None` = `42`. |

### Grain Pipeline

| Field | Type | Default | Description |
|---|---|---|---|
| `num_workers` | `int` | `4` | Number of data loading workers. |
| `num_threads` | `int` | `16` | Number of threads for Grain `ReadOptions`. |
| `prefetch_buffer_size` | `int` | `1_000` | Grain prefetch buffer size. |

### Optimizer

| Field | Type | Default | Description |
|---|---|---|---|
| `optimizer` | `str \| Callable` | `"adamw"` | Optimizer name or callable. See [Supported Optimizers](#supported-optimizers). |
| `optimizer_kwargs` | `dict` | `{}` | Extra keyword arguments passed to the optimizer constructor. |
| `learning_rate` | `float` | `5e-5` | Peak learning rate. |
| `weight_decay` | `float` | `0.0` | Weight decay coefficient. |
| `max_grad_norm` | `float \| None` | `1.0` | Gradient clipping norm. `None` = disabled. |

### Schedule

| Field | Type | Default | Description |
|---|---|---|---|
| `lr_scheduler` | `str \| Callable` | `"cosine"` | LR schedule name or callable `(lr, total, warmup) → schedule`. See [Supported Schedulers](#supported-schedulers). |
| `warmup_steps` | `int` | `0` | Number of warmup steps. |
| `warmup_ratio` | `float` | `0.0` | Alternative: warmup as a fraction of total steps. |

### Training Duration

| Field | Type | Default | Description |
|---|---|---|---|
| `max_steps` | `int` | `-1` | Maximum optimization steps. `-1` = determined by `num_epochs` and dataset size. |
| `num_epochs` | `int` | `1` | Number of training epochs. |

### Evaluation

| Field | Type | Default | Description |
|---|---|---|---|
| `eval_strategy` | `"epochs" \| "steps"` | `"epochs"` | When to run evaluation. |
| `eval_max_steps` | `int` | `-1` | Max eval steps. `-1` = full eval dataset. Must be `> 0` for streaming eval. |
| `eval_steps` | `int \| None` | `None` | Evaluate every N steps (when `eval_strategy="steps"`). |
| `eval_epochs` | `int \| None` | `None` | Evaluate every N epochs (when `eval_strategy="epochs"`). |
| `eval_per_device_batch_size` | `int \| None` | `None` | Eval batch size. `None` = use `per_device_batch_size`. |

### Precision

| Field | Type | Default | Description |
|---|---|---|---|
| `dtype` | `str` | `"bfloat16"` | Parameter and compute dtype. |
| `grad_dtype` | `str \| None` | `None` | Gradient dtype. `None` = same as `dtype`. |
| `remat` | `bool` | `False` | Enable gradient checkpointing (recompute activations during backward). |

### Sharding & Parallelism

| Field | Type | Default | Description |
|---|---|---|---|
| `sharding` | `str \| int \| bool \| None` | `"auto"` | Sharding strategy (see below). |
| `mesh_shape` | `tuple[int, ...] \| None` | `None` | Custom device mesh shape. |

**Sharding strategies:**

| Value | Behavior |
|---|---|
| `"auto"` | DDP if model fits on one device with ≥1.5 GiB headroom, FSDP otherwise. |
| `None` | Skip — assumes user-managed sharding. |
| `False` | Single-device (first device). |
| `int` | Place on device with the given ID. |
| `"ddp"` | Replicate across all devices (Distributed Data Parallel). |
| `"fsdp"` | Shard parameters across all devices (Fully Sharded Data Parallel). |

### Checkpointing

| Field | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `str` | `"./output"` | Output directory for checkpoints and logs. |
| `save_steps` | `int` | `500` | Checkpoint every N steps. |
| `save_total_limit` | `int \| None` | `3` | Max checkpoints kept (Orbax rotation). `None` = unlimited. |
| `resume_from` | `str \| None` | `None` | Checkpoint path to resume from. |

### Logging

| Field | Type | Default | Description |
|---|---|---|---|
| `logging_steps` | `int` | `10` | Log every N steps. |
| `log_to` | `list[str]` | `[]` | Logging backends: `"wandb"`, `"tensorboard"`. |
| `run_name` | `str \| None` | `None` | Name for W&B / TensorBoard run. |
| `logging_fn` | `dict[str, Callable] \| None` | `None` | Custom metric functions. Each receives the full loss dict as `**kwargs`. |

**Example `logging_fn`:**

```python
config = TrainerConfig(
    logging_fn={
        "perplexity": lambda **kw: jnp.exp(kw["loss"]),
    }
)
```

### Advanced Hooks

#### `build_optimizer_fn`

| Field | Type | Default |
|---|---|---|
| `build_optimizer_fn` | `Callable \| None` | `None` |

Custom optimizer builder that overrides the default `build_optimizer`. Useful when a custom optimizer requires special initialization or the default builder cannot handle it.

**Signature:** `(config, total_steps) → (optimizer, schedule)`

#### `return_optimizer_extra_args_fn`

| Field | Type | Default |
|---|---|---|
| `return_optimizer_extra_args_fn` | `Callable \| None` | `None` |

Provides extra keyword arguments to `optimizer.update()`. Some optimizers require additional values such as `value`, `grad`, `value_fn`, or `grad_fn`.

**Signature:**

```python
def return_optimizer_extra_args_fn(
    value,                     # scalar loss for this update
    grad,                      # accumulated gradients
    value_fn,                  # helper: returns only the loss
    grad_fn,                   # helper: returns only the gradients
    compute_loss_and_grads_fn, # (model, loss_fn, batch) → (loss, aux, grads)
) -> dict
```

---

## Optimizer (`optim`)

### `build_optimizer`

```python
def build_optimizer(config: TrainerConfig, total_steps: int) -> tuple[optax.GradientTransformation, Schedule]
```

Builds an Optax optimizer chain from `TrainerConfig`:

1. Resolves the LR schedule via `get_scheduler`.
2. Resolves the optimizer via `get_optimizer`.
3. Chains weight decay (if nonzero) with the optimizer.
4. Wraps with `optax.with_extra_args_support`.

Returns `(optimizer, schedule)`.

### `get_optimizer`

```python
def get_optimizer(config: TrainerConfig) -> Callable
```

Resolves an optimizer by name or passthrough callable. Strips `"galore_"` prefix if present.

### `get_scheduler`

```python
def get_scheduler(config: TrainerConfig) -> Callable
```

Resolves a learning rate schedule by name or passthrough callable.

### Supported Optimizers

All Optax core and contrib optimizers are supported:

**Core:** `adabelief`, `adadelta`, `adafactor`, `adagrad`, `adam`, `adamax`, `adamaxw`, `adamw`, `adan`, `amsgrad`, `fromage`, `lamb`, `lars`, `lbfgs`, `lion`, `lookahead`, `nadam`, `nadamw`, `noisy_sgd`, `novograd`, `optimistic_adam`, `optimistic_adam_v2`, `optimistic_gradient_descent`, `polyak_sgd`, `radam`, `rmsprop`, `rprop`, `sgd`, `sign_sgd`, `sm3`, `yogi`

**Contrib:** `ademamix`, `dadapt_adamw`, `dog`, `dowg`, `dpsgd`, `madgrad`, `mechanize`, `momo`, `momo_adam`, `muon`, `prodigy`, `sam`

To use a GaLore-wrapped variant, prefix with `"galore_"` (e.g. `"galore_adamw"`).

### Supported Schedulers

| Name | Optax Function |
|---|---|
| `"constant"` | `optax.constant_schedule` |
| `"linear"` | `optax.linear_schedule` |
| `"cosine"` | `optax.cosine_decay_schedule` |
| `"cosine_onecycle"` | `optax.cosine_onecycle_schedule` |
| `"exponential"` | `optax.exponential_decay` |
| `"join"` | `optax.join_schedules` |
| `"linear_onecycle"` | `optax.linear_onecycle_schedule` |
| `"piecewise_constant"` | `optax.piecewise_constant_schedule` |
| `"piecewise_interpolate"` | `optax.piecewise_interpolate_schedule` |
| `"polynomial"` | `optax.polynomial_schedule` |
| `"sgdr"` | `optax.sgdr_schedule` |
| `"warmup_constant"` | `optax.warmup_constant_schedule` |
| `"warmup_cosine_decay"` | `optax.warmup_cosine_decay_schedule` |
| `"warmup_exponential_decay"` | `optax.warmup_exponential_decay_schedule` |

Custom scheduler kwargs can be passed via `optimizer_kwargs` in the config.

### GaLore (Gradient Low-Rank Projection)

Memory-efficient training by projecting gradients into a low-rank subspace.

#### `galore_wrapper`

```python
def galore_wrapper(
    inner_opt: optax.GradientTransformation,
    r: int = 128,
    update_proj_gap: int = 200,
    scale: float = 1.0,
) -> optax.GradientTransformation
```

Wraps an Optax optimizer with GaLore. The SVD-based projector update is handled outside of JIT via `update_galore_projectors`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `r` | `int` | `128` | Projection rank. |
| `update_proj_gap` | `int` | `200` | Steps between projector updates. |
| `scale` | `float` | `1.0` | Gradient scaling factor after projection. |

#### `update_galore_projectors`

```python
def update_galore_projectors(galore_state, grads, params, r=128) -> GaloreState
```

Performs SVD-based projector update. Must be called from Python (outside JIT) every `update_proj_gap` steps.

#### `find_galore_state` / `set_galore_state`

```python
def find_galore_state(opt_state) -> tuple[index | None, GaloreState | None]
def set_galore_state(opt_state, idx, new_gstate) -> opt_state
```

Utility functions for locating and replacing `GaloreState` inside an Optax chain.

---

## Data Pipeline

### `grain_from_source`

```python
def grain_from_source(
    data: Any,
    *,
    adapter: Callable | None = None,
    preprocess_fn: Callable | None = None,
    dict_mode: str = "items",
    read_opts: grain.ReadOptions | None = None,
    batch_size: int | None = None,
    seed: int = 42,
) -> grain.IterDataset
```

Converts common Python/HF data containers into a Grain pipeline.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `Any` | — | Input data: mapping, random-access dataset, iterable, generator, or HF dataset. |
| `adapter` | `Callable \| None` | `None` | Per-sample transform applied at access time. |
| `preprocess_fn` | `Callable \| None` | `None` | Preprocessing function. |
| `dict_mode` | `str` | `"items"` | How to convert mappings: `"items"`, `"values"`, or `"keys"`. |
| `read_opts` | `ReadOptions \| None` | `None` | Grain read options (threading, prefetch). |
| `batch_size` | `int \| None` | `None` | Batch size. `None` = unbatched. |
| `seed` | `int` | `42` | Shuffle seed (map-style only). |

**Behavior:**
- Mapping / random-access data → `MapDataset` (shuffled, converted to `IterDataset`).
- Iterable-only data → `IterDataset` (no shuffle).

### `GeneralRandomAccessSource`

```python
class GeneralRandomAccessSource(grain.sources.RandomAccessDataSource)
```

Wraps random-access Python objects (lists, HF datasets, `__len__` + `__getitem__` objects) as a Grain source.

| Parameter | Type | Description |
|---|---|---|
| `data` | `Any` | Random-access data source. |
| `adapter` | `Callable \| None` | Optional per-sample transform. |
| `dict_mode` | `str` | For mapping inputs: `"items"` (default), `"values"`, `"keys"`. |

### `GeneralIterDataset`

```python
class GeneralIterDataset(grain.IterDataset)
```

Wraps iterable-only data (HF `IterableDataset`, generators, iterators) as a Grain `IterDataset` with checkpointing support.

| Parameter | Type | Description |
|---|---|---|
| `data` | `Iterable \| Callable` | Iterable data or factory function returning an iterable. |
| `adapter` | `Callable \| None` | Optional per-sample transform. |

### `process_dataset`

```python
def process_dataset(
    dataset, config: TrainerConfig, is_eval=False
) -> tuple[grain.IterDataset, int | None]
```

Resolves a dataset source, builds Grain pipeline with read options and batching. Returns `(pipeline, num_examples)`. Used internally by trainers.

### `load_source`

```python
def load_source(dataset, config: TrainerConfig | None, is_eval=False) -> dataset
```

Resolves a dataset from multiple input types:

1. **String** → tries `load_dataset_fn`, then `load_from_disk`, then `load_dataset`.
2. **Dict** → converts to `datasets.Dataset.from_dict`.
3. **Other** → returns as-is.

---

## Model Processing

### `process_model`

```python
def process_model(model, config: TrainerConfig) -> tuple[model, config]
```

Applies device placement and sharding based on `config.sharding`. See [Sharding strategies](#sharding--parallelism) for details.

### `apply_remat`

```python
def apply_remat(model, *, skip_root=True) -> model
```

Wraps `__call__` of all submodules with `nnx.remat` for gradient checkpointing. Reduces peak memory at the cost of recomputation during the backward pass.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nnx.Module` | — | Model to modify in-place. |
| `skip_root` | `bool` | `True` | Whether to skip wrapping the root module. |

---

## Experimental

> [!WARNING]
> The following APIs are **experimental** and may change. Use `Trainer` directly if you encounter any problems.

### `SFTTrainer`

Supervised Fine-Tuning trainer. Extends `Trainer` to automatically tokenize, pad, and truncate text datasets.

```python
class SFTTrainer(
    model,
    dataset,
    processor,
    trconfig: Optional[SFTConfig] = None,
    dsconfig: Optional[DatasetConfig] = None,
    loss_fn: Optional[Callable] = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nnx.Module` | — | The model to fine-tune. |
| `dataset` | `Any` | — | Training dataset. |
| `processor` | `PreTrainedTokenizer` | — | HuggingFace tokenizer. **Required.** |
| `trconfig` | `SFTConfig \| None` | `SFTConfig()` | SFT-specific configuration. |
| `dsconfig` | `DatasetConfig \| None` | `DatasetConfig()` | Dataset configuration. |
| `loss_fn` | `Callable \| None` | `causal_lm_loss` | Custom loss function override. |

#### `SFTConfig`

Extends `TrainerConfig`.

| Field | Type | Default | Description |
|---|---|---|---|
| `max_seq_len` | `int` | `1024` | Maximum sequence length for tokenization. |
| `dataset_text_field` | `str` | `"text"` | Column name containing the text to fine-tune on. |
| `formatting_func` | `Callable \| None` | `None` | Optional `(example) → str` to format each sample. |

---

### `GRPOTrainer`

Group Relative Policy Optimization trainer with generation rollouts, reward computation, and PPO-style clipped policy gradient.

```python
class GRPOTrainer(
    model,
    ref_model,
    reward_funcs: list[Callable],
    dataset,
    processor=None,
    trconfig: Optional[GRPOConfig] = None,
    dsconfig=None,
    loss_fn: Optional[Callable] = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nnx.Module` | — | The policy model to train. |
| `ref_model` | `nnx.Module` | — | Frozen reference model for KL penalty. |
| `reward_funcs` | `list[Callable]` | — | Reward functions `f(completion_str) → float`. |
| `dataset` | `Any` | — | Dataset of prompts. |
| `processor` | Tokenizer | `None` | Tokenizer for decoding generations. |
| `trconfig` | `GRPOConfig \| None` | `GRPOConfig()` | GRPO-specific configuration. |
| `loss_fn` | `Callable \| None` | `grpo_loss_fn` | Custom loss function override. |

#### `GRPOConfig`

Extends `TrainerConfig`.

| Field | Type | Default | Description |
|---|---|---|---|
| `max_seq_len` | `int` | `1024` | Maximum total sequence length. |
| `num_generations` | `int` | `8` | Group size `G` — completions per prompt. |
| `max_prompt_length` | `int` | `512` | Maximum prompt token length. |
| `max_completion_length` | `int` | `512` | Maximum completion token length. |
| `beta` | `float` | `0.0` | KL divergence penalty coefficient. |
| `clip_eps` | `float` | `0.2` | PPO clipping ratio. |
| `entropy_coeff` | `float` | `0.0` | Entropy bonus coefficient. |
| `mu_epochs` | `int` | `1` | Inner optimization epochs per generation. |
| `dataset_prompt_field` | `str` | `"prompt"` | Column name for prompts. |
| `dataset_responses_field` | `str` | `"responses"` | Column name for responses. |

---

### `DPOTrainer`

Direct Preference Optimization trainer with chosen/rejected pair formatting and multiple loss variants.

```python
class DPOTrainer(
    model,
    ref_model,
    dataset,
    processor=None,
    trconfig: Optional[DPOConfig] = None,
    dsconfig=None,
    loss_fn: Optional[Callable] = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nnx.Module` | — | The policy model to train. |
| `ref_model` | `nnx.Module` | — | Frozen reference model for KL divergence. |
| `dataset` | `Any` | — | Dataset with prompt/chosen/rejected fields. |
| `processor` | Tokenizer | `None` | Tokenizer for automatic pair preprocessing. |
| `trconfig` | `DPOConfig \| None` | `DPOConfig()` | DPO-specific configuration. |
| `loss_fn` | `Callable \| None` | `dpo_loss_fn` | Custom loss function override. |

#### `DPOConfig`

Extends `TrainerConfig`.

| Field | Type | Default | Description |
|---|---|---|---|
| `beta` | `float` | `0.1` | DPO temperature / KL penalty. |
| `max_prompt_length` | `int` | `512` | Max prompt length. |
| `max_seq_len` | `int` | `1024` | Max total sequence length. |
| `dataset_prompt_field` | `str` | `"prompt"` | Column name for prompts. |
| `dataset_chosen_field` | `str` | `"chosen"` | Column name for chosen responses. |
| `dataset_rejected_field` | `str` | `"rejected"` | Column name for rejected responses. |
| `loss_type` | `str` | `"sigmoid"` | Loss variant: `"sigmoid"`, `"hinge"`, `"ipo"`, or `"kto_pair"`. |
| `label_smoothing` | `float` | `0.0` | Label smoothing factor. |

---

### `DSFTTrainer`

Diffusion Supervised Fine-Tuning trainer for image generation models.

```python
class DSFTTrainer(
    model,
    dataset,
    noise_scheduler,
    processor=None,
    trconfig: Optional[DSFTConfig] = None,
    dsconfig: Optional[DatasetConfig] = None,
    loss_fn: Optional[Callable] = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `nnx.Module` | — | The diffusion model. |
| `dataset` | `Any` | — | Image dataset. |
| `noise_scheduler` | `Any` | — | Noise scheduler for timestep/noise generation. |
| `processor` | Tokenizer | `None` | Optional text tokenizer for conditional generation. |
| `trconfig` | `DSFTConfig \| None` | `DSFTConfig()` | DSFT-specific configuration. |
| `loss_fn` | `Callable \| None` | `diffusion_loss` | Custom loss function override. |

#### `DSFTConfig`

Extends `TrainerConfig`.

| Field | Type | Default | Description |
|---|---|---|---|
| `image_size` | `tuple[int, int]` | `(256, 256)` | Target image resolution. |
| `dataset_image_field` | `str` | `"image"` | Column name for images. |
| `dataset_text_field` | `str` | `"text"` | Column name for conditioning text. |
| `formatting_func` | `Callable \| None` | `None` | Optional `(example) → dict` to extract `"image"` / `"text"`. |
| `num_train_timesteps` | `int` | `1000` | Total diffusion timesteps. |
| `prediction_type` | `str` | `"epsilon"` | Prediction target: `"epsilon"`, `"v_prediction"`, or `"sample"`. |

---

### Experimental Loss Functions

#### `causal_lm_loss`

```python
def causal_lm_loss(model, batch) -> scalar
```

Standard autoregressive causal LM loss. Calls `model(input_ids, attention_mask=..., labels=...)` and returns `outputs.loss`.

#### `compute_loss_and_grads`

```python
def compute_loss_and_grads(model, loss_fn, batch) -> tuple[loss, aux, grads]
```

Computes the loss, auxiliary outputs, and gradients for a single micro-batch. Handles both scalar and dict returns from `loss_fn`.

#### `grpo_loss_fn`

```python
def grpo_loss_fn(model, batch) -> scalar
```

Standard GRPO loss combining clipped policy gradient, KL penalty, and entropy bonus.

#### `dpo_loss_fn`

```python
def dpo_loss_fn(model, batch) -> scalar
```

DPO loss with support for sigmoid, hinge, IPO, and KTO-pair variants.

#### `diffusion_loss`

```python
def diffusion_loss(model, batch) -> scalar
```

Standard MSE diffusion loss: `mean((noise_pred - noise)²)`.

#### `token_log_probs`

```python
def token_log_probs(logits, labels) -> jnp.ndarray  # (B, S-1)
```

Per-token log-probabilities `log P(label | context)`. Used internally by GRPO and DPO losses.
