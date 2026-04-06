# Parameter-Efficient Fine-Tuning (PEFT)

Fine-tune large models by training only a tiny fraction of parameters. Zlynx implements 6 PEFT methods that seamlessly replace `nnx.Linear` layers with lightweight adapters.

---

## Quick Start

```python
from zlynx.modules.peft import apply_peft

# Apply LoRA to attention layers — modifies the model in-place
model = apply_peft(
    model,
    method="lora",
    r=16,
    alpha=32,
    target_modules=["q_proj", "v_proj"],
)
```

That's it. The `q_proj` and `v_proj` layers are now `LoraLinear` wrappers. The base weights are frozen; only the low-rank adapter parameters are trained.

---

## How `apply_peft` Works

`apply_peft` walks the model tree recursively. For each `nnx.Linear` layer whose name matches a string in `target_modules`, it:

1. **Freezes** the original weight by converting `nnx.Param` → `nnx.Variable` (removed from optimizer state)
2. **Wraps** the layer with the chosen adapter class
3. **Adds** small trainable `nnx.Param` matrices

```python
apply_peft(
    model,                                 # any nnx.Module
    method="lora",                         # adapter type (see below)
    r=8,                                   # rank
    alpha=16,                              # scaling factor (scaling = alpha / r)
    target_modules=["q_proj", "v_proj"],   # which layers to adapt
    rngs=nnx.Rngs(42),                     # optional, for initializing adapter params
)
```

The `target_modules` uses substring matching — `"q_proj"` matches any attribute name containing `"q_proj"`.

---

## Available Methods

### LoRA — Low-Rank Adaptation

The standard approach. Adds a low-rank update `A @ B` to the frozen weight.

```
output = (x @ W_frozen) + (x @ A @ B) × (α/r)
```

- **A**: `(in, r)` — initialized with random normal
- **B**: `(r, out)` — initialized to zero (so the adapter starts as identity)

```python
model = apply_peft(model, method="lora", r=16, alpha=32, target_modules=["q_proj", "v_proj"])
```

**Trainable params**: `2 × r × dim` per adapted layer

---

### DoRA — Weight-Decomposed Low-Rank Adaptation

Decomposes the weight into **magnitude** and **direction**. The direction is updated via LoRA, while magnitude is a separate learnable vector. This mimics full fine-tuning more closely than standard LoRA.

```
W' = W_frozen + A @ B × (α/r)
direction = W' / ‖W'‖_columns
output = x @ (m × direction)
```

- **m**: `(1, out)` — magnitude vector, initialized to column norms of the original weight
- **A, B**: same as LoRA

```python
model = apply_peft(model, method="dora", r=16, alpha=32, target_modules=["q_proj", "v_proj"])
```

**Trainable params**: `2 × r × dim + out_features` per layer

---

### VeRA — Vector-based Random Adaptation

Extreme parameter reduction. The A and B matrices are **randomly initialized and frozen** — only tiny scaling vectors `d` and `b` are trained.

```
output = (x @ W_frozen) + (x @ A_frozen × diag(d) @ B_frozen × b) × (α/r)
```

- **A**: `(in, r)` — frozen random
- **B**: `(r, out)` — frozen random
- **d**: `(r,)` — trainable scaling vector
- **b**: `(out,)` — trainable scaling vector

```python
model = apply_peft(model, method="vera", r=16, alpha=32, target_modules=["q_proj", "v_proj"])
```

**Trainable params**: `r + out_features` per layer — dramatically fewer than LoRA

---

### LoHa — Hadamard Product Adaptation

Uses the element-wise (Hadamard) product of two low-rank paths for high expressiveness:

```
ΔW = (A1 @ B1) ⊙ (A2 @ B2)
output = (x @ W_frozen) + (x @ ΔW) × (α/r)
```

- **A1, A2**: `(in, r)` — random normal
- **B1**: `(r, out)` — initialized to zero
- **B2**: `(r, out)` — random normal

```python
model = apply_peft(model, method="loha", r=16, alpha=32, target_modules=["q_proj", "v_proj"])
```

**Trainable params**: `4 × r × dim` per layer (2× LoRA, but much more expressive)

---

### LoKr — Kronecker Product Adaptation

Uses the Kronecker product of a learnable low-rank matrix and a random matrix:

```
ΔW = kron(A @ B, O)[:in, :out]
output = (x @ W_frozen) + (x @ ΔW) × (α/r)
```

- **A, B**: `(r, r)` — learnable low-rank pair
- **O**: `(in//r, out//r)` — learnable parameter matrix

```python
model = apply_peft(model, method="lokr", r=16, alpha=32, target_modules=["q_proj", "v_proj"])
```

---

### AdaLoRA — Adaptive Low-Rank Adaptation

Explicitly parameterizes the update as an SVD: `P @ diag(E) @ Q`. The singular values `E` can be dynamically pruned during training for adaptive rank.

```
output = (x @ W_frozen) + (x @ P @ diag(E) @ Q) × (α/r)
```

- **P**: `(in, r)` — left singular vectors
- **E**: `(r,)` — singular values (initialized to zero)
- **Q**: `(r, out)` — right singular vectors

```python
model = apply_peft(model, method="adalora", r=16, alpha=32, target_modules=["q_proj", "v_proj"])
```

---

## Choosing a Method

| Method      | Trainable Params | Expressiveness | Best for                                        |
| ----------- | ---------------- | -------------- | ----------------------------------------------- |
| **LoRA**    | Medium           | Good           | General-purpose, proven baseline                |
| **DoRA**    | Medium+          | Very good      | When you need closest match to full fine-tuning |
| **VeRA**    | Very low         | Moderate       | Extreme parameter budget constraints            |
| **LoHa**    | High             | Very high      | Tasks requiring high expressiveness             |
| **LoKr**    | Medium           | Good           | Structured weight updates                       |
| **AdaLoRA** | Medium           | Good           | When optimal rank is unknown (adaptive pruning) |

**Default recommendation:** Start with **LoRA** (`r=16`, `alpha=32`). If quality isn't sufficient, try **DoRA**.

---

## Choosing `target_modules`

For transformer models, common targets are:

```python
# Attention only (most common)
target_modules = ["q_proj", "v_proj"]

# Attention + output
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Attention + MLP (most parameters adapted)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

More target modules = more trainable parameters = better quality but slower training and more memory.

---

## Choosing Rank (`r`) and Alpha (`α`)

- **Rank `r`**: Controls the size of the low-rank matrices. Higher = more capacity but more parameters.
  - Common values: `4`, `8`, `16`, `32`, `64`
  - Start with `16` for most tasks

- **Alpha `α`**: Scaling factor. The adapter output is scaled by `α / r`.
  - Rule of thumb: set `alpha = 2 × r` (e.g. `r=16, alpha=32`)
  - Higher alpha = stronger adapter influence

---

## Full Example

```python
from zlynx.models.llama import LlamaConfig, LlamaLanguageModel
from zlynx.modules.peft import apply_peft
from zlynx.trainer import Trainer, TrainerConfig
from flax import nnx

# Load a model
config = LlamaConfig(vocab_size=32000, hidden_size=2048, num_hidden_layers=16, head_dim=64)
model = LlamaLanguageModel(config)

# Apply LoRA
model = apply_peft(
    model,
    method="lora",
    r=16,
    alpha=32,
    target_modules=["q_proj", "v_proj"],
    rngs=nnx.Rngs(42),
)

# Train as usual — only adapter params are updated
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    train_dataset="your_dataset",
    config=TrainerConfig(
        learning_rate=2e-4,
        num_epochs=3,
        processing_train_dataset_fn=preprocess,
    ),
)
trainer.train()

# Save — includes both frozen base weights and trained adapters
model.save("./peft_checkpoint")
```

---

## Next Steps

Learn about the [GaLore Optimizer](./02_galore.md) for memory-efficient full-parameter training.
