# Push to Hub

How to share your trained models on HuggingFace Hub and Kaggle Models.

---

## Overview

Zlynx makes it easy to push your trained models directly to:
- **HuggingFace Hub** — share with the open-source community
- **Kaggle Models** — share on Kaggle

---

## Push to HuggingFace

### Prerequisites

```python
from huggingface_hub import login
login()  # Add your token
```

Or set via environment variable:
```bash
export HF_TOKEN=your_token_here
```

### Push your model

```python
model.push_hf("username/my-model", private=False)
```

### Private model

```python
model.push_hf("username/my-private-model", private=True)
```

---

## Format Options

### Safetensors (default)

Safetensors format is HuggingFace-compatible and recommended:

```python
model.push_hf("username/my-model", private=False, format="safetensors")
```

### Orbax format

```python
model.push_hf("username/my-model", private=False, format="orbax")
```

### Both formats

```python
model.push_hf("username/my-model", private=False, format="all")
```

---

## Safetensors Options

### Max shard size

Control the max size per safetensors file (default: 3GB):

```python
model.push_hf("username/my-model", private=False, max_shard_size_gb=1)
```

---

## Push to Kaggle

### Prerequisites

```python
import kagglehub
kagglehub.login()
```

### Push your model

```python
model.push_kaggle("username/my-model")
```

This uploads to `username/my-model` under the "flax" framework.

### With variation

```python
model.push_kaggle("username/my-model", variation="v2")
```

Useful for versioning: `variation="v1"`, `variation="v2"`, etc.

### Format options

```python
# Push as safetensors (default)
model.push_kaggle("username/my-model", format="safetensors")

# Push as orbax
model.push_kaggle("username/my-model", format="orbax")

# Push both formats
model.push_kaggle("username/my-model", format="all")
```

---

## Load from Hub

### Load from HuggingFace

```python
from zlynx import Z

class MyModel(Z): ...

model = MyModel.load_hf("username/my-model")
```

### Load from Kaggle

```python
import kagglehub
kagglehub.login()

model = MyModel.load_kaggle("username/my-model")

# With specific variation
model = MyModel.load_kaggle("username/my-model", variation="v2")
```

---

## Full Example: Train & Push

```python
from zlynx import Z
from zlynx.trainer import Trainer, TrainerConfig

# 1. Define your model
class MyModel(Z): ...

# 2. Load pretrained weights
model = MyModel.load_hf("base-model/weights")

# 3. Fine-tune
trainer = Trainer(model, loss_fn, train_dataset, config=TrainerConfig(num_epochs=3))
trainer.train()

# 4. Push to HuggingFace
model.push_hf("username/my-finetuned-model", private=False)

# 5. Also push to Kaggle
model.push_kaggle("username/my-finetuned-model")
```

---

## Next Steps

- **[Sharding & Multi-Device Training](./05_sharding.md)** — scale your training
- **[PEFT Adapters](../advanced/01_peft.md)** — add LoRA, DoRA adapters
- **[GaLore Optimizer](../advanced/02_galore.md)** — memory-efficient training
