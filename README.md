# Zlynx

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zlynx-ai/zlynx)

A lightweight, highly-customizable deep learning library built on **JAX** and **Flax NNX**. Designed for researchers and developers who want fine-grained control over model architectures, training loops, and distributed setups without the bloat of massive frameworks.

## Install

```bash
uv pip install zlynx
```

## Define & Load Models

```python
from zlynx import Z

class MyModel(Z): ...

# Load from HuggingFace
model = MyModel.load_hf("username/my-model", format="safetensors")

# Load from Kaggle
model = MyModel.load_kaggle("username/my-model", sharding="fsdp")

# Load from local checkpoint
model = MyModel.load("./checkpoint", key=jax.random.key(0))
```

## Built-in Llama

```python
from zlynx.models.llama import LlamaConfig, LlamaLanguageModel

config = LlamaConfig(vocab_size=32000, hidden_size=512, num_hidden_layers=2)
model = LlamaLanguageModel(config)

# Generate
output_ids = model.generate(input_ids, key=jax.random.key(0), max_new_tokens=128)
```

## Train

```python
from zlynx.trainer import Trainer, TrainerConfig

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    train_dataset=dataset,
    config=TrainerConfig(
        batch_size=32,
        learning_rate=5e-5,
        num_epochs=3,
        sharding="auto",
    ),
)
trainer.train()
```

## PEFT (LoRA, DoRA, VeRA, LoHa, LoKr, AdaLoRA)

```python
from zlynx.modules.peft import apply_peft

model = apply_peft(model, method="lora", r=16, alpha=32, target_modules=["q_proj", "v_proj"])
```

## Save & Push

```python
model.save("./my-model", format="safetensors")
model.push_hf("username/my-model")
model.push_kaggle("username/my-model")
```

## Features

- **Checkpointing** — Orbax + SafeTensors, HuggingFace Hub & Kaggle integration
- **Training** — gradient accumulation, LR scheduling, multi-backend logging (W&B, TensorBoard)
- **Sharding** — auto, DDP, FSDP with one config change
- **PEFT** — 6 adapter methods via `apply_peft()`
- **GaLore** — gradient low-rank projection for memory-efficient full fine-tuning
- **Data** — Grain-based pipeline accepting lists, HF datasets, dicts, and iterables
- **Modules** — Attention (GQA/MQA), MLP (SwiGLU), RMSNorm, RoPE, KV Cache, DiT blocks

## Documentation

- [Getting Started](docs/tutorials/getting_started/01_installation.md)
- [MNIST Tutorial](docs/tutorials/mnist.md)
- [API Reference — Trainer](docs/api-references/trainer.md)
- [API Reference — Modules](docs/api-references/modules.md)
- [API Reference — Models](docs/api-references/models.md)
