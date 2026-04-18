# Zlynx API Reference

Complete API documentation for the Zlynx framework.

---

## Trainer

Training loop, configuration, optimizer utilities, loss functions, data pipeline, and logging.

- [Trainer API Reference](./api-references/trainer.md)

### Quick Example

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

**Key classes:** `Trainer`, `TrainerConfig`, `SFTConfig`, `GRPOConfig`, `DPOConfig`, `DSFTConfig`

---

## Modules

Reusable building-block layers for constructing models in Flax NNX.

- [Modules API Reference](./api-references/modules.md)

**Key classes:** `Attention`, `KVCache`, `MLP`, `RMSNorm`, `RotaryEmbedding`, `AdaLayerNormZero`, `TimestepEmbedder`, `PatchEmbed`

**PEFT adapters:** `LoraLinear`, `DoraLinear`, `VeraLinear`, `LohaLinear`, `LokrLinear`, `AdaloraLinear`, `apply_peft`

---

## Models

Model architectures, base class for checkpointing, configuration structs, and inference utilities.

- [Models API Reference](./api-references/models.md)

### Quick Example

```python
from zlynx import Z

class MyModel(Z): ...

# Save & load
model.save("./checkpoint", format="safetensors")
model = MyModel.load("./checkpoint")

# HuggingFace Hub
model = MyModel.load_hf("username/my-model", sharding="fsdp")
model.push_hf("username/my-model")

# Kaggle
model = MyModel.load_kaggle("username/my-model")
model.push_kaggle("username/my-model")
```

**Key classes:** `Z`, `LanguageModel`, `Config`, `LanguageConfig`, `ModelOutput`, `CausalLMOutput`

**Architectures:** `LlamaLanguageModel`, `DiT`
