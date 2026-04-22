"""Smoke test for Trainer with checkpointing and multi-device sharding."""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import optax

from zlynx.model.llama import LlamaConfig, LlamaLanguageModel
from zlynx.trainer.trainer import Trainer, TrainerConfig
from zlynx.trainer.dataset import DatasetConfig


# ── tiny model ──
config = LlamaConfig(
    vocab_size=16,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    head_dim=8,
)

# ── dummy dataset ──
data = [
    {"input_ids": jnp.array([1, 2, 3, 4, 5]), "labels": jnp.array([2, 3, 4, 5, 6])}
    for _ in range(32)
]

def cross_entropy_loss(model, batch):
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    logits = model(input_ids)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean()

dsconfig = DatasetConfig(shuffle=False)

def test_checkpointing():
    print(f"\n{'='*50}")
    print("Test 1: Checkpointing with save_steps=10")
    print(f"{'='*50}")

    model = LlamaLanguageModel(config)
    trconfig = TrainerConfig(
        batch_size=4, gradient_accumulation_steps=2,
        learning_rate=1e-3, lr_scheduler="constant",
        num_epochs=10, logging_steps=5, save_steps=10,
        save_total_limit=2, output_dir="./tmp_test_ckpt",
        log_to=["stdout"],
        sharding="auto", # explicitly use auto
    )
    trainer = Trainer(model=model, dataset=data, loss_fn=cross_entropy_loss,
                      trconfig=trconfig, dsconfig=dsconfig)
    trainer.train()

    # check checkpoints
    from pathlib import Path
    ckpt_dir = Path("./tmp_test_ckpt/checkpoints")
    if ckpt_dir.exists():
        ckpts = [p.name for p in sorted(ckpt_dir.iterdir()) if p.is_dir()]
        print(f"Checkpoints found: {ckpts}")
        print(f"max_to_keep=2, so should have at most 2 checkpoint dirs")

def test_custom_sharding():
    print(f"\n{'='*50}")
    print("Test 1b: Custom sharding (sharding=None)")
    print(f"{'='*50}")

    model_custom = LlamaLanguageModel(config)
    trconfig_custom = TrainerConfig(
        batch_size=4, learning_rate=1e-3, lr_scheduler="constant",
        max_steps=5, logging_steps=5, save_steps=0,
        sharding=None, output_dir="./tmp_test_custom",
        log_to=["stdout"],
    )
    trainer_custom = Trainer(model=model_custom, dataset=data, loss_fn=cross_entropy_loss,
                             trconfig=trconfig_custom, dsconfig=dsconfig)
    trainer_custom.train()
    print("Custom sharding (skip) ✅")

def test_dp_sharding():
    print(f"\n{'='*50}")
    print(f"Test 2: DP sharding ({len(jax.devices())} devices)")
    print(f"{'='*50}")

    model_dp = LlamaLanguageModel(config)
    trconfig_dp = TrainerConfig(
        batch_size=4, learning_rate=1e-3, lr_scheduler="constant",
        max_steps=5, logging_steps=5, save_steps=0,
        sharding="dp", output_dir="./tmp_test_dp",
        log_to=["stdout"],
    )
    trainer_dp = Trainer(model=model_dp, dataset=data, loss_fn=cross_entropy_loss,
                         trconfig=trconfig_dp, dsconfig=dsconfig)
    trainer_dp.train()
    print("DP sharding ✅")

def test_fsdp_sharding():
    print(f"\n{'='*50}")
    print(f"Test 3: FSDP sharding ({len(jax.devices())} devices)")
    print(f"{'='*50}")

    model_fsdp = LlamaLanguageModel(config)
    trconfig_fsdp = TrainerConfig(
        batch_size=4, learning_rate=1e-3, lr_scheduler="constant",
        max_steps=5, logging_steps=5, save_steps=0,
        sharding="fsdp", output_dir="./tmp_test_fsdp",
        log_to=["stdout"],
    )
    trainer_fsdp = Trainer(model=model_fsdp, dataset=data, loss_fn=cross_entropy_loss,
                           trconfig=trconfig_fsdp, dsconfig=dsconfig)
    trainer_fsdp.train()
    print("FSDP sharding ✅")

def test_explicit_device_id():
    print(f"\n{'='*50}")
    print(f"Test 4: Explicit device ID (device=0)")
    print(f"{'='*50}")

    model_dev = LlamaLanguageModel(config)
    trconfig_dev = TrainerConfig(
        batch_size=4, learning_rate=1e-3, lr_scheduler="constant",
        max_steps=5, logging_steps=5, save_steps=0,
        sharding=0, output_dir="./tmp_test_dev",
        log_to=["stdout"],
    )
    trainer_dev = Trainer(model=model_dev, dataset=data, loss_fn=cross_entropy_loss,
                          trconfig=trconfig_dev, dsconfig=dsconfig)
    trainer_dev.train()
    print("Device ID placement ✅")

def test_tp_sharding():
    print(f"\n{'='*50}")
    print(f"Test 5: TP sharding ({len(jax.devices())} devices)")
    print(f"{'='*50}")

    model_tp = LlamaLanguageModel(config)
    trconfig_tp = TrainerConfig(
        batch_size=4, learning_rate=1e-3, lr_scheduler="constant",
        max_steps=5, logging_steps=5, save_steps=0,
        sharding="tp", output_dir="./tmp_test_tp",
        log_to=["stdout"],
    )
    trainer_tp = Trainer(model=model_tp, dataset=data, loss_fn=cross_entropy_loss,
                         trconfig=trconfig_tp, dsconfig=dsconfig)
    trainer_tp.train()
    print("TP sharding ✅")

if __name__ == "__main__":
    test_checkpointing()
    test_custom_sharding()
    test_dp_sharding()
    test_fsdp_sharding()
    test_explicit_device_id()
    test_tp_sharding()

    print(f"\n{'='*50}")
    print("All tests passed! 🎉")
    print(f"{'='*50}")
