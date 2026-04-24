"""
Memory profiling script for TPU v5e-8 with FSDP + GaLore.
Run on Kaggle with TPU v5e-8 accelerator.

Usage: python profile_tpu_memory.py
"""

import time
import jax
import jax.numpy as jnp
from flax import nnx

# Print device info
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
for i, d in enumerate(jax.devices()):
    stats = d.memory_stats()
    if stats is not None:
        total = stats.get("bytes_limit", 0) / 1e9
        print(f"  Device {i}: {total:.2f} GB HBM")
    else:
        print(f"  Device {i}: memory_stats unavailable (CPU?)")

print("\n" + "="*60)


def get_memory_per_device():
    stats = jax.devices()[0].memory_stats()
    if stats is None:
        return 0, 0
    used = stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0))
    total = stats.get("bytes_limit", 0)
    return used, total


def report(label):
    used, total = get_memory_per_device()
    pct = used / total * 100 if total > 0 else 0
    print(f"[{label}] Device 0: {used/1e9:.3f} GB / {total/1e9:.2f} GB ({pct:.1f}%)")
    return used


# ── Config ──
from zlynx.model.meta.llama import LlamaConfig, LlamaLanguageModel

config = LlamaConfig(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=22,
    attention_head=32,
    kv_head=8,
    head_dim=64,
    dtype="bfloat16",
    param_dtype="float32",
)

BATCH_SIZE = 2
SEQ_LEN = 512

print(f"\nModel config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, intermediate={config.intermediate_size}")
print(f"Batch: {BATCH_SIZE}, SeqLen: {SEQ_LEN}")
print("="*60)

# ── Step 0: Baseline ──
baseline = report("0. Baseline")

# ── Step 1: Create model ──
model = LlamaLanguageModel(config, key=jax.random.key(0))

from zlynx.trainer.process import count_params, param_bytes
n_params = count_params(model)
print(f"\nTotal params: {n_params:,} ({n_params/1e9:.3f}B)")
print(f"Param memory (f32): {param_bytes(model, jnp.float32)/1e9:.3f} GB")
print(f"Param memory (bf16): {param_bytes(model, jnp.bfloat16)/1e9:.3f} GB")

after_model = report("1. After model creation")

# ── Step 2: FSDP ──
from zlynx.trainer.process import process_model
from zlynx.trainer.trainer_config import TrainerConfig

trconfig = TrainerConfig(
    batch_size=BATCH_SIZE,
    optimizer="galore_adamw",
    optimizer_kwargs={"galore_r": 128, "galore_update_proj_gap": 200},
    sharding="fsdp",
    max_grad_norm=1.0,
)

model = process_model(model, trconfig)
after_fsdp = report("2. After FSDP sharding")

# ── Step 3: GaLore optimizer ──
from zlynx.trainer.optim import build_optimizer, find_galore_state, set_galore_state, update_galore_projectors

opt = build_optimizer(trconfig, total_steps=1000)
optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)
after_opt = report("3. After GaLore optimizer init")

# ── Step 4: Dummy batch ──
dummy_batch = {
    "input_ids": jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
    "attention_mask": jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32),
}
after_batch = report("4. After batch allocation")

# ── Step 5: Forward+Backward ──
from zlynx.trainer.loss_fn import causal_lm_loss, compute_loss_and_grads

print("\nCompiling forward+backward...")
t0 = time.time()
loss, aux, grads = compute_loss_and_grads(model, causal_lm_loss, dummy_batch)
jax.block_until_ready(loss)
print(f"  Compiled in {time.time()-t0:.1f}s")
after_fwd_bwd = report("5. After forward+backward")

# ── Step 6: GaLore projector update (SVD outside JIT) ──
print("\nRunning GaLore projector update (SVD outside JIT)...")
t0 = time.time()
galore_idx, cur_gstate = find_galore_state(optimizer.opt_state)
params = nnx.state(model, nnx.Param)
new_gstate = update_galore_projectors(cur_gstate, grads, params, r=128)
optimizer.opt_state = set_galore_state(optimizer.opt_state, galore_idx, new_gstate)
jax.block_until_ready(jax.tree_util.tree_leaves(new_gstate.projector))
print(f"  SVD projector update in {time.time()-t0:.1f}s")
after_proj = report("6. After GaLore projector update")

# ── Step 7: Optimizer update (no SVD, just project → adam → project back) ──
print("\nCompiling optimizer.update (GaLore, no SVD)...")
t0 = time.time()
optimizer.update(model, grads)
print(f"  Compiled in {time.time()-t0:.1f}s")
after_update = report("7. After optimizer.update()")

del grads, loss, aux
after_cleanup = report("8. Steady state (grads freed)")

# ── Summary ──
print("\n" + "="*60)
print("MEMORY DELTA SUMMARY — GaLore (Device 0)")
print("="*60)
print(f"  Baseline:            {baseline/1e9:.3f} GB")
print(f"  + Model:             {(after_model - baseline)/1e9:.3f} GB")
print(f"  + FSDP:              {(after_fsdp - after_model)/1e9:.3f} GB")
print(f"  + GaLore init:       {(after_opt - after_fsdp)/1e9:.3f} GB")
print(f"  + Fwd+Bwd:           {(after_fwd_bwd - after_batch)/1e9:.3f} GB")
print(f"  + Projector SVD:     {(after_proj - after_fwd_bwd)/1e9:.3f} GB")
print(f"  + Optimizer step:    {(after_update - after_proj)/1e9:.3f} GB")
print(f"  Peak (step 7):       {after_update/1e9:.3f} GB")
print(f"  Steady (step 8):     {after_cleanup/1e9:.3f} GB")

# ── Batch size sweep with GaLore ──
print("\n" + "="*60)
print("BATCH SIZE SWEEP (GaLore + FSDP)")
print("="*60)
for bs in [1, 2, 4, 8]:
    try:
        batch = {
            "input_ids": jnp.ones((bs, SEQ_LEN), dtype=jnp.int32),
            "attention_mask": jnp.ones((bs, SEQ_LEN), dtype=jnp.int32),
        }
        t0 = time.time()
        loss, _, grads = compute_loss_and_grads(model, causal_lm_loss, batch)
        # projector update (only on first iter for timing)
        if bs == 1:
            _, gs = find_galore_state(optimizer.opt_state)
            p = nnx.state(model, nnx.Param)
            new_gs = update_galore_projectors(gs, grads, p, r=128)
            optimizer.opt_state = set_galore_state(optimizer.opt_state, galore_idx, new_gs)
        optimizer.update(model, grads)
        jax.block_until_ready(loss)
        elapsed = time.time() - t0
        used, total = get_memory_per_device()
        print(f"  batch_size={bs}: {used/1e9:.3f} GB / {total/1e9:.2f} GB  ✓  ({elapsed:.1f}s)")
        del loss, grads, batch
    except Exception as e:
        print(f"  batch_size={bs}: FAILED — {e}")
