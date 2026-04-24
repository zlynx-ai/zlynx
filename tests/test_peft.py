import jax
import jax.numpy as jnp
from flax import nnx
from zlynx.model.meta.llama import LlamaConfig, LlamaLanguageModel
from zlynx.module.peft import apply_peft, LoraLinear, DoraLinear

def test_lora():
    print("=== Testing LoRA ===")
    config = LlamaConfig(
        vocab_size=16, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, head_dim=8,
        dtype="bfloat16", param_dtype="float32"
    )
    model = LlamaLanguageModel(config, key=jax.random.key(0))
    
    # Check original structure
    assert isinstance(model.model.blocks[0].self_attention.q_proj, nnx.Linear)
    
    # Apply LoRA
    print("Applying LoRA to q_proj and v_proj...")
    model = apply_peft(model, method="lora", r=4, alpha=8, target_modules=["q_proj", "v_proj"])
    
    # Check new structure
    assert isinstance(model.model.blocks[0].self_attention.q_proj, LoraLinear)
    assert isinstance(model.model.blocks[0].self_attention.v_proj, LoraLinear)
    # k_proj is NOT in target_modules, so it should remain nnx.Linear
    assert isinstance(model.model.blocks[0].self_attention.k_proj, nnx.Linear)
    
    print("Structure verified.")
    
    # Check parameter status
    _, state = nnx.split(model)
    # Base kernel should be a standard Variable, not Param (so it won't be trained)
    # Lora A and B should be Params
    print("Forward pass test...")
    input_ids = jnp.ones((2, 10), dtype=jnp.int32)
    out = model(input_ids)
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 10, 16)
    print("LoRA Forward Pass OK!")


def test_dora():
    print("\n=== Testing DoRA ===")
    config = LlamaConfig(
        vocab_size=16, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, head_dim=8,
        dtype="bfloat16", param_dtype="float32"
    )
    model = LlamaLanguageModel(config, key=jax.random.key(1))
    
    print("Applying DoRA to gate_proj and up_proj...")
    model = apply_peft(model, method="dora", r=4, alpha=8, target_modules=["gate_proj", "up_proj"])
    
    assert isinstance(model.model.blocks[0].mlp.gate_proj, DoraLinear)
    assert isinstance(model.model.blocks[0].mlp.up_proj, DoraLinear)
    assert isinstance(model.model.blocks[0].mlp.down_proj, nnx.Linear)
    
    print("Structure verified.")
    
    print("Forward pass test...")
    input_ids = jnp.ones((2, 10), dtype=jnp.int32)
    out = model(input_ids)
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 10, 16)
    print("DoRA Forward Pass OK!")

def test_vera():
    print("\n=== Testing VeRA ===")
    config = LlamaConfig(
        vocab_size=16, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, head_dim=8,
        dtype="bfloat16", param_dtype="float32"
    )
    model = LlamaLanguageModel(config, key=jax.random.key(2))
    
    print("Applying VeRA to o_proj and down_proj...")
    from zlynx.module.peft import VeraLinear
    model = apply_peft(model, method="vera", r=4, alpha=8, target_modules=["o_proj", "down_proj"])
    
    assert isinstance(model.model.blocks[0].self_attention.o_proj, VeraLinear)
    assert isinstance(model.model.blocks[0].mlp.down_proj, VeraLinear)
    
    print("Structure verified.")
    print("Forward pass test...")
    input_ids = jnp.ones((2, 10), dtype=jnp.int32)
    out = model(input_ids)
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 10, 16)
    print("VeRA Forward Pass OK!")

def test_loha():
    print("\n=== Testing LoHa ===")
    config = LlamaConfig(
        vocab_size=16, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, head_dim=8,
        dtype="bfloat16", param_dtype="float32"
    )
    model = LlamaLanguageModel(config, key=jax.random.key(3))
    
    print("Applying LoHa to q_proj and v_proj...")
    from zlynx.module.peft import LohaLinear
    model = apply_peft(model, method="loha", r=4, alpha=8, target_modules=["q_proj", "v_proj"])
    
    assert isinstance(model.model.blocks[0].self_attention.q_proj, LohaLinear)
    
    print("Structure verified.")
    print("Forward pass test...")
    input_ids = jnp.ones((2, 10), dtype=jnp.int32)
    out = model(input_ids)
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 10, 16)
    print("LoHa Forward Pass OK!")


def test_lokr():
    print("\n=== Testing LoKr ===")
    config = LlamaConfig(
        vocab_size=16, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, head_dim=8,
        dtype="bfloat16", param_dtype="float32"
    )
    model = LlamaLanguageModel(config, key=jax.random.key(4))
    
    print("Applying LoKr to gate_proj and up_proj...")
    from zlynx.module.peft import LokrLinear
    model = apply_peft(model, method="lokr", r=4, alpha=8, target_modules=["gate_proj", "up_proj"])
    
    assert isinstance(model.model.blocks[0].mlp.gate_proj, LokrLinear)
    
    print("Structure verified.")
    print("Forward pass test...")
    input_ids = jnp.ones((2, 10), dtype=jnp.int32)
    out = model(input_ids)
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 10, 16)
    print("LoKr Forward Pass OK!")

def test_adalora():
    print("\n=== Testing AdaLoRA ===")
    config = LlamaConfig(
        vocab_size=16, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, head_dim=8,
        dtype="bfloat16", param_dtype="float32"
    )
    model = LlamaLanguageModel(config, key=jax.random.key(5))
    
    print("Applying AdaLoRA to o_proj and down_proj...")
    from zlynx.module.peft import AdaloraLinear
    model = apply_peft(model, method="adalora", r=4, alpha=8, target_modules=["o_proj", "down_proj"])
    
    assert isinstance(model.model.blocks[0].self_attention.o_proj, AdaloraLinear)
    assert isinstance(model.model.blocks[0].mlp.down_proj, AdaloraLinear)
    
    print("Structure verified.")
    print("Forward pass test...")
    input_ids = jnp.ones((2, 10), dtype=jnp.int32)
    out = model(input_ids)
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 10, 16)
    print("AdaLoRA Forward Pass OK!")


if __name__ == "__main__":
    test_lora()
    test_dora()
    test_vera()
    test_loha()
    test_lokr()
    test_adalora()
    print("\nALL PEFT ARCHITECTURE TESTS PASSED! 🎉")
