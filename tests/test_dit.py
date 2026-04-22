import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from zlynx.model.dit.config import DiTConfig
from zlynx.model.dit.model import DiT

def test_dit_forward():
    print("\n=== Testing DiT Forward Pass ===")
    
    # Smaller configuration for testing
    config = DiTConfig(
        img_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=128,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        frequency_embedding_size=64
    )
    
    # Initialize model
    rngs = nnx.Rngs(jax.random.key(42))
    model = DiT(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
    
    # Create random inputs: Batch size 2, 32x32 RGB image
    batch_size = 2
    x = jax.random.normal(rngs.dropout(), (batch_size, 32, 32, 3))
    
    # Timesteps (e.g. 1000 steps max, random values between 0-999)
    t = jnp.array([10.5, 500.2], dtype=jnp.float32)
    
    # Forward pass
    out = model(x, t)
    
    # Validate output shape
    # Number of patches = (32 // 4) ** 2 = 8 ** 2 = 64
    # Output dim per patch = patch_size * patch_size * in_channels = 4 * 4 * 3 = 48
    expected_shape = (batch_size, 64, 48)
    
    print(f"Input image shape: {x.shape}")
    print(f"Timestep shape: {t.shape}")
    print(f"Output un-noised shape: {out.shape}")
    
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print("Test passed successfully!")

if __name__ == "__main__":
    test_dit_forward()
