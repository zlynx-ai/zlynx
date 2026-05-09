import jax
import jax.numpy as jnp
import optax
from zlynx.trainer.optim import build_optimizer
from zlynx.trainer.trainer import TrainerConfig

def test_galore():
    print("=== Testing GaLore ===")
    
    # 1. Setup dummy parameters: one large matrix (target for GaLore) and one small bias (ignored by GaLore)
    params = {
        "large_w_right": jnp.ones((10, 20)),  # m < n (right projection)
        "large_w_left": jnp.ones((20, 10)),   # m > n (left projection)
        "small_bias": jnp.ones((10,))         # 1D (ignored)
    }
    
    # 2. Configure GaLore optimizer
    config = TrainerConfig(
        optimizer="galore_adamw",
        learning_rate=1e-3,
        weight_decay=0.01,
        optimizer_kwargs={"galore_r": 4, "galore_update_proj_gap": 2} # Update proj every 2 steps
    )
    
    opt, schedule = build_optimizer(config, total_steps=10)
    
    # 3. Init optimizer
    print("Initializing optimizer state...")
    state = opt.init(params)
    
    # Check that inner state shapes are reduced for large matrices
    # state is ChainState -> (ScaleByAdamState, AddDecayedWeightsState, ScaleByScheduleState) inside GaloreState
    # Let's just run an update to see if it crashes
    
    # 4. Mock gradients
    grads = {
        "large_w_right": jax.random.normal(jax.random.key(1), (10, 20)),
        "large_w_left": jax.random.normal(jax.random.key(2), (20, 10)),
        "small_bias": jax.random.normal(jax.random.key(3), (10,))
    }
    
    # 5. Step 0 (Projection computed)
    print("Step 0 Update...")
    updates, state = opt.update(grads, state, params)
    
    assert updates["large_w_right"].shape == (10, 20)
    assert updates["large_w_left"].shape == (20, 10)
    assert updates["small_bias"].shape == (10,)
    
    # 6. Step 1 (Projection reused)
    print("Step 1 Update...")
    updates, state = opt.update(grads, state, params)
    
    # 7. Step 2 (Projection recomputed)
    print("Step 2 Update...")
    updates, state = opt.update(grads, state, params)
    
    print("GaLore Optimization passed all shape checks!")

if __name__ == "__main__":
    test_galore()
