
import jax, jax.numpy as jnp
from flax import nnx

def get_act_fn(act):
    if callable(act) or not isinstance(act, str):
        return act
    return getattr(jax.nn, act)


def count_params(model) -> int:
    """Count total number of parameters in a model."""
    _, state = nnx.split(model)
    leaves = jax.tree.leaves(state)
    return sum(p.size for p in leaves if hasattr(p, 'size'))


def param_bytes(model, dtype=jnp.float32) -> int:
    """Estimate model memory in bytes for a given dtype."""
    return count_params(model) * jnp.dtype(dtype).itemsize