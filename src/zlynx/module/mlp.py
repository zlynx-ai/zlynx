

import jax, jax.numpy as jnp
from flax import nnx

from ..utils import get_act_fn


class MLP(nnx.Module):
    def __init__(
        self, key, 
        hidden_size: int, 
        intermediate_dize: int, 
        act_fn=jax.nn.silu, 
        bias: bool=False, 
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32
    ):
        super().__init__()
        gate_key, up_key, down_key = jax.random.split(key, 3)
        self.gate_proj = nnx.Linear(
            hidden_size, intermediate_dize, 
            use_bias=bias, dtype=dtype, 
            param_dtype=param_dtype,
            rngs=nnx.Rngs(gate_key)
        )
        self.up_proj = nnx.Linear(
            hidden_size, intermediate_dize, 
            use_bias=bias, dtype=dtype, 
            param_dtype=param_dtype,
            rngs=nnx.Rngs(up_key)
        )
        self.down_proj = nnx.Linear(
            intermediate_dize, hidden_size, 
            use_bias=bias, dtype=dtype, 
            param_dtype=param_dtype,
            rngs=nnx.Rngs(down_key)
        )
        self.act_fn = get_act_fn(act_fn)

    def __call__(self, hidden_states: jax.Array):
        hidden_states = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(hidden_states)