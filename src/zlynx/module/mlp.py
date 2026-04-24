

import jax, jax.numpy as jnp
from flax import nnx

from ..utils import get_act_fn


class MLP(nnx.Module):
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_dize: int, 
        *, rngs: nnx.Rngs,
        act_fn = None, 
        bias: bool = False, 
        dtype = "bfloat16",
        param_dtype = "float32"
    ):
        if act_fn is None:
            act_fn = "silu"

        self.gate_proj = nnx.Linear(
            hidden_size, intermediate_dize, 
            use_bias=bias, dtype=dtype, 
            param_dtype=param_dtype,
            rngs=rngs
        )
        self.up_proj = nnx.Linear(
            hidden_size, intermediate_dize, 
            use_bias=bias, dtype=dtype, 
            param_dtype=param_dtype,
            rngs=rngs
        )
        self.down_proj = nnx.Linear(
            intermediate_dize, hidden_size, 
            use_bias=bias, dtype=dtype, 
            param_dtype=param_dtype,
            rngs=rngs
        )
        self.act_fn = get_act_fn(act_fn)

    def __call__(self, hidden_states: jax.Array):
        hidden_states = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(hidden_states)