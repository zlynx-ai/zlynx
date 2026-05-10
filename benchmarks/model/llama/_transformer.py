

import jax
from jax.experimental import pallas as pl 
from flax import nnx
from zlynx.model.meta.llama.model import LlamaModel
from zlynx.module import Attention, MLP, RMSNorm
from zlynx.module.block import create_module_from_config


class LlamaBlock(nnx.Module):
    def __init__(
        self, config, *, 
        rngs: nnx.Rngs, layer_idx: int | None = None
    ):

        hidden_size = config.hidden_size
        norm_eps = config.norm_eps

        self.self_attention = create_module_from_config(
            Attention, config, rngs=rngs, layer_idx=layer_idx
        )

        self.mlp = create_module_from_config(
            MLP, config, rngs=rngs
        )

        self.input_layernorm = RMSNorm(hidden_size, norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, norm_eps)

        def _block_kernel(x_ref, input_layernorm, o_ref):
            o_ref[...] = input_layernorm(x_ref[...])

        self._block_kernel = _block_kernel
        self.is_cpu = jax.devices()[0].platform == "cpu"

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array,
        position_embedding: tuple[jax.Array],
        past_key_value: tuple | None = None,
    ):
        residual = hidden_states
        hidden_states = pl.pallas_call(
            self._block_kernel, 
            out_shape=jax.ShapeDtypeStruct(hidden_states.shape, hidden_states.dtype),
            interpret=self.is_cpu,
        )(hidden_states, self.input_layernorm)
        # hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value = self.self_attention(
            hidden_states, attention_mask, position_embedding, past_key_value
        )
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual
        return hidden_states, present_key_value
    

class LlamaBlock_N(nnx.Module):
    def __init__(
        self, config, *, 
        rngs: nnx.Rngs, layer_idx: int | None = None
    ):

        hidden_size = config.hidden_size
        norm_eps = config.norm_eps

        self.self_attention = create_module_from_config(
            Attention, config, rngs=rngs, layer_idx=layer_idx
        )

        self.mlp = create_module_from_config(
            MLP, config, rngs=rngs
        )

        self.input_layernorm = RMSNorm(hidden_size, norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, norm_eps)

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array,
        position_embedding: tuple[jax.Array],
        past_key_value: tuple | None = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value = self.self_attention(
            hidden_states, attention_mask, position_embedding, past_key_value
        )
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual
        return hidden_states, present_key_value
