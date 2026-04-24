
import jax
import jax.numpy as jnp
from flax import nnx

from ....core.inferences import LanguageModel
from ....module import MLP, Attention, RMSNorm, RotaryEmbedding
from ....module.block import create_block, call_block
from .config import LlamaConfig


class LlamaTransformer(nnx.Module):
    def __init__(self, config: LlamaConfig, *, rngs: nnx.Rngs, layer_idx: int | None = None):

        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        attention_head = config.attention_head
        head_dim = config.head_dim
        kv_head = config.kv_head
        attention_bias = config.attention_bias
        dtype = config.dtype
        param_dtype = config.param_dtype
        act_fn = config.act_fn
        bias = config.bias
        norm_eps = config.norm_eps

        self.self_attention = Attention(
            hidden_size, attention_head,
            head_dim, kv_head,
            rngs = rngs, bias = attention_bias,
            layer_idx = layer_idx, dtype=dtype,
            param_dtype=param_dtype,
        )
        self.mlp = MLP(
            hidden_size, intermediate_size,
            rngs = rngs, act_fn = act_fn,
            bias = bias, dtype = dtype,
            param_dtype = param_dtype,
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


class Llama(nnx.Module):
    def __init__(self, config: LlamaConfig,* , rngs: nnx.Rngs):
        
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        num_hidden_layers = config.num_hidden_layers
        dtype = config.dtype
        param_dtype = config.param_dtype
        norm_eps = config.norm_eps
        base = config.rope_theta
        head_dim = config.head_dim
        max_position_embedding = config.max_position_embedding
        rope_scaling = config.rope_scaling

        self.embed_tokens = nnx.Embed(
            vocab_size,
            hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.rotary = RotaryEmbedding(
            base,
            head_dim,
            max_position_embedding,
            rope_scaling,
        )

        self.layernorm = RMSNorm(hidden_size, norm_eps)

        self.blocks = create_block(
            num_hidden_layers, 
            LlamaTransformer, 
            module_args=(config,),
            rngs=rngs,
            in_axes=(0, 0),
            layer_idx=jnp.arange(num_hidden_layers),
        )


    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        past_key_values: list | None = None,
    ):
        if position_ids is None:
            if attention_mask is not None:
                position_ids = jnp.cumsum(attention_mask, axis=-1) - 1
                position_ids = jnp.maximum(position_ids, 0)
            else:
                B, S = input_ids.shape
                position_ids = jnp.expand_dims(jnp.arange(S), axis=0).repeat(B, axis=0)

        hidden_states = self.embed_tokens(input_ids)
        position_embedding = self.rotary(hidden_states, position_ids)

        if attention_mask is not None:
            q_len = input_ids.shape[1]
            q_mask = jnp.ones(input_ids.shape, dtype=jnp.bool_)
            k_mask = attention_mask > 0

            causal_mask = nnx.make_attention_mask(q_mask, k_mask)

            if q_len > 1:
                causal_mask = nnx.combine_masks(
                    causal_mask, nnx.make_causal_mask(input_ids)
                )
        else:
            causal_mask = None

        hidden_states, present_key_values = call_block(
            self.blocks,
            hidden_states,
            module_args=(causal_mask, position_embedding, past_key_values),
            return_aux=True,
        )
        return self.layernorm(hidden_states), present_key_values


class LlamaLanguageModel(LanguageModel):
    def __init__(
        self, config: LlamaConfig, *, 
        rngs: nnx.Rngs = None
    ):
        
        self.set_config(config)
        
        if rngs is None:
            rngs = nnx.Rngs(42)
        
        self.model = Llama(config, rngs=rngs)
        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            use_bias=config.bias,
            dtype=config.dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        labels: jax.Array | None = None,
        past_key_values: list | None = None,
    ):
        from ....core.outputs import CausalLMOutput
        import optax

        hidden_states, present_key_values = self.model(
            input_ids, attention_mask, position_ids, past_key_values
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]

            # Use optax softmax cross entropy
            loss_all = optax.softmax_cross_entropy_with_integer_labels(
                logits=shift_logits, labels=shift_labels
            )

            if attention_mask is not None:
                # Discard padding tokens from the loss
                shift_mask = attention_mask[..., 1:]
                loss = jnp.sum(loss_all * shift_mask) / jnp.maximum(jnp.sum(shift_mask), 1)
            else:
                loss = jnp.mean(loss_all)

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            # hidden_states=hidden_states if getattr(self.config, "output_hidden_states", False) else None,
            past_key_values=present_key_values,
        )
