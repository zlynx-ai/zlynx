
import jax
import jax.numpy as jnp
from flax import nnx

from ...core.base import Z
from ...core.inferences import LanguageModel
from ...modules import MLP, Attention, RMSNorm, RotaryEmbedding
from ...utils import get_dtype, get_act_fn
from .config import LlamaConfig


class LlamaTransformer(nnx.Module):
    def __init__(self, key, config: LlamaConfig, layer_idx: int | None = None):
        super().__init__()
        attention_key, mlp_key = jax.random.split(key, 2)

        self.self_attention = Attention(
            attention_key,
            config.hidden_size,
            config.attention_head,
            config.head_dim,
            config.kv_head,
            config.attention_bias,
            layer_idx,
            dtype=get_dtype(config.dtype),
            param_dtype=get_dtype(config.param_dtype),
        )
        self.mlp = MLP(
            mlp_key,
            config.hidden_size,
            config.intermediate_size,
            get_act_fn(config.act_fn),
            config.bias,
            dtype=get_dtype(config.dtype),
            param_dtype=get_dtype(config.param_dtype),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.norm_eps)

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
    def __init__(self, config: LlamaConfig, key):
        super().__init__()
        self.num_hidden_layers = config.num_hidden_layers
        embedding_key, transformer_key = jax.random.split(key, 2)
        self.embed_tokens = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=get_dtype(config.dtype),
            param_dtype=get_dtype(config.param_dtype),
            rngs=nnx.Rngs(embedding_key),
        )

        self.rotary = RotaryEmbedding(
            config.base,
            config.head_dim,
            config.max_position_embedding,
            config.rope_scaling,
        )

        self.layernorm = RMSNorm(config.hidden_size, config.norm_eps)

        self.blocks = nnx.List(
            [
                LlamaTransformer(key, config, layer_idx)
                for layer_idx, key in enumerate(
                    jax.random.split(transformer_key, config.num_hidden_layers)
                )
            ]
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

        present_key_values = []
        for idx, layer in enumerate(self.blocks[: self.num_hidden_layers]):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present_kv = layer(hidden_states, causal_mask, position_embedding, layer_past)
            present_key_values.append(present_kv)

        return self.layernorm(hidden_states), present_key_values


class LlamaLanguageModel(LanguageModel, Z):
    def __init__(
        self, config: LlamaConfig, key: jax.typing.ArrayLike = jax.random.key(42)
    ):
        nnx.Module.__init__(self)
        LanguageModel.__init__(self, config=config)
        model_key, lm_head_key = jax.random.split(key, 2)
        self.model = Llama(config=config, key=model_key)
        self.lm_head = nnx.Linear(
            config.hidden_size,
            config.vocab_size,
            use_bias=config.bias,
            dtype=get_dtype(config.dtype),
            param_dtype=get_dtype(config.param_dtype),
            rngs=nnx.Rngs(lm_head_key),
        )

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        labels: jax.Array | None = None,
        past_key_values: list | None = None,
    ):
        from zlynx.core.outputs import CausalLMOutput
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
