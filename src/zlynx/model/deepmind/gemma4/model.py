from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from ....core.base import Z
from ....core.inferences import LanguageModel
from ....core.outputs import CausalLMOutput
from ....module.norm import RMSNorm
from ....module.mlp import MLP
from ....module.rope import compute_rope, apply_partial_rope
from .config import Gemma4Config


K_MASK = -2.3819763e38


class Gemma4Embedder(nnx.Module):
    def __init__(self, vocab_size: int, embed_dim: int, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, *, rngs: nnx.Rngs):
        self.embed_dim = embed_dim
        self.embed_tokens = nnx.Embed(vocab_size, embed_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def encode(self, tokens: jax.Array) -> jax.Array:
        x = self.embed_tokens(tokens)
        return x * jnp.sqrt(self.embed_dim).astype(x.dtype)

    def decode(self, x: jax.Array) -> jax.Array:
        return x @ self.embed_tokens.embedding.value.T


class Gemma4Attention(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        attn_type: str,                      # "local" or "global"
        rope_proportion: float,
        base_frequency: int,
        sliding_window_size: int | None = None,
        attn_logits_softcap: float | None = None,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.attn_type = attn_type
        self.rope_proportion = rope_proportion
        self.base_frequency = base_frequency
        self.sliding_window_size = sliding_window_size
        self.attn_logits_softcap = attn_logits_softcap

        q_key, k_key, v_key, o_key = jax.random.split(rngs.params(), 4)

        self.q_proj = nnx.Linear(embed_dim, num_heads * head_dim, use_bias=False,
                                  dtype=dtype, param_dtype=param_dtype, rngs=nnx.Rngs(q_key))
        self.k_proj = nnx.Linear(embed_dim, num_kv_heads * head_dim, use_bias=False,
                                  dtype=dtype, param_dtype=param_dtype, rngs=nnx.Rngs(k_key))
        self.v_proj = nnx.Linear(embed_dim, num_kv_heads * head_dim, use_bias=False,
                                  dtype=dtype, param_dtype=param_dtype, rngs=nnx.Rngs(v_key))
        self.o_proj = nnx.Linear(num_heads * head_dim, embed_dim, use_bias=False,
                                  dtype=dtype, param_dtype=param_dtype, rngs=nnx.Rngs(o_key))

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def __call__(
        self,
        x: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
    ) -> jax.Array:
        B, S, _ = x.shape

        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, S, self.num_kv_heads, self.head_dim)

        # QK norm (per head)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        cos, sin = compute_rope(positions, self.head_dim, self.base_frequency)
        q, k = apply_partial_rope(q, k, cos, sin, self.rope_proportion)

        # GQA: repeat KV heads to match query heads
        if self.num_heads > self.num_kv_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, num_groups, axis=2)
            v = jnp.repeat(v, num_groups, axis=2)

        # (B, S, num_heads, S) attention logits
        # q: (B, S, H, D), k: (B, S, H, D)
        logits = jnp.einsum('bihd,bjhd->bhij', q, k)
        scale = jnp.sqrt(self.head_dim).astype(logits.dtype)
        logits = logits / scale

        if self.attn_logits_softcap is not None:
            logits = jnp.tanh(logits / self.attn_logits_softcap) * self.attn_logits_softcap

        # Build mask: causal + optional sliding window
        mask = attention_mask[:, None, :, :]  # (B, 1, S, S)
        if self.attn_type == "local" and self.sliding_window_size is not None:
            q_idx = jnp.arange(S)[None, None, :, None]
            k_idx = jnp.arange(S)[None, None, None, :]
            window_mask = (k_idx >= q_idx - self.sliding_window_size + 1)
            mask = mask & window_mask

        logits = jnp.where(mask, logits, K_MASK)
        probs = jax.nn.softmax(logits, axis=-1).astype(v.dtype)

        # (B, S, H, D)
        out = jnp.einsum('bhij,bjhd->bihd', probs, v)
        out = out.reshape(B, S, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Gemma4Block(nnx.Module):
    def __init__(
        self,
        config: Gemma4Config,
        attn_type: str,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        *,
        rngs: nnx.Rngs,
    ):
        attn_key, mlp_key = jax.random.split(rngs.params(), 2)

        # Local vs global attention params
        if attn_type == "global":
            num_kv_heads = config.num_global_kv_heads or config.num_kv_heads
            head_dim = config.global_key_size
            rope_proportion = config.global_rope_proportion
            base_frequency = config.global_base_frequency
        else:
            num_kv_heads = config.num_kv_heads
            head_dim = config.head_dim
            rope_proportion = config.local_rope_proportion
            base_frequency = config.local_base_frequency

        self.pre_attention_norm = RMSNorm(config.embed_dim)
        self.self_attn = Gemma4Attention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            attn_type=attn_type,
            rope_proportion=rope_proportion,
            base_frequency=base_frequency,
            sliding_window_size=config.sliding_window_size if attn_type == "local" else None,
            attn_logits_softcap=config.attn_logits_softcap,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=nnx.Rngs(attn_key),
        )
        self.post_attention_norm = RMSNorm(config.embed_dim) if config.use_post_attn_norm else None

        self.pre_ffw_norm = RMSNorm(config.embed_dim)
        self.mlp = MLP(
            mlp_key,
            hidden_size=config.embed_dim,
            intermediate_dize=config.hidden_dim,
            act_fn=jax.nn.gelu,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.post_ffw_norm = RMSNorm(config.embed_dim) if config.use_post_ffw_norm else None

        self.skip_scale = nnx.Param(jnp.ones((1,), dtype=param_dtype))

    def __call__(self, x: jax.Array, positions: jax.Array, attention_mask: jax.Array) -> jax.Array:
        # Attention
        residual = x
        x = self.pre_attention_norm(x)
        x = self.self_attn(x, positions, attention_mask)
        if self.post_attention_norm is not None:
            x = self.post_attention_norm(x)
        x = x + residual

        # FFW
        residual = x
        x = self.pre_ffw_norm(x)
        x = self.mlp(x)
        if self.post_ffw_norm is not None:
            x = self.post_ffw_norm(x)
        x = x + residual

        # Skip scale
        x = x * self.skip_scale.value

        return x


class Gemma4Model(nnx.Module):
    def __init__(self, config: Gemma4Config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, *, rngs: nnx.Rngs):
        embed_key, *block_keys = jax.random.split(rngs.params(), config.num_layers + 1)

        self.embedder = Gemma4Embedder(config.vocab_size, config.embed_dim,
                                        dtype=dtype, param_dtype=param_dtype, rngs=nnx.Rngs(embed_key))

        attn_types = config.attention_types()
        self.layers = nnx.List([
            Gemma4Block(config, attn_type=attn_types[i], dtype=dtype, param_dtype=param_dtype, rngs=nnx.Rngs(block_keys[i]))
            for i in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.embed_dim)

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
    ) -> jax.Array:
        B, S = input_ids.shape

        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(S)[None, :], (B, S))

        if attention_mask is None:
            attention_mask = jnp.tril(jnp.ones((B, S, S), dtype=bool))
        else:
            # Combine user mask with causal mask
            causal = jnp.tril(jnp.ones((S, S), dtype=bool))[None, :, :]
            attention_mask = causal & (attention_mask[:, None, :] > 0)

        x = self.embedder.encode(input_ids)

        for layer in self.layers:
            x = layer(x, position_ids, attention_mask)

        return self.norm(x)


class Gemma4LanguageModel(LanguageModel, Z):
    def __init__(
        self,
        config: Gemma4Config,
        key: Optional[jax.Array] = None,
    ):
        nnx.Module.__init__(self)
        LanguageModel.__init__(self, config=config)

        if key is None:
            key = jax.random.key(42)

        dtype = config.dtype
        param_dtype = config.param_dtype

        self.model = Gemma4Model(config, dtype=dtype, param_dtype=param_dtype, rngs=nnx.Rngs(key))
        self.config = config

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        labels: jax.Array | None = None,
        past_key_values: object = None,
    ) -> CausalLMOutput:
        hidden = self.model(input_ids, attention_mask, position_ids)
        logits = self.model.embedder.decode(hidden)

        # Final logit softcap
        if self.config.final_logit_softcap is not None:
            logits = jnp.tanh(logits / self.config.final_logit_softcap) * self.config.final_logit_softcap

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss_all = optax.softmax_cross_entropy_with_integer_labels(
                logits=shift_logits, labels=shift_labels
            )
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:]
                loss = jnp.sum(loss_all * shift_mask) / jnp.maximum(jnp.sum(shift_mask), 1)
            else:
                loss = jnp.mean(loss_all)

        return CausalLMOutput(loss=loss, logits=logits, past_key_values=None)

    @classmethod
    def load(  # type: ignore[override]
        cls,
        path: str | Path,
        *,
        config: Gemma4Config,
        key: Optional[jax.Array] = None,
        **kwargs: Any,
    ) -> "Gemma4LanguageModel":
        from gemma.gm import ckpts as gm_ckpts  # pyright: ignore[reportMissingTypeStubs]

        if key is None:
            key = jax.random.key(42)

        model = nnx.eval_shape(lambda: cls(config=config, key=key))
        gdef, state = nnx.split(model)

        linen_params = gm_ckpts.load_params(path)

        flat_linen: dict[tuple[str, ...], Any] = {}
        for key_path, value in jax.tree_util.tree_leaves_with_path(linen_params):
            k = tuple(str(p.key) for p in key_path)  # type: ignore[union-attr]
            flat_linen[k] = value

        flat_state = dict(nnx.to_flat_state(state))
        for linen_key, value in flat_linen.items():
            nnx_key = ('_gemma',) + linen_key
            if nnx_key in flat_state:
                flat_state[nnx_key] = type(flat_state[nnx_key])(value)

        nnx.update(state, nnx.FlatState(list(flat_state.items()), sort=False).to_nested_state())
        return nnx.merge(gdef, state)
