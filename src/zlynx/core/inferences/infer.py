




from flax import nnx
import functools
import jax, jax.numpy as jnp



@functools.partial(
    jax.jit, static_argnames=("temperature", "top_k", "top_p", "repetition_penalty")
)
def sample_token(
    logits, input_ids, input_mask, key, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0
):
    if temperature == 0.0:
        return jax.lax.argmax(logits, axis=1, index_dtype=jnp.int32), key

    logits = logits / temperature

    # Repetition Penalty
    if repetition_penalty != 1.0:
        one_hots = jax.nn.one_hot(input_ids, logits.shape[-1])
        valid_one_hots = jnp.where(jnp.expand_dims(input_mask, -1), one_hots, 0.0)
        score_mask = valid_one_hots.any(axis=1)
        penalized_logits = jnp.where(
            logits > 0, logits / repetition_penalty, logits * repetition_penalty
        )
        logits = jnp.where(score_mask, penalized_logits, logits)

    # Top-K Sampling
    if top_k > 0:
        top_k_vals, _ = jax.lax.top_k(logits, top_k)
        min_vals = top_k_vals[:, -1:]
        logits = jnp.where(logits < min_vals, -jnp.inf, logits)

    # Nucleus (Top-P) Sampling
    if top_p < 1.0:
        sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)

        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

        mask = cumulative_probs > top_p

        mask = jnp.roll(mask, 1, axis=-1)
        mask = mask.at[:, 0].set(False)  # Always keep at least the most probable token

        inv_indices = jnp.argsort(sorted_indices, axis=-1)
        mask_in_original_order = jnp.take_along_axis(mask, inv_indices, axis=-1)

        logits = jnp.where(mask_in_original_order, -jnp.inf, logits)

    key, subkey = jax.random.split(key)
    result = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
    return result, key


@nnx.jit
def _decode_step(model, input_ids, attention_mask, position_ids, past_key_values):
    return model(input_ids, attention_mask, position_ids, past_key_values=past_key_values)


class LanguageModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.config = kwargs.get("config", None)

    def generate(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array = None,
        key: jax.Array | None = None,
        max_new_tokens: int = 64,
        ctxlen: int = 2048,
        batch: int | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
        suppress_tokens: list[int] | None = None,
    ):
        B, S = input_ids.shape
        if batch is None:
            batch = B

        cfg = getattr(self, "config", getattr(self, "kwargs", {}).get("config", None))

        if attention_mask is None:
            attention_mask = jnp.ones((batch, S), dtype=jnp.bool_)
        else:
            attention_mask = attention_mask.astype(jnp.bool_)

        if key is None:
            temperature = 0.0
            key = jax.random.key(0)

        # Init past key values: list of (k_cache, v_cache, cache_index) per layer
        num_layers = cfg.num_hidden_layers
        kv_head = cfg.kv_head if cfg.kv_head else cfg.attention_head
        head_dim = cfg.head_dim
        from ...utils import get_dtype
        cache_dtype = get_dtype(cfg.dtype) if cfg.dtype else jnp.bfloat16

        cache_index = jnp.int32(0)
        past_key_values = [
            (
                jnp.zeros((batch, ctxlen, kv_head, head_dim), dtype=cache_dtype),
                jnp.zeros((batch, ctxlen, kv_head, head_dim), dtype=cache_dtype),
                cache_index,
            )
            for _ in range(num_layers)
        ]

        out_ids = jnp.zeros((batch, ctxlen), dtype=jnp.int32)
        out_ids = out_ids.at[:, :S].set(input_ids.astype(jnp.int32))
        out_mask = jnp.zeros((batch, ctxlen), dtype=jnp.bool_)
        out_mask = out_mask.at[:, :S].set(attention_mask)

        finished = jnp.zeros((batch,), dtype=jnp.bool_)

        # Build suppress mask
        suppress_mask = None
        if suppress_tokens:
            vocab_size = cfg.vocab_size
            suppress_mask = jnp.zeros(vocab_size, dtype=jnp.bool_)
            for t in suppress_tokens:
                suppress_mask = suppress_mask.at[t].set(True)

        # ── Prefill: direct call (no JIT) to populate past key values ──
        prompt_position_ids = jnp.cumsum(attention_mask.astype(jnp.int32), axis=-1) - 1
        prompt_position_ids = jnp.maximum(prompt_position_ids, 0)
        outputs = self(input_ids, attention_mask, prompt_position_ids, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        last_logit = outputs.logits[:, -1, :]

        # ── Decode: JIT-compiled steps with past key values ──
        for i in range(max_new_tokens):
            cur_logit = last_logit
            if suppress_mask is not None:
                cur_logit = jnp.where(suppress_mask, -1e9, cur_logit)

            next_token, key = sample_token(
                cur_logit, out_ids, out_mask,
                key, temperature, top_k, top_p, repetition_penalty
            )

            if eos_token_id is not None:
                next_token = jnp.where(finished, eos_token_id, next_token)
                finished = finished | (next_token == eos_token_id)
                if jnp.all(finished):
                    out_ids = out_ids.at[:, S + i].set(next_token.astype(jnp.int32))
                    break

            out_ids = out_ids.at[:, S + i].set(next_token.astype(jnp.int32))
            out_mask = out_mask.at[:, S + i].set(True)

            next_token_2d = jnp.expand_dims(next_token.astype(jnp.int32), axis=1)
            decode_pos = jnp.expand_dims(
                jnp.sum(out_mask.astype(jnp.int32), axis=-1) - 1, axis=-1
            )

            outputs = _decode_step(self, next_token_2d, out_mask, decode_pos, past_key_values)
            past_key_values = outputs.past_key_values
            last_logit = outputs.logits[:, -1, :]

        return out_ids


class DiffusionModel:
    def __init__(self):
        pass