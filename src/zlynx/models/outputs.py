from dataclasses import dataclass
from typing import Optional, Any
import jax

@dataclass
class ModelOutput:
    """
    Base class for all model outputs as data classes.
    Mimics Hugging Face Transformers output formats.
    """
    pass

@dataclass
class CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`jax.Array` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`jax.Array` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(jax.Array)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jax.Array` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
        past_key_values (`list`, *optional*):
            List of `(key_cache, value_cache, cache_index)` tuples per layer.
    """
    loss: Optional[jax.Array] = None
    logits: Optional[jax.Array] = None
    # hidden_states: Optional[tuple[jax.Array, ...]] = None
    past_key_values: Optional[list] = None

    def tree_flatten(self):
        # return ((self.loss, self.logits, self.hidden_states, self.past_key_values), None)
        return ((self.loss, self.logits, self.past_key_values), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

jax.tree_util.register_pytree_node(
    CausalLMOutput,
    CausalLMOutput.tree_flatten,
    CausalLMOutput.tree_unflatten
)
