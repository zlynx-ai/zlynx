from flax import struct
from typing import Optional, Dict
import jax

@struct.dataclass
class ModelOutput:
    """
    Base class for all model outputs as data classes.

    Args:
        loss (`jax.Array | float`, optional):
            Scalar loss value.

        hidden_states (`tuple[jax.Array, ...]`, optional):
            Tuple of hidden states from the model.

        auxiliary (`Dict`, optional):
            Dictionary of arbitrary extra outputs for flexibility.
            Can contain any additional values the model may want to return.
    """
    loss: Optional[jax.Array] = None
    hidden_states: Optional[tuple[jax.Array, ...]] = None
    auxiliary: Optional[Dict] = None

@struct.dataclass
class CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`jax.Array | float`, optional):
            Scalar loss value.

        logits (`jax.Array`, optional):
            Token prediction logits, typically of shape
            `(batch_size, sequence_length, vocab_size)`.

        hidden_states (`tuple[jax.Array, ...]`, optional):
            Tuple of hidden states from the model.

        past_key_values (`list`, optional):
            Cache of past key/value states used for autoregressive decoding.

        auxiliary (`Dict`, optional):
            Dictionary of arbitrary extra outputs for flexibility.
            Can contain any additional values the model may want to return.
    """
    logits: Optional[jax.Array] = None
    past_key_values: Optional[list] = None
