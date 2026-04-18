


from flax import struct



@struct.dataclass
class Config:
    arch: str | None = None
    conf: str | None = None
  

@struct.dataclass
class LanguageConfig(Config):
    vocab_size: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    act_fn: str | None = None
    num_hidden_layers: int | None = None
    norm_eps: float | None = None
    bias: bool | None = None
    dtype: str | None = None
    param_dtype: str | None = None
    use_cache: bool | None = None

    # attn
    attention_head: int | None = None
    kv_head: int | None = None
    head_dim: int | None = None
    attention_bias: bool | None = None

    # rope
    base: float | None = None
    original_max_position_embedding: int | None = None
    max_position_embedding: int | None = None
    rope_scaling: dict | None = None

